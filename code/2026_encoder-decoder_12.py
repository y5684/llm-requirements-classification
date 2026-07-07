MODEL_NAME_OR_PATH = "flan-t5-base" # choices: "flan-t5-base", "flan-t5-large", "t5-base", "t5-large", "bart-base", "bart-large"

TUNING_METHOD = "full" # choose from ["full", "lora", "prefix", "prompt", "ptuning"]

TRAIN_FILE = "multi/train.csv"
VAL_FILE   = "multi/val.csv"
TEST_FILE  = "multi/test.csv"

TEXT_COL = "Requirement"
TYPE_COL = "Specific_Type"

LABEL_LIST = ["FR","A","L","LF","MN","O","PE","SC","SE","US","FT","PO"]
NUM_LABELS = 12

EPOCHS = 8
SEED   = 42

LOCAL_FILES_ONLY  = True
TRUST_REMOTE_CODE = False

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 4

PER_DEVICE_TRAIN_BS = 2
PER_DEVICE_EVAL_BS  = 8
GRAD_ACCUM_STEPS    = 16
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
LR_SCHEDULER        = "linear"
MAX_GRAD_NORM       = 1.0

GEN_MAX_NEW_TOKENS  = 4
GEN_NUM_BEAMS       = 1

USE_CLASS_WEIGHTS   = False

ENABLE_BALANCED_SAMPLING = (TUNING_METHOD in ["prompt", "ptuning"])

PREFIX_VIRTUAL_TOKENS  = 30
PROMPT_VIRTUAL_TOKENS  = 12
PTUNING_VIRTUAL_TOKENS = 128

LR_FULL    = 1e-5
LR_LORA    = 7e-5
LR_PREFIX  = 1e-3
LR_PROMPT  = 1e-3
LR_PTUNING = 5e-4

INSTRUCTION = (
    "You are classifying a software requirement into exactly one of these labels:\n"
    "FR, A, L, LF, MN, O, PE, SC, SE, US, FT, PO.\n"
    "Output only the label, without explanation.\n"
    "Requirement: "
)

from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)

from peft import (
    TaskType,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    get_peft_model,
    PromptEncoderReparameterizationType,
    PromptTuningInit,
)

label2id = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
id2label = {i: lbl for lbl, i in label2id.items()}

def normalize_label(x: str) -> str:
    return str(x).strip().upper()

def parse_generated_label(text: str) -> int:
    s = str(text).strip().upper().replace("\n", " ").strip()
    if not s:
        return -1

    first = s.split()[0].strip(".,:;!\"'()[]{}<>")

    for lbl in sorted(LABEL_LIST, key=len, reverse=True):
        if first == lbl:
            return label2id[lbl]

    compact = first.replace("-", "").replace("_", "")
    for lbl in sorted(LABEL_LIST, key=len, reverse=True):
        if compact == lbl:
            return label2id[lbl]

    for lbl in sorted(LABEL_LIST, key=len, reverse=True):
        if lbl in s:
            return label2id[lbl]

    return -1

def detect_t5_lora_targets(model) -> list:
    module_names = [n for n, _ in model.named_modules()]
    has_q = any(n.endswith(".q") or n.endswith("q") for n in module_names)
    has_v = any(n.endswith(".v") or n.endswith("v") for n in module_names)
    has_k = any(n.endswith(".k") or n.endswith("k") for n in module_names)
    has_o = any(n.endswith(".o") or n.endswith("o") for n in module_names)

    if has_q and has_v:
        return ["q", "v"]

    cands = []
    for name, ok in [("q", has_q), ("k", has_k), ("v", has_v), ("o", has_o)]:
        if ok:
            cands.append(name)
    return cands if cands else None

def compute_class_weights_multiclass(labels: list, num_labels: int) -> torch.Tensor:
    labels = [int(x) for x in labels]
    counts = Counter(labels)
    total = sum(counts.get(i, 0) for i in range(num_labels))
    weights = []
    for i in range(num_labels):
        c = counts.get(i, 0)
        w = (total / (num_labels * c)) if c > 0 else 0.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)

class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "class_id" in inputs:
            inputs = dict(inputs)
            inputs.pop("class_id")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        class_id = inputs.get("class_id", None)
        labels = inputs.get("labels")

        forward_inputs = {k: v for k, v in inputs.items() if k != "class_id"}
        outputs = model(**forward_inputs)

        if labels is None or labels.size(1) < 2:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits.float()
        vocab = logits.size(-1)
        bs, tgt_len = labels.size()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        active = shift_labels != -100
        flat_logits = shift_logits.view(-1, vocab)
        flat_labels = shift_labels.view(-1)

        ce = F.cross_entropy(
            flat_logits, flat_labels,
            reduction="none",
            ignore_index=-100
        ).view(bs, tgt_len - 1)

        denom = active.sum(dim=1).clamp(min=1)
        per_ex_loss = (ce * active.float()).sum(dim=1) / denom.float()

        if class_id is None:
            loss = per_ex_loss.mean()
        else:
            w = self.class_weights.to(per_ex_loss.device)
            loss = (per_ex_loss * w[class_id]).mean()

        if not torch.isfinite(loss):
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

class BalancedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, balance_sampling: bool = False, num_labels: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.balance_sampling = balance_sampling
        self.num_labels = num_labels

    def get_train_dataloader(self):
        if not self.balance_sampling:
            return super().get_train_dataloader()

        labels = np.array(self.train_dataset["class_id"], dtype=int)
        counts = np.bincount(labels, minlength=self.num_labels)
        class_w = 1.0 / np.maximum(counts, 1)
        sample_w = class_w[labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.double),
            num_samples=len(sample_w),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "class_id" in inputs:
            inputs = dict(inputs)
            inputs.pop("class_id")
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if "class_id" in inputs:
            inputs = dict(inputs)
            inputs.pop("class_id")
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

def compute_metrics_from_generations(eval_pred, tokenizer):
    preds, labels = eval_pred

    if isinstance(preds, (tuple, list)):
        preds = preds[0]

    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = np.asarray(preds).astype(np.int64)

    labels = np.asarray(labels)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels).astype(np.int64)

    if tokenizer.vocab_size is not None and preds.size > 0:
        max_id = int(tokenizer.vocab_size) - 1
        preds = np.clip(preds, 0, max_id)
        labels = np.clip(labels, 0, max_id)

    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    pred_y = np.array([parse_generated_label(t) for t in pred_texts], dtype=int)
    gold_y = np.array([label2id.get(normalize_label(t), -1) for t in gold_texts], dtype=int)

    valid = gold_y != -1
    pred_y = pred_y[valid]
    gold_y = gold_y[valid]

    unknown = (pred_y == -1)
    pred_y_fixed = pred_y.copy()
    pred_y_fixed[unknown] = 0

    return {
        "accuracy": accuracy_score(gold_y, pred_y_fixed),
        "f1_macro": f1_score(gold_y, pred_y_fixed, average="macro", zero_division=0),
        "f1_weighted": f1_score(gold_y, pred_y_fixed, average="weighted", zero_division=0),
        "precision_macro": precision_score(gold_y, pred_y_fixed, average="macro", zero_division=0),
        "recall_macro": recall_score(gold_y, pred_y_fixed, average="macro", zero_division=0),
        "unknown_rate": float(unknown.mean()) if len(unknown) else 0.0,
    }

set_seed(SEED)

datasets = load_dataset(
    "csv",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE, "test": TEST_FILE}
)

def add_fields(ex):
    t = normalize_label(ex[TYPE_COL])
    if t not in label2id:
        raise ValueError(f"Unknown label '{t}'. Expected one of: {LABEL_LIST}")
    y = label2id[t]
    return {
        "class_id": y,
        "target_text": t,
        "source_text": INSTRUCTION + str(ex[TEXT_COL]),
    }

datasets = datasets.map(add_fields)

print("12-class distribution:")
print("  train      :", Counter(datasets["train"]["class_id"]))
print("  validation :", Counter(datasets["validation"]["class_id"]))
print("  test       :", Counter(datasets["test"]["class_id"]))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)

def tokenize_fn(batch):
    model_inputs = tokenizer(
        batch["source_text"],
        truncation=True,
        max_length=MAX_SOURCE_LENGTH,
    )

    labels = tokenizer(
        text_target=batch["target_text"],
        truncation=True,
        max_length=MAX_TARGET_LENGTH,
    )
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["class_id"] = batch["class_id"]
    return model_inputs

tokenized = datasets.map(tokenize_fn, batched=True)

NEED_CLASS_ID = bool(USE_CLASS_WEIGHTS or ENABLE_BALANCED_SAMPLING)
if NEED_CLASS_ID:
    keep_cols = set(tokenizer.model_input_names + ["labels", "class_id"])
else:
    keep_cols = set(tokenizer.model_input_names + ["labels"])

for split in tokenized.keys():
    remove_cols = [c for c in tokenized[split].column_names if c not in keep_cols]
    tokenized[split] = tokenized[split].remove_columns(remove_cols)

tokenized.set_format(type="torch")

base = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)

modules_to_save = ["lm_head"] if hasattr(base, "lm_head") else None

if TUNING_METHOD == "full":
    model = base
    LEARNING_RATE = LR_FULL

elif TUNING_METHOD == "lora":
    target_modules = detect_t5_lora_targets(base)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_LORA

elif TUNING_METHOD == "prefix":
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=PREFIX_VIRTUAL_TOKENS,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PREFIX

elif TUNING_METHOD == "prompt":
    token_dim  = getattr(base.config, "d_model", None)
    num_layers = getattr(base.config, "num_layers", None)
    num_heads  = getattr(base.config, "num_heads", None)

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=PROMPT_VIRTUAL_TOKENS,
        prompt_tuning_init_text="Output only one label: FR, A, L, LF, MN, O, PE, SC, SE, US, FT, PO.",
        tokenizer_name_or_path=MODEL_NAME_OR_PATH,
        token_dim=token_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PROMPT

elif TUNING_METHOD == "ptuning":
    token_dim  = getattr(base.config, "d_model", None)
    num_layers = getattr(base.config, "num_layers", None)
    num_heads  = getattr(base.config, "num_heads", None)

    peft_config = PromptEncoderConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=PTUNING_VIRTUAL_TOKENS,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.2,
        encoder_num_layers=2,
        encoder_hidden_size=256,
        token_dim=token_dim,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PTUNING

else:
    raise ValueError(f"Unknown TUNING_METHOD: {TUNING_METHOD}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_flan_t5_12cls",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER,
    max_grad_norm=MAX_GRAD_NORM,
    predict_with_generate=True,
    remove_unused_columns=not NEED_CLASS_ID,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=False,
    report_to="none",
    logging_steps=50,
)

trainer_kwargs = {}
trainer_class = Seq2SeqTrainer

if USE_CLASS_WEIGHTS:
    weights = compute_class_weights_multiclass(tokenized["train"]["class_id"], num_labels=NUM_LABELS)
    print("Class weights:", weights.tolist())
    trainer_class = WeightedSeq2SeqTrainer
    trainer_kwargs["class_weights"] = weights

if ENABLE_BALANCED_SAMPLING:
    trainer = BalancedSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics_from_generations(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        balance_sampling=True,
        num_labels=NUM_LABELS,
        **trainer_kwargs,
    )
else:
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics_from_generations(p, tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        **trainer_kwargs,
    )

trainer.train()

print("\nValidation metrics:")
trainer.evaluate(max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=GEN_NUM_BEAMS)

print("\nTest set report:")
test_output = trainer.predict(
    tokenized["test"],
    max_new_tokens=GEN_MAX_NEW_TOKENS,
    num_beams=GEN_NUM_BEAMS
)

pred_ids = test_output.predictions
label_ids = test_output.label_ids

if isinstance(pred_ids, (tuple, list)):
    pred_ids = pred_ids[0]
if hasattr(pred_ids, "ndim") and pred_ids.ndim == 3:
    pred_ids = np.argmax(pred_ids, axis=-1)

pred_ids = np.asarray(pred_ids).astype(np.int64)
label_ids_fixed = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids).astype(np.int64)

if tokenizer.vocab_size is not None and pred_ids.size > 0:
    max_id = int(tokenizer.vocab_size) - 1
    pred_ids = np.clip(pred_ids, 0, max_id)
    label_ids_fixed = np.clip(label_ids_fixed, 0, max_id)

pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
gold_texts = tokenizer.batch_decode(label_ids_fixed, skip_special_tokens=True)

pred_y = np.array([parse_generated_label(t) for t in pred_texts], dtype=int)
gold_y = np.array([label2id.get(normalize_label(t), -1) for t in gold_texts], dtype=int)

valid = gold_y != -1
pred_y = pred_y[valid]
gold_y = gold_y[valid]

unknown_mask = (pred_y == -1)
unknown_count = int(unknown_mask.sum())
total_count = int(len(pred_y))
unknown_rate = unknown_count / total_count if total_count > 0 else 0.0

print(f"Unknown count: {unknown_count}/{total_count} ({unknown_rate:.4f})")

pred_counts = Counter(pred_y.tolist())
print("Pred distribution (raw, id->count):", dict(pred_counts))
print("Pred distribution (raw, label->count):",
      {id2label.get(k, "UNK"): v for k, v in pred_counts.items()})

pred_y_fixed = pred_y.copy()
pred_y_fixed[unknown_mask] = 0

print("\nClassification report (unknown mapped to FR=0):")
report = classification_report(
    gold_y,
    pred_y_fixed,
    labels=list(range(NUM_LABELS)),
    target_names=[f"{id2label[i]}({i})" for i in range(NUM_LABELS)],
    digits=4,
    zero_division=0,
)
print(report)

with open("results_flan_t5_12cls.txt", "w", encoding="utf-8") as f:
    f.write(report)
