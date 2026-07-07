MODEL_NAME_OR_PATH = "ModernBERT-large" # choices: "roberta-base", "roberta-large", "deberta-v3-base", "deberta-v3-large", "distilbert-base-uncased", "ModernBERT-base", "ModernBERT-large"

TUNING_METHOD = "full" # choose from ["full", "lora", "prefix", "prompt", "ptuning"]

TRAIN_FILE = "binary/train.csv"
VAL_FILE   = "binary/val.csv"
TEST_FILE  = "binary/test.csv"

TEXT_COL   = "Requirement"
LABEL_COL  = "Specific_Type"
NUM_LABELS = 2
EPOCHS     = 3
SEED       = 42

LOCAL_FILES_ONLY    = True
TRUST_REMOTE_CODE   = True
USE_CLASS_WEIGHTS   = True
USE_DYNAMIC_PADDING = True

PER_DEVICE_TRAIN_BS = 2
PER_DEVICE_EVAL_BS  = 8
GRAD_ACCUM_STEPS    = 16
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
LR_SCHEDULER        = "linear"
MAX_GRAD_NORM       = 1.0

MAX_SEQ_LEN_CAP = 1024

PREFIX_VIRTUAL_TOKENS  = 30
PROMPT_VIRTUAL_TOKENS  = 12
PTUNING_VIRTUAL_TOKENS = 128

LR_FULL    = 1e-5
LR_LORA    = 7e-5
LR_PREFIX  = 1e-3
LR_PROMPT  = 1e-3
LR_PTUNING = 5e-4

import sys
import numpy as np
import torch

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
def _torch_compile_disabled(model=None, *args, **kwargs):
    if callable(model):
        return model
    def deco(fn):
        return fn
    return deco
torch.compile = _torch_compile_disabled


from collections import Counter
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
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

def normalize_label(x: str) -> str:
    return str(x).strip().upper()

def infer_max_length(tokenizer, config) -> int:
    cfg_max = getattr(config, "max_position_embeddings", None)
    tok_max = getattr(tokenizer, "model_max_length", None)
    candidates = []
    if isinstance(cfg_max, int) and 0 < cfg_max < 100000:
        candidates.append(cfg_max)
    if isinstance(tok_max, int) and 0 < tok_max < 100000:
        candidates.append(tok_max)
    return min(candidates) if candidates else 512

def get_added_virtual_tokens(tuning_method: str) -> int:
    if tuning_method == "prefix":
        return PREFIX_VIRTUAL_TOKENS
    if tuning_method == "prompt":
        return PROMPT_VIRTUAL_TOKENS
    if tuning_method == "ptuning":
        return PTUNING_VIRTUAL_TOKENS
    return 0

def detect_classifier_module_name(model) -> str:
    if hasattr(model, "classifier"):
        return "classifier"
    if hasattr(model, "score"):
        return "score"
    return ""

def infer_lora_target_modules(base_model) -> list:
    module_names = [name for name, _ in base_model.named_modules()]
    candidate_sets = [
        ["query", "key", "value"],
        ["q_lin", "k_lin", "v_lin"],
        ["query_proj", "key_proj", "value_proj"],
        ["q_proj", "k_proj", "v_proj"],
    ]
    def score(cands):
        s = 0
        for m in module_names:
            for c in cands:
                if m.endswith(c):
                    s += 1
        return s
    best = max(candidate_sets, key=score)
    return best if score(best) > 0 else []

def compute_class_weights_safe(labels: list, num_labels: int) -> torch.Tensor:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_labels)
    total = counts.sum()
    weights = np.zeros(num_labels, dtype=np.float32)
    for i in range(num_labels):
        if counts[i] > 0:
            weights[i] = float(total) / (float(num_labels) * float(counts[i]))
        else:
            weights[i] = 0.0
    w = torch.tensor(weights, dtype=torch.float32)
    return w, counts

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    prec_w = precision_score(labels, preds, average="weighted", zero_division=0)
    rec_w  = recall_score(labels, preds, average="weighted", zero_division=0)
    f1_w   = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "f1_macro": f1_macro,
    }

set_seed(SEED)

datasets = load_dataset(
    "csv",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE, "test": TEST_FILE}
)

def map_label(ex):
    t = normalize_label(ex[LABEL_COL])
    return {"labels": 0 if t == "FR" else 1}

datasets = datasets.map(map_label)

print("Label distribution:")
print("  train      :", Counter(datasets["train"]["labels"]))
print("  validation :", Counter(datasets["validation"]["labels"]))
print("  test       :", Counter(datasets["test"]["labels"]))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)
config = AutoConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
)

model_max_len = infer_max_length(tokenizer, config)
added_tokens = get_added_virtual_tokens(TUNING_METHOD)
max_input_length = max(8, model_max_len - added_tokens)
max_input_length = min(max_input_length, MAX_SEQ_LEN_CAP)

print(f"max_position/model_max_len={model_max_len}, added_virtual_tokens={added_tokens}, "
      f"cap={MAX_SEQ_LEN_CAP}, max_input_length={max_input_length}")

def tokenize_fn(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=max_input_length,
    )

datasets = datasets.map(
    tokenize_fn,
    batched=True,
    remove_columns=[c for c in datasets["train"].column_names if c not in ["labels"]],
)

class_weights = None
if USE_CLASS_WEIGHTS:
    train_labels = [int(x) for x in datasets["train"]["labels"]]
    class_weights, counts = compute_class_weights_safe(train_labels, num_labels=NUM_LABELS)
    print("Label counts:", counts.tolist())
    print("Class weights (final):", class_weights.tolist())
    if float(class_weights.sum()) == 0.0:
        raise RuntimeError("Class weights are all zeros — check label mapping/extraction.")

model_input_cols = [c for c in tokenizer.model_input_names if c in datasets["train"].column_names]
datasets.set_format(type="torch", columns=model_input_cols + ["labels"])

data_collator = None
if USE_DYNAMIC_PADDING:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
    num_labels=NUM_LABELS,
)

base.config.id2label = {0: "FR", 1: "NFR"}
base.config.label2id = {"FR": 0, "NFR": 1}

classifier_name = detect_classifier_module_name(base)

if TUNING_METHOD == "full":
    model = base
    LEARNING_RATE = LR_FULL

elif TUNING_METHOD == "lora":
    target_modules = ["Wqkv", "Wo"]

    print("LoRA target_modules (ModernBERT):", target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
        modules_to_save=[classifier_name] if classifier_name else None,
    )

    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_LORA

elif TUNING_METHOD == "prefix":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=PREFIX_VIRTUAL_TOKENS,
        modules_to_save=[classifier_name] if classifier_name else None,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PREFIX

elif TUNING_METHOD == "prompt":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=PROMPT_VIRTUAL_TOKENS,
        prompt_tuning_init_text=(
            "Given a requirement text, classify whether it is Functional Requirement (FR) "
            "or Non-Functional Requirement (NFR)."
        ),
        tokenizer_name_or_path=MODEL_NAME_OR_PATH,
        modules_to_save=[classifier_name] if classifier_name else None,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PROMPT

elif TUNING_METHOD == "ptuning":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    peft_config = PromptEncoderConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=PTUNING_VIRTUAL_TOKENS,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.3,
        encoder_num_layers=16,
        encoder_hidden_size=128,
        modules_to_save=[classifier_name] if classifier_name else None,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PTUNING

else:
    raise ValueError(f"Unknown TUNING_METHOD: {TUNING_METHOD}")

if classifier_name:
    head_trainable = any((classifier_name in n and p.requires_grad) for n, p in model.named_parameters())
    print(f"Check head trainability (module='{classifier_name}'):")
    print("  head_trainable =", head_trainable)

training_args = TrainingArguments(
    output_dir="./results",
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
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=False,
    report_to="none",
    logging_steps=50,
)

if USE_CLASS_WEIGHTS:
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

trainer.train()

print("\nValidation metrics:")
trainer.evaluate()

print("\nTest set report:")
test_output = trainer.predict(datasets["test"])
test_preds = np.argmax(test_output.predictions, axis=1)
test_labels = test_output.label_ids
print(classification_report(test_labels, test_preds, digits=4, target_names=["FR(0)", "NFR(1)"]))
