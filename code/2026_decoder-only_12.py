MODEL_NAME_OR_PATH = "Qwen3-4B" # decoder-only model choices: "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B-Instruct-2507", "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"

TUNING_METHOD = "full" # choose from ["full", "lora", "prefix", "prompt", "ptuning"]

TRAIN_FILE = "multi/train.csv"
VAL_FILE   = "multi/val.csv"
TEST_FILE  = "multi/test.csv"

TEXT_COL   = "Requirement"
LABEL_COL  = "Specific_Type"

LABEL_LIST = ["FR","A","L","LF","MN","O","PE","SC","SE","US","FT","PO"]
NUM_LABELS = 12

EPOCHS     = 10
SEED       = 42

LOCAL_FILES_ONLY    = False
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

LORA_R = 8
LORA_ALPHA = 128
LORA_DROPOUT = 0.1

DISABLE_TORCH_COMPILE = True

import os
import sys
from collections import Counter
import numpy as np
import torch

if DISABLE_TORCH_COMPILE:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

def _torch_compile_disabled(model=None, *args, **kwargs):
    if callable(model):
        return model
    def deco(fn):
        return fn
    return deco

if DISABLE_TORCH_COMPILE or sys.version_info >= (3, 12):
    torch.compile = _torch_compile_disabled

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
    return min(candidates) if candidates else 1024

def get_added_virtual_tokens(tuning_method: str) -> int:
    if tuning_method == "prefix":
        return PREFIX_VIRTUAL_TOKENS
    if tuning_method == "prompt":
        return PROMPT_VIRTUAL_TOKENS
    if tuning_method == "ptuning":
        return PTUNING_VIRTUAL_TOKENS
    return 0

def detect_classifier_module_name(model) -> str:
    if hasattr(model, "score"):
        return "score"
    if hasattr(model, "classifier"):
        return "classifier"
    return ""

def compute_class_weights_safe(labels: list, num_labels: int):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_labels)
    total = counts.sum()
    weights = np.zeros(num_labels, dtype=np.float32)
    for i in range(num_labels):
        if counts[i] > 0:
            weights[i] = float(total) / (float(num_labels) * float(counts[i]))
        else:
            weights[i] = 0.0
    return torch.tensor(weights, dtype=torch.float32), counts

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
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    prec_macro = precision_score(labels, preds, average="macro", zero_division=0)
    rec_macro  = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
    }

def infer_lora_target_modules_decoder_only(model) -> list:
    module_names = [n for n, _ in model.named_modules()]
    cand = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if any(any(m.endswith(c) for m in module_names) for c in cand):
        return cand

    alt = ["c_attn", "c_proj"]
    if any(any(m.endswith(c) for m in module_names) for c in alt):
        return alt

    import torch.nn as nn
    from collections import Counter
    leaf = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf.append(name.split(".")[-1])
    freq = Counter(leaf)
    targets = [k for k, v in freq.items() if v >= 2]
    return sorted(targets)

set_seed(SEED)

label2id = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
id2label = {i: lbl for lbl, i in label2id.items()}

datasets = load_dataset(
    "csv",
    data_files={"train": TRAIN_FILE, "validation": VAL_FILE, "test": TEST_FILE}
)

def map_label_12(ex):
    t = normalize_label(ex[LABEL_COL])
    if t not in label2id:
        raise ValueError(f"Unknown label '{t}'. Expected one of: {LABEL_LIST}")
    return {"labels": label2id[t]}

datasets = datasets.map(map_label_12)

print("Label distribution:")
print("  train      :", Counter(datasets["train"]["labels"]))
print("  validation :", Counter(datasets["validation"]["labels"]))
print("  test       :", Counter(datasets["test"]["labels"]))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=LOCAL_FILES_ONLY,
    trust_remote_code=TRUST_REMOTE_CODE,
    use_fast=True,
)

tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
        raise RuntimeError("Class weights are all zeros — check label mapping.")

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

if getattr(base.config, "pad_token_id", None) is None:
    base.config.pad_token_id = tokenizer.pad_token_id

base.config.label2id = label2id
base.config.id2label = id2label

classifier_name = detect_classifier_module_name(base)

if TUNING_METHOD == "full":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    model = base
    LEARNING_RATE = LR_FULL

elif TUNING_METHOD == "lora":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False

    target_modules = infer_lora_target_modules_decoder_only(base)
    if not target_modules:
        raise ValueError("Failed to infer LoRA target_modules for this decoder-only model.")
    print("LoRA target_modules:", target_modules)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_LORA

elif TUNING_METHOD == "prefix":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = True

    num_layers = getattr(base.config, "num_hidden_layers", None) or getattr(base.config, "num_layers", None)
    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "n_embd", None)
    num_heads = getattr(base.config, "num_attention_heads", None)
    num_kv_heads = getattr(base.config, "num_key_value_heads", None) or num_heads

    if num_layers is None or hidden_size is None or num_heads is None:
        raise ValueError("Cannot read num_layers/hidden_size/num_attention_heads from config; cannot run prefix safely.")

    token_dim = int(hidden_size * (num_kv_heads / num_heads))
    print(f"[prefix] num_layers={num_layers}, hidden={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, token_dim={token_dim}")

    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=PREFIX_VIRTUAL_TOKENS,
        num_layers=num_layers,
        token_dim=token_dim,
        num_attention_heads=num_kv_heads,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PREFIX

elif TUNING_METHOD == "prompt":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=PROMPT_VIRTUAL_TOKENS,
        prompt_tuning_init_text=(
            "Classify this software requirement into one of 12 categories: "
            "FR, A, L, LF, MN, O, PE, SC, SE, US, FT, PO. Text:"
        ),
        tokenizer_name_or_path=MODEL_NAME_OR_PATH,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PROMPT

elif TUNING_METHOD == "ptuning":
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    peft_config = PromptEncoderConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=PTUNING_VIRTUAL_TOKENS,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.3,
        encoder_num_layers=16,
        encoder_hidden_size=128,
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = LR_PTUNING

else:
    raise ValueError(f"Unknown TUNING_METHOD: {TUNING_METHOD}")

if classifier_name and TUNING_METHOD != "full":
    head_trainable = any((classifier_name in n and p.requires_grad) for n, p in model.named_parameters())
    print(f"Check head trainability (module='{classifier_name}'):")
    print("  head_trainable =", head_trainable)

training_args = TrainingArguments(
    output_dir="./results_decoderonly_12cls",
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
    label_names=["labels"],
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
preds = np.argmax(test_output.predictions, axis=1)
labels = test_output.label_ids

report = classification_report(
    labels,
    preds,
    digits=4,
    target_names=[f"{id2label[i]}({i})" for i in range(NUM_LABELS)],
    zero_division=0,
)

with open("results_decoderonly_12cls.txt", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
