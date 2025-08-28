# ==============================================
# Configuration — modify this section only
# ==============================================
MODEL_NAME_OR_PATH = "t5-base"   # local path or identifier
TUNING_METHOD      = "full"    # choose from ["full", "lora", "prefix", "prompt", "ptuning"]
TRAIN_FILE         = "train.csv"
VAL_FILE           = "val.csv"
TEST_FILE          = "test.csv"
EPOCHS             = 8

# ==============================================
# Imports
# ==============================================
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import (
    TaskType,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    get_peft_model,
    PromptTuningInit,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==============================================
# 1. Load and preprocess the data
# ==============================================
datasets = load_dataset("csv", data_files={
    "train": TRAIN_FILE,
    "validation": VAL_FILE,
    "test": TEST_FILE
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=True)
tokenizer.model_max_length = 1024

# ─── 1. Add a text column: "Functional Requirement" or "Non-Functional Requirement"
label_map = {
    "FR": "Functional Requirement",
    "A": "Availability",
    "L": "Legal & Licensing",
    "LF": "Look & Feel",
    "MN": "Maintainability",
    "O": "Operability",
    "PE": "Performance",
    "SC": "Scalability",
    "SE": "Security",
    "US": "Usability",
    "FT": "Fault Tolerance",
    "PO": "Portability",
}

def map_label_str(ex):
    return {
        "label_str": label_map.get(ex["Specific_Type"], "Unknown")
    }

datasets = datasets.map(map_label_str)

# ─── 2. Calculate the maximum input/output length
PREFIX_VIRTUAL_TOKENS = 30
PROMPT_VIRTUAL_TOKENS = 12
PTUNING_VIRTUAL_TOKENS = 128

added = {
    "prefix": PREFIX_VIRTUAL_TOKENS,
    "prompt": PROMPT_VIRTUAL_TOKENS,
    "ptuning": PTUNING_VIRTUAL_TOKENS
}.get(TUNING_METHOD, 0)

max_input_length  = tokenizer.model_max_length - added
max_target_length = 32  

# ─── 3. Tokenize
def preprocess(batch):
    model_inputs = tokenizer(
        batch["Requirement"],
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
    )
    labels = tokenizer(
        batch["label_str"],
        truncation=True,
        padding="max_length",
        max_length=max_target_length,
    )
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["label_str"] = batch["label_str"]
    return model_inputs

datasets = datasets.map(
    preprocess,
    batched=True,
    remove_columns=["Requirement", "Specific_Type"]
)

datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# ==============================================
# 2. Load model and apply the chosen tuning method
# ==============================================
base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=True)

if TUNING_METHOD == "full":
    model = base
    LEARNING_RATE = 1e-5

elif TUNING_METHOD == "lora":
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8, lora_alpha=128, lora_dropout=0.1, bias="none",
    )
    model = get_peft_model(base, peft_config)
    LEARNING_RATE = 7e-5

elif TUNING_METHOD == "prefix":
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=PREFIX_VIRTUAL_TOKENS,
    )
    model = get_peft_model(base, peft_config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    LEARNING_RATE = 7e-3

elif TUNING_METHOD == "prompt":
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=PROMPT_VIRTUAL_TOKENS,
        prompt_tuning_init_text=(
            "Given a requirement text, classify it into one of the following: "
            "Functional Requirement, Availability, Legal & Licensing, Look & Feel, "
            "Maintainability, Operability, Performance, Scalability, Security, "
            "Usability, Fault Tolerance, or Portability."
        ),
        tokenizer_name_or_path=MODEL_NAME_OR_PATH,
    )
    model = get_peft_model(base, peft_config)
    LEARNING_RATE = 5e-3

elif TUNING_METHOD == "ptuning":
    peft_config = PromptEncoderConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=PTUNING_VIRTUAL_TOKENS,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.3,
        encoder_num_layers=16,
        encoder_hidden_size=128,
    )
    model = get_peft_model(base, peft_config)
    LEARNING_RATE = 5e-4

else:
    raise ValueError(f"Unknown TUNING_METHOD: {TUNING_METHOD}")

# ==============================================
# 3. Metrics for generative
# ==============================================
def compute_metrics_seq2seq(eval_pred):
    decoded_preds  = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    y_pred = [txt.strip() for txt in decoded_preds]
    y_true = [txt.strip() for txt in decoded_labels]

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

# ==============================================
# 4. Training setup
# ==============================================
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    metric_for_best_model="eval_f1_weighted",
    load_best_model_at_end=True,
    predict_with_generate=True,
    # predict_with_generate=False,
    generation_max_length=max_target_length,
    generation_num_beams=2,
    save_total_limit=1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_seq2seq,
)

# ==============================================
# 5. Train, validate and evaluate on the test set
# ==============================================
trainer.train()
trainer.evaluate()

test_pred = trainer.predict(datasets["test"])
decoded_preds  = tokenizer.batch_decode(
    test_pred.predictions, skip_special_tokens=True
)

decoded_labels = tokenizer.batch_decode(
    test_pred.label_ids,    skip_special_tokens=True
)

y_pred = [txt.strip() for txt in decoded_preds]
y_true = [txt.strip() for txt in decoded_labels]

target_names = [
    "Functional Requirement",
    "Availability",
    "Legal & Licensing",
    "Look & Feel",
    "Maintainability",
    "Operability",
    "Performance",
    "Scalability",
    "Security",
    "Usability",
    "Fault Tolerance",
    "Portability"
]
# 1. Count the number of predictions that do not fall into these two categories
unknown_count = sum(1 for pred in y_pred if pred not in target_names)
print(f"Number of predictions not in {target_names}: {unknown_count}")

# 2. Classify the predictions of all unknown categories as Non-Functional requirements
y_pred_mapped = [
    pred if pred in target_names else "Non-Functional Requirement"
    for pred in y_pred
]

# 3. Print the original classification report
print("The original classification report:")
print(classification_report(
    y_true, y_pred,
    target_names=target_names,
    digits=4
))

# 4. Print the classification report after mapping
print("\nThe classification report after mapping:")
print(classification_report(
    y_true, y_pred_mapped,
    target_names=target_names,
    digits=4
))
