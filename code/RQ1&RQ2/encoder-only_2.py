# ==============================================
# Configuration — modify this section only
# ==============================================
MODEL_NAME_OR_PATH = "roberta-large"   # local path or identifier
TUNING_METHOD = "full"      # choose from ["full", "lora", "prefix", "prompt", "ptuning"]
TRAIN_FILE = "train.csv"
VAL_FILE   = "val.csv"
TEST_FILE  = "test.csv"
NUM_LABELS = 2
EPOCHS     = 3

# ==============================================
# Imports
# ==============================================
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from sklearn.metrics import classification_report

from peft import (
    TaskType,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    get_peft_model,
    PromptEncoderReparameterizationType,
    PeftType,
    PromptTuningInit,
)

import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evaluate import load

# ==============================================
# 1. Load and preprocess the data
# ==============================================
datasets = load_dataset("csv", data_files={
    "train": TRAIN_FILE,
    "validation": VAL_FILE,
    "test": TEST_FILE
})

# Map Specific_Type → binary label (0 for FR, 1 for non-FR)
datasets = datasets.map(lambda ex: {"labels": 0 if ex["Specific_Type"] == "FR" else 1})

# Load tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=True)
config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=True)
tokenizer.model_max_length = config.max_position_embeddings

# Tokenize the Requirement field
PREFIX_VIRTUAL_TOKENS = 30
PROMPT_VIRTUAL_TOKENS = 12
PTUNING_VIRTUAL_TOKENS = 128

added = 0
if TUNING_METHOD == "prefix":
    added = PREFIX_VIRTUAL_TOKENS
elif TUNING_METHOD == "prompt":
    added = PROMPT_VIRTUAL_TOKENS
elif TUNING_METHOD == "ptuning":
    added = PTUNING_VIRTUAL_TOKENS

max_input_length = tokenizer.model_max_length - added

# ---- tokenize ----
datasets = datasets.map(
    lambda batch: tokenizer(
        batch["Requirement"],
        truncation=True,
        padding="max_length",
        max_length=max_input_length
    ),
    batched=True
)

# Keep only the inputs + label for PyTorch
datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ==============================================
# 2. Load model and apply the chosen tuning method
# ==============================================
base = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    local_files_only=True,
    num_labels=NUM_LABELS
)

if TUNING_METHOD == "full":
    model = base
    LEARNING_RATE = 1e-5

elif TUNING_METHOD == "lora":
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none",
        target_modules=None,
        # target_modules=["q_lin", "k_lin", "v_lin"], # ditilbert-base-uncased 
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = 7e-5

elif TUNING_METHOD == "prefix":
    # base.config.use_cache = False  
    peft_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_CLS, 
        num_virtual_tokens=PREFIX_VIRTUAL_TOKENS,
    )
    model = get_peft_model(base, peft_config)
    model.cls_layer_name = "classifier"
    model.gradient_checkpointing_disable()
    model.print_trainable_parameters()
    LEARNING_RATE = 7e-3

elif TUNING_METHOD == "prompt":
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=PROMPT_VIRTUAL_TOKENS,
        prompt_tuning_init_text=(
            "Given a requirement text, classify whether it's "
            "Functional Requirement (FR) or Non‑Functional (NFR)."
        ),
        tokenizer_name_or_path=MODEL_NAME_OR_PATH,
        # num_layers=base.config.num_hidden_layers, # ditilbert-base-uncased
        # token_dim=base.config.hidden_size, # ditilbert-base-uncased
        # num_attention_heads=base.config.num_attention_heads # ditilbert-base-uncased
    )
    model = get_peft_model(base, peft_config)
    LEARNING_RATE = 5e-3

elif TUNING_METHOD == "ptuning":
    peft_config = PromptEncoderConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=PTUNING_VIRTUAL_TOKENS,
        encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
        encoder_dropout=0.3,
        encoder_num_layers=16,
        encoder_hidden_size=128,
        # num_layers=base.config.num_hidden_layers, # ditilbert-base-uncased
        # token_dim=base.config.hidden_size, # ditilbert-base-uncased
        # num_attention_heads=base.config.num_attention_heads # ditilbert-base-uncased

    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    LEARNING_RATE = 5e-4 
    
else:
    raise ValueError(f"Unknown TUNING_METHOD: {TUNING_METHOD}")

acc_metric = evaluate.load("accuracy")
f1_metirc = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
# acc_metric = load('metrics/accuracy.py')
# f1_metirc = load('metrics/f1.py')
# precision_metric = load('metrics/precision.py')
# recall_metric = load('metrics/recall.py')

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# ==============================================
# 3. Training setup
# ==============================================
training_args = TrainingArguments(
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
    metric_for_best_model="f1_score",
    load_best_model_at_end=True,
    gradient_checkpointing=False,
    save_total_limit=1,
    # label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=eval_metric,
)

# ==============================================
# 4. Train, evaluate, and report on test set
# ==============================================
trainer.train()
trainer.evaluate()

test_output = trainer.predict(datasets["test"])
preds = np.argmax(test_output.predictions, axis=1)
labels = test_output.label_ids

print(classification_report(labels, preds, digits=4))

