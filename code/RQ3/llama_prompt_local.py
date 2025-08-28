"""
Requirements classification · Local Llama-3.1-8B-Instruct prompting experiment (binary/multi + 3 templates)
"""

import os
import re
import time
from typing import List, Dict, Any

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ============================
# CONFIG — edit parameters here
# ============================
CONFIG = {
    # Data & output
    "CSV_PATH": "test.csv",         
    "OUT_CSV":  "preds_llama_local.csv",
    "NROWS":     None,                  

    # Single-experiment selection (or use the list below to run multiple in one go)
    "TASK":     "binary",               # "binary" or "multi"
    "TEMPLATE": "explain",              # "basic" / "explain" / "steps"

    # Model & inference
    "MODEL_PATH": "Llama-3.1-8B-Instruct",  
    "DTYPE": "auto",                  
    "DEVICE_MAP": "auto",             
    "ATTN_IMPL": None,                 
    "TEMPERATURE": 0.0,
    "TOP_P": 1.0,
    "DO_SAMPLE": False,                 
    "MAX_NEW_TOKENS": 1024,            
    "RETRIES": 2,                      

    # Minimal follow-up prompt when parsing fails 
    "REPAIR_ON_UNK": True,
    "REPAIR_MAX_NEW_TOKENS": 1024,
}

# Whether to run multiple experiment configs (inherits model/generation settings above)
RUN_MULTI_EXPERIMENTS = True
MULTI_EXPERIMENTS = [
    {"TASK": "binary", "TEMPLATE": "basic",   "OUT_CSV": "preds_bin_basic.csv"},
    {"TASK": "binary", "TEMPLATE": "explain", "OUT_CSV": "preds_bin_explain.csv"},
    {"TASK": "binary", "TEMPLATE": "steps",   "OUT_CSV": "preds_bin_steps.csv"},
    {"TASK": "multi",  "TEMPLATE": "basic",   "OUT_CSV": "preds_multi_basic.csv"},
    {"TASK": "multi",  "TEMPLATE": "explain", "OUT_CSV": "preds_multi_explain.csv"},
    {"TASK": "multi",  "TEMPLATE": "steps",   "OUT_CSV": "preds_multi_steps.csv"},
]

# ============================
# Prompt templates 
# ============================
PROMPTS: Dict[str, Dict[str, Dict[str, str]]] = {
    "binary": {
        "basic": {
            "system": (
                "You are a precise software requirements engineer. "
                "Follow instructions exactly and respect output constraints."
            ),
            "user": (
                "Decide whether the following requirement is a Functional Requirement (FR)\n"
                "or a Non-Functional Requirement (NFR).\n\n"
                "Requirement:\n{requirement}\n\n"
                "Final rules:\n"
                "- Output the final line exactly as one of:\n"
                "  <label>FR</label>  or  <label>NFR</label>\n"
                "- Do not output anything after </label>.\n\n"
                "Answer:\n"
            ),
        },
        "explain": {
            "system": (
                "You are a precise software requirements engineer. Provide a concise, audit-friendly result."
            ),
            "user": (
                "Classify the requirement as FR (functional behavior the system must perform) \n"
                "or NFR (constraints such as performance, security, usability, reliability, \n"
                "maintainability, availability, scalability, operability, look & feel, licensing, portability).\n\n"
                "Requirement:\n{requirement}\n\n"
                "Output format (two lines):\n"
                "Reason: <= 12 words (short and concrete)\n"
                "<label>FR</label>  or  <label>NFR</label>\n\n"
                "Final rules:\n"
                "- The second line must be exactly <label>FR</label> or <label>NFR</label>.\n"
                "- Do not output anything after </label>.\n"
            ),
        },
        "steps": {
            "system": (
                "You are a precise software requirements engineer. Think step by step briefly."
            ),
            "user": (
                "Classify the requirement into FR (functional behavior) or NFR (quality/constraint).\n\n"
                "Requirement:\n{requirement}\n\n"
                "Let's think step by step in 3 bullets:\n"
                "1) Does it describe a system behavior/output?\n"
                "2) If not, which quality/constraint is emphasized?\n"
                "3) Resolve ambiguity and choose one.\n\n"
                "Final rules:\n"
                "- After the bullets, output a single final line with exactly one of:\n"
                "  <label>FR</label>  or  <label>NFR</label>\n"
                "- Do not output anything after </label>.\n\n"
                "Answer:\n"
            ),
        },
    },
    "multi": {
        "basic": {
            "system": (
                "You are a precise software requirements engineer. Follow instructions exactly."
            ),
            "user": (
                "Classify the requirement into exactly one of the following 12 categories \n"
                "(output only the ALL-CAPS abbreviation):\n\n"
                "- FR  Functional Requirement\n"
                "- A   Availability\n"
                "- L   Legal & Licensing\n"
                "- LF  Look & Feel\n"
                "- MN  Maintainability\n"
                "- O   Operability\n"
                "- PE  Performance\n"
                "- SC  Scalability\n"
                "- SE  Security\n"
                "- US  Usability\n"
                "- FT  Fault Tolerance\n"
                "- PO  Portability\n\n"
                "Requirement:\n{requirement}\n\n"
                "Final rules:\n"
                "- Output the final line exactly as one of:\n"
                "  <label>FR</label>, <label>A</label>, <label>L</label>, <label>LF</label>, <label>MN</label>,\n"
                "  <label>O</label>, <label>PE</label>, <label>SC</label>, <label>SE</label>, <label>US</label>,\n"
                "  <label>FT</label>, <label>PO</label>\n"
                "- Do not output anything after </label>.\n\n"
                "Answer:\n"
            ),
        },
        "explain": {
            "system": (
                "You are a precise software requirements engineer. Provide a concise, audit-friendly result."
            ),
            "user": (
                "Classify the requirement into one of the 12 categories (output only the abbreviation):\n"
                "- FR  Functional Requirement\n"
                "- A   Availability\n"
                "- L   Legal & Licensing\n"
                "- LF  Look & Feel\n"
                "- MN  Maintainability\n"
                "- O   Operability\n"
                "- PE  Performance\n"
                "- SC  Scalability\n"
                "- SE  Security\n"
                "- US  Usability\n"
                "- FT  Fault Tolerance\n"
                "- PO  Portability\n\n"
                "Hints:\n"
                "- FR: functional behavior/output the system must perform.\n"
                "- Non-functional (A/L/LF/MN/O/PE/SC/SE/US/FT/PO): quality or constraint dimension.\n\n"
                "Requirement:\n{requirement}\n\n"
                "Output format (two lines):\n"
                "Reason: <= 12 words (short and concrete)\n"
                "<label>FR</label> / <label>A</label> / <label>L</label> / <label>LF</label> / <label>MN</label> /\n"
                "<label>O</label> / <label>PE</label> / <label>SC</label> / <label>SE</label> / <label>US</label> /\n"
                "<label>FT</label> / <label>PO</label>\n\n"
                "Final rules:\n"
                "- The second line must be exactly one <label>...</label> from the list.\n"
                "- Do not output anything after </label>.\n"
            ),
        },
        "steps": {
            "system": (
                "You are a precise software requirements engineer. Think step by step briefly."
            ),
            "user": (
                "Classify the requirement into exactly one of the 12 categories (output only the abbreviation):\n"
                "- FR  Functional Requirement\n"
                "- A   Availability\n"
                "- L   Legal & Licensing\n"
                "- LF  Look & Feel\n"
                "- MN  Maintainability\n"
                "- O   Operability\n"
                "- PE  Performance\n"
                "- SC  Scalability\n"
                "- SE  Security\n"
                "- US  Usability\n"
                "- FT  Fault Tolerance\n"
                "- PO  Portability\n\n"
                "Requirement:\n{requirement}\n\n"
                "Let's think step by step in 4 bullets:\n"
                "1) Decide FR vs NFR (is it a concrete system behavior?).\n"
                "2) If FR, stop and choose FR.\n"
                "3) If NFR, identify which quality/constraint dimension best fits (A/L/LF/MN/O/PE/SC/SE/US/FT/PO).\n"
                "4) Resolve ambiguity by picking the single best category.\n\n"
                "Final rules:\n"
                "- After the bullets, output a single final line with exactly one of:\n"
                "  <label>FR</label>, <label>A</label>, <label>L</label>, <label>LF</label>, <label>MN</label>,\n"
                "  <label>O</label>, <label>PE</label>, <label>SC</label>, <label>SE</label>, <label>US</label>,\n"
                "  <label>FT</label>, <label>PO</label>\n"
                "- Do not output anything after </label>.\n\n"
                "Answer:\n"
            ),
        },
    },
}

# ============= Label sets & parsing regexes (robust version) =============
LABELS_BINARY = ["FR", "NFR"]
LABELS_MULTI  = ["FR","A","L","LF","MN","O","PE","SC","SE","US","FT","PO"]

END_BIN_RE   = re.compile(r"<label>\s*(FR|NFR)\s*</label>\s*$", re.IGNORECASE | re.DOTALL)
END_MULTI_RE = re.compile(r"<label>\s*(FR|A|L|LF|MN|O|PE|SC|SE|US|FT|PO)\s*</label>\s*$",
                          re.IGNORECASE | re.DOTALL)
BIN_TAG_RE   = re.compile(r"<label>\s*(FR|NFR)\s*</label>", re.IGNORECASE)
MULTI_TAG_RE = re.compile(r"<label>\s*(FR|A|L|LF|MN|O|PE|SC|SE|US|FT|PO)\s*</label>", re.IGNORECASE)

def extract_label(text: str, task: str) -> str:
    if not text:
        return "UNK"
    t = text.strip()

    m_end = (END_BIN_RE if task == "binary" else END_MULTI_RE).search(t)
    if m_end:
        return m_end.group(1).upper()

    allm = (BIN_TAG_RE if task == "binary" else MULTI_TAG_RE).findall(t)
    if allm:
        return allm[-1].upper()

    lines = t.splitlines()
    last = (lines[-1] if lines else t).strip().upper()
    if task == "binary":
        if last == "NFR": return "NFR"
        if last == "FR":  return "FR"
        if re.search(r"\bNFR\b", last): return "NFR"
        if re.search(r"\bFR\b",  last): return "FR"
        return "UNK"
    else:
        for lab in ["FR","A","L","LF","MN","O","PE","SC","SE","US","FT","PO"]:
            if last == lab or re.search(rf"\b{lab}\b", last):
                return lab
        return "UNK"

def normalize_gold(label: str, task: str) -> str:
    if label is None:
        return "UNK"
    s_up = str(label).strip().upper()
    if task == "binary":
        return "FR" if s_up == "FR" else "NFR"
    else:
        if s_up in ("F", "FR"): return "FR"
        if s_up == "PO": return "PO"
        return s_up

def ensure_requirement_column(df: pd.DataFrame) -> str:
    for cand in ["Requirement", "requirement", "text", "Text", "req", "Req"]:
        if cand in df.columns: return cand
    raise ValueError("The requirement text column was not found.")

# ==================== Model loading & per-sample generation ====================
def pick_dtype(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

def load_model_and_tokenizer(cfg: Dict[str, Any]):
    dtype = pick_dtype(cfg["DTYPE"])
    kwargs = {
        "torch_dtype": dtype,
        "device_map": cfg["DEVICE_MAP"],
    }
    if cfg["ATTN_IMPL"]:
        kwargs["attn_implementation"] = cfg["ATTN_IMPL"]

    tok = AutoTokenizer.from_pretrained(cfg["MODEL_PATH"], use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(cfg["MODEL_PATH"], **kwargs)
    return tok, mdl

@torch.inference_mode()
def generate_once(tokenizer, model, messages: List[Dict[str, str]], cfg: Dict[str, Any]) -> str:
    """
    Build a single input using the chat template, then generate.
    Llama-3.1-Instruct provides a chat template; set add_generation_prompt=True.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    gen_out = model.generate(
        **inputs,
        do_sample=cfg["DO_SAMPLE"],
        temperature=cfg["TEMPERATURE"],
        top_p=cfg["TOP_P"],
        max_new_tokens=cfg["MAX_NEW_TOKENS"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return output_text

def build_messages(task: str, template: str, req_text: str) -> List[Dict[str,str]]:
    tpl = PROMPTS[task][template]
    return [
        {"role": "system", "content": tpl["system"]},
        {"role": "user",   "content": tpl["user"].format(requirement=req_text)},
    ]

def build_repair_messages(task: str, req_text: str) -> List[Dict[str,str]]:
    if task == "binary":
        txt = (
            "Output exactly one of the following and nothing else:\n"
            "<label>FR</label>  or  <label>NFR</label>\n\n"
            "Requirement:\n" + req_text + "\n"
        )
    else:
        txt = (
            "Output exactly one of the following and nothing else:\n"
            "<label>FR</label> <label>A</label> <label>L</label> <label>LF</label> <label>MN</label> "
            "<label>O</label> <label>PE</label> <label>SC</label> <label>SE</label> <label>US</label> "
            "<label>FT</label> <label>PO</label>\n\n"
            "Requirement:\n" + req_text + "\n"
        )
    return [
        {"role": "system", "content": "Return exactly one token of the form <label>...</label> and nothing else."},
        {"role": "user",   "content": txt},
    ]

# ======================= Run a single experiment (with latency sampling logs) =======================
def run_one_experiment(
    csv_path: str, out_csv: str, task: str, template: str, cfg: Dict[str, Any]
):
    # 1) Data
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if cfg["NROWS"]:
        df = df.head(cfg["NROWS"]).copy()
    req_col = ensure_requirement_column(df)
    if "Specific_Type" not in df.columns:
        raise ValueError("Didnot find the label column")

    golds = [normalize_gold(x, task) for x in df["Specific_Type"].tolist()]

    # 2) Model
    tokenizer, model = load_model_and_tokenizer(cfg)

    # 3) Per-sample generation (sample-print latency and output length)
    preds, raws, latencies = [], [], []
    print(f"\n=== Running local experiment: task={task}, template={template}, model={cfg['MODEL_PATH']} ===")
    print(f"Data: {csv_path} (n={len(df)}) ; max_new_tokens={cfg['MAX_NEW_TOKENS']} ; do_sample={cfg['DO_SAMPLE']}")
    for i, req in enumerate(tqdm(df[req_col].astype(str).tolist(), desc="Infer"), start=1):
        messages = build_messages(task, template, req)

        t0 = time.perf_counter()
        raw = None
        for attempt in range(cfg["RETRIES"] + 1):
            try:
                raw = generate_once(tokenizer, model, messages, cfg)
                break
            except Exception:
                if attempt >= cfg["RETRIES"]:
                    raise
                time.sleep(0.2 + np.random.rand() * 0.4)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        dt = time.perf_counter() - t0
        latencies.append(dt)

        if i <= 3 or i % 50 == 0:
            try:
                approx_tokens = len(tokenizer(raw, add_special_tokens=False)["input_ids"])
            except Exception:
                approx_tokens = -1
            head = (raw[:160] if raw else "").replace("\n", " ")
            print(f"\n[dbg] i={i}  time={dt:.2f}s  out_tokens≈{approx_tokens}\n[raw head] {head}\n---")

        lab = extract_label(raw, task)

        if lab == "UNK" and cfg["REPAIR_ON_UNK"]:
            repair_msgs = build_repair_messages(task, req)
            repair_cfg = dict(cfg)
            repair_cfg["DO_SAMPLE"] = False
            repair_cfg["TEMPERATURE"] = 0.0
            repair_cfg["MAX_NEW_TOKENS"] = cfg["REPAIR_MAX_NEW_TOKENS"]
            try:
                raw2 = generate_once(tokenizer, model, repair_msgs, repair_cfg)
                lab2 = extract_label(raw2, task)
                if lab2 != "UNK":
                    lab, raw = lab2, raw2
            except Exception:
                pass

        preds.append(lab)
        raws.append(raw)

    # 4) Evaluation
    labels_order = LABELS_BINARY if task == "binary" else LABELS_MULTI
    valid_set = set(labels_order)
    gold_eval = [g if g in valid_set else "UNK" for g in golds]
    pred_eval = [p if p in valid_set else "UNK" for p in preds]
    labels_with_unk = labels_order + (["UNK"] if ("UNK" in gold_eval or "UNK" in pred_eval) else [])

    unk_rate = sum(p == "UNK" for p in preds) / len(preds)
    if unk_rate > 0:
        print(f"[Warn] UNK rate = {unk_rate:.2%}")

    print("\n==== Classification Report ====")
    print(classification_report(gold_eval, pred_eval, digits=4, labels=labels_with_unk, zero_division=0))

    print("==== Confusion Matrix (rows=true, cols=pred) ====")
    cm = confusion_matrix(gold_eval, pred_eval, labels=labels_with_unk)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in labels_with_unk], columns=[f"P:{l}" for l in labels_with_unk])
    print(cm_df)

    # 5) Save
    out_df = df.copy()
    out_df["gold_norm"] = gold_eval
    out_df["pred"]      = pred_eval
    out_df["raw_output"]= raws
    if out_csv:
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved predictions to: {out_csv}")

    # 6) Summary stats of generation latency
    if latencies:
        arr = np.array(latencies, dtype=float)
        print("\n==== Per-sample latency (s) ====")
        print(f"mean={arr.mean():.2f}  p50={np.percentile(arr,50):.2f}  p90={np.percentile(arr,90):.2f}  max={arr.max():.2f}")

# ======================= Entry point =======================
if __name__ == "__main__":
    if RUN_MULTI_EXPERIMENTS:
        for exp in MULTI_EXPERIMENTS:
            run_one_experiment(
                csv_path=CONFIG["CSV_PATH"],
                out_csv=exp["OUT_CSV"],
                task=exp["TASK"],
                template=exp["TEMPLATE"],
                cfg=CONFIG,
            )
    else:
        run_one_experiment(
            csv_path=CONFIG["CSV_PATH"],
            out_csv=CONFIG["OUT_CSV"],
            task=CONFIG["TASK"],
            template=CONFIG["TEMPLATE"],
            cfg=CONFIG,
        )
