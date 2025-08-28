import os
import re
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import random 
import openai

# ============================
# CONFIG
# ============================
CONFIG = {
    # Data & output
    "CSV_PATH": "test.csv",               
    "OUT_CSV":  "preds_llama.csv",        
    "NROWS":     None,                    

    # Task & template
    "TASK":     "multi",                  # choose "binary" or "multi"
    "TEMPLATE": "steps",                  # template: "basic" / "explain" / "steps"

    # LLM settings
    "MODEL_ID":  "LLM-Research/Llama-4-Maverick-17B-128E-Instruct",
    "BASE_URL":  "https://api-inference.modelscope.cn/v1",
    "API_KEY":   "",  
    "TEMPERATURE": 0.0,
    "MAX_TOKENS":  1024,
    "RETRIES":     3,
}

CONFIG.update({
    "RESUME": True,           
    "RATE_LIMIT_QPS": 0.18,   
    "PAUSE_EVERY": 100,       
    "PAUSE_SECONDS": 45,      
    "AUTO_SAVE_EVERY": 25,    
    "RETRIES": 12,            
})

# If you want to run multiple experiment groups at once, uncomment MULTI_EXPERIMENTS and set RUN_MULTI_EXPERIMENTS=True
RUN_MULTI_EXPERIMENTS = True
MULTI_EXPERIMENTS = [
    {"TASK": "binary", "TEMPLATE": "basic",   "OUT_CSV": "preds_binary_basic.csv"},
    {"TASK": "binary", "TEMPLATE": "explain", "OUT_CSV": "preds_binary_explain.csv"},
    {"TASK": "binary", "TEMPLATE": "steps",   "OUT_CSV": "preds_binary_steps.csv"},
    {"TASK": "multi",  "TEMPLATE": "basic",   "OUT_CSV": "preds_multi_basic.csv"},
    {"TASK": "multi",  "TEMPLATE": "explain", "OUT_CSV": "preds_multi_explain.csv"},
    {"TASK": "multi",  "TEMPLATE": "steps",   "OUT_CSV": "preds_multi_steps.csv"},
]

# ============================
# Prompt templates
# ============================
PROMPTS = {
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

# ============= Label sets & parsing regex =============
LABELS_BINARY = ["FR", "NFR"]
LABELS_MULTI  = ["FR","A","L","LF","MN","O","PE","SC","SE","US","FT","PO"]

BIN_TAG_RE   = re.compile(r"<label>\s*(FR|NFR)\s*</label>", re.IGNORECASE)
MULTI_TAG_RE = re.compile(r"<label>\s*(FR|A|L|LF|MN|O|PE|SC|SE|US|FT|PO)\s*</label>", re.IGNORECASE)

def extract_label(text: str, task: str) -> str:
    """Parse the final <label>...</label> from model output; if it fails, do a relaxed match; otherwise return 'UNK'."""
    if not text:
        return "UNK"
    t = text.strip()
    if task == "binary":
        m = BIN_TAG_RE.search(t)
        if m: return m.group(1).upper()
        up = t.upper()
        if "NFR" in up: return "NFR"
        if "FR"  in up: return "FR"
        return "UNK"
    else:
        m = MULTI_TAG_RE.search(t)
        if m: return m.group(1).upper()
        up = t.upper()
        for lab in ["LF","MN","PE","SC","SE","US","FT","PO","FR","A","L","O"]:
            if re.search(rf"\b{lab}\b", up): 
                return lab
        return "UNK"

def normalize_gold(label: str, task: str) -> str:
    """Normalize gold labels to the prediction set; binary: FR/NFR; multi: 12 ALL-CAPS categories."""
    if label is None:
        return "UNK"
    s_up = str(label).strip().upper()
    if task == "binary":
        return "FR" if s_up == "FR" else "NFR"
    else:
        if s_up in ("F", "FR"): return "FR"
        if s_up == "PO" or s_up == "PO": return "PO"
        return s_up

def ensure_requirement_column(df: pd.DataFrame) -> str:
    for cand in ["Requirement", "requirement", "text", "Text", "req", "Req"]:
        if cand in df.columns: return cand
    raise ValueError("The requirement text column was not found.")

# ================= OpenAI-compatible client (ModelScope) =================
from openai import OpenAI

_last_call_ts = 0.0

def rate_limit_sleep(qps: float):
    """Ensure the interval between two calls is >= 1/qps seconds; ignored if qps<=0."""
    global _last_call_ts
    if qps and qps > 0:
        min_interval = 1.0 / qps
        now = time.time()
        wait = (_last_call_ts + min_interval) - now
        if wait > 0:
            time.sleep(wait)
        _last_call_ts = time.time()

def make_client(base_url: str, api_key: str):
    if not api_key:
        raise RuntimeError("Lack of ModelScope API Key")
    return OpenAI(base_url=base_url, api_key=api_key)

def build_messages(task: str, template: str, req_text: str):
    tpl = PROMPTS[task][template]
    sys_msg  = {"role": "system", "content": tpl["system"]}
    usr_msg  = {"role": "user", "content": tpl["user"].format(requirement=req_text)}
    return [sys_msg, usr_msg]

def call_llm(client, model_id, messages, temperature=0.0, max_tokens=1024, retries=3, retry_base=1.8):
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return resp.choices[0].message.content
        except openai.RateLimitError as e:
            retry_after = None
            try:
                retry_after = int(getattr(e, "response", None).headers.get("retry-after"))
            except Exception:
                retry_after = None
            delay = retry_after if retry_after and retry_after > 0 else (retry_base ** i) * 2.0
            delay = max(5.0, delay) + random.uniform(0, 1.5)
            print(f"[429] Rate limited. Sleep {delay:.1f}s then retry ({i+1}/{retries})")
            time.sleep(delay)
            continue
        except Exception as e:
            if i == retries - 1:
                raise
            delay = (retry_base ** i) + random.uniform(0, 0.8)
            print(f"[Retry] {type(e).__name__}: {e}. Sleep {delay:.1f}s ({i+1}/{retries})")
            time.sleep(delay)
    return ""

# ======================= Run a single experiment ==========================
def run_one_experiment(csv_path: str, out_csv: str, task: str, template: str,
                       base_url: str, api_key: str, model_id: str,
                       temperature: float, max_tokens: int, retries: int, nrows=None):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if nrows:
        df = df.head(nrows).copy()
    req_col = ensure_requirement_column(df)
    if "Specific_Type" not in df.columns:
        raise ValueError("Didnot find 'Specific_Type'。")

    preds, raws, start_idx = [], [], 0
    if out_csv and CONFIG.get("RESUME") and os.path.exists(out_csv):
        try:
            old = pd.read_csv(out_csv, encoding="utf-8-sig")
            if "pred" in old.columns and len(old) <= len(df):
                preds = old["pred"].astype(str).tolist()
                raws  = old.get("raw_output", pd.Series([""]*len(preds))).astype(str).tolist()
                start_idx = len(preds)
                print(f"[Resume] has had {start_idx}/{len(df)}, keeping running")
        except Exception:
            pass

    golds = [normalize_gold(x, task) for x in df["Specific_Type"].tolist()]

    client = make_client(base_url, api_key)
    preds, raws = [], []

    print(f"\n=== Running experiment: task={task}, template={template}, model={model_id} ===")
    print(f"Data: {csv_path} (n={len(df) if nrows is None else nrows})")
    
    for idx, req in enumerate(tqdm(df[req_col].astype(str).tolist(), desc="Infer")):
        if idx < start_idx:
            continue
    
        rate_limit_sleep(CONFIG["RATE_LIMIT_QPS"])

        msgs = build_messages(task, template, req)
        raw = call_llm(
            client, model_id, msgs,
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"],
            retries=CONFIG["RETRIES"]
        )
        lab = extract_label(raw, task)
        preds.append(lab)
        raws.append(raw)
    
        if CONFIG["PAUSE_EVERY"] and (idx + 1) % CONFIG["PAUSE_EVERY"] == 0:
            print(f"[CoolDown] has dealt with {idx+1}，waiting for {CONFIG['PAUSE_SECONDS']}s to avoid rate limiting")
            time.sleep(CONFIG["PAUSE_SECONDS"])

        if out_csv and CONFIG["AUTO_SAVE_EVERY"] and (idx + 1) % CONFIG["AUTO_SAVE_EVERY"] == 0:
            partial_df = df.iloc[:idx+1].copy()
            labels_order = LABELS_BINARY if task == "binary" else LABELS_MULTI
            valid_set = set(labels_order)
            gold_eval  = [g if g in valid_set else "UNK" for g in golds[:idx+1]]
            pred_eval  = [p if p in valid_set else "UNK" for p in preds]
            partial_df["gold_norm"] = gold_eval
            partial_df["pred"]      = pred_eval
            partial_df["raw_output"]= raws
            partial_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[AutoSave] has writen {out_csv} {idx+1}")

    unk_rate = sum(p == "UNK" for p in preds) / len(preds) if preds else 0.0
    if unk_rate > 0:
        print(f"[Warn] UNK rate = {unk_rate:.2%}")
    
    labels_order = LABELS_BINARY if task == "binary" else LABELS_MULTI
    valid_set = set(labels_order)
    gold_eval = [g if g in valid_set else "UNK" for g in golds[:len(preds)]]
    pred_eval = [p if p in valid_set else "UNK" for p in preds]
    labels_with_unk = labels_order + (["UNK"] if ("UNK" in gold_eval or "UNK" in pred_eval) else [])
    
    print("\n==== Classification Report ====")
    print(classification_report(gold_eval, pred_eval, digits=4, labels=labels_with_unk, zero_division=0))
    
    print("==== Confusion Matrix (rows=true, cols=pred) ====")
    cm = confusion_matrix(gold_eval, pred_eval, labels=labels_with_unk)
    cm_df = pd.DataFrame(cm, index=[f"T:{l}" for l in labels_with_unk], columns=[f"P:{l}" for l in labels_with_unk])
    print(cm_df)
    
    out_df = df.iloc[:len(preds)].copy()
    out_df["gold_norm"] = gold_eval
    out_df["pred"]      = pred_eval
    out_df["raw_output"]= raws
    if out_csv:
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nSaved predictions to: {out_csv}")

# ======================= Entry point ==========================
if __name__ == "__main__":
    if RUN_MULTI_EXPERIMENTS:
        for exp in MULTI_EXPERIMENTS:
            run_one_experiment(
                csv_path=CONFIG["CSV_PATH"],
                out_csv=exp.get("OUT_CSV", "preds.csv"),
                task=exp["TASK"],
                template=exp["TEMPLATE"],
                base_url=CONFIG["BASE_URL"],
                api_key=CONFIG["API_KEY"],
                model_id=CONFIG["MODEL_ID"],
                temperature=CONFIG["TEMPERATURE"],
                max_tokens=CONFIG["MAX_TOKENS"],
                retries=CONFIG["RETRIES"],
                nrows=CONFIG["NROWS"],
            )
    else:
        run_one_experiment(
            csv_path=CONFIG["CSV_PATH"],
            out_csv=CONFIG["OUT_CSV"],
            task=CONFIG["TASK"],
            template=CONFIG["TEMPLATE"],
            base_url=CONFIG["BASE_URL"],
            api_key=CONFIG["API_KEY"],
            model_id=CONFIG["MODEL_ID"],
            temperature=CONFIG["TEMPERATURE"],
            max_tokens=CONFIG["MAX_TOKENS"],
            retries=CONFIG["RETRIES"],
            nrows=CONFIG["NROWS"],
        )
