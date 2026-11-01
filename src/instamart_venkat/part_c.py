import json
import re
import time
from pathlib import Path

import pandas as pd

from .config import ALIASES, ARTIFACTS
from .utils import load_csvs, resolve_column, write_json

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\w)")
TOKEN_BUDGET = 12000

def mask_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL_REDACTED]", str(text))
    return PHONE_RE.sub("[PHONE_REDACTED]", text)

def estimate_tokens(text: str) -> int:
    # Conservative offline estimate; not a tokenizer-specific exact count.
    return max(1, (len(str(text)) + 3) // 4)

def deterministic_classify(text: str) -> dict:
    t = str(text).lower()
    rules = [
        (("refund","money back","reversal"),"refund","high","payments"),
        (("late","delay","not arrived","delivery"),"delivery","medium","delivery"),
        (("damaged","broken","spoiled","quality"),"product_quality","high","quality"),
        (("payment failed","charged twice","double charge"),"payment","urgent","payments"),
        (("app","login","otp","technical"),"technical","medium","technical"),
    ]
    for terms, category, priority, queue in rules:
        if any(term in t for term in terms):
            return {"category":category,"priority":priority,"sentiment":"negative","queue":queue}
    return {"category":"other","priority":"low","sentiment":"neutral","queue":"general"}

def _mock_call(prompt: str):
    """Compatibility bridge for common interfaces; returns None if no known API is exposed."""
    try:
        import mock_llm
    except Exception:
        return None
    for name in ("classify_ticket","complete","generate","chat","call_llm","mock_completion"):
        fn = getattr(mock_llm, name, None)
        if callable(fn):
            try:
                result = fn(prompt)
                if isinstance(result, dict): return result
                if isinstance(result, str):
                    try: return json.loads(result)
                    except json.JSONDecodeError: return {"raw_response":result}
            except TypeError:
                continue
            except Exception:
                raise
    return None

def call_with_retries(prompt: str, max_attempts: int = 3, base_delay: float = .2):
    last_error = None
    for attempt in range(max_attempts):
        try:
            result = _mock_call(prompt)
            if result is not None:
                return result
            return None
        except (TimeoutError, ConnectionError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(base_delay * (2 ** attempt))
    if last_error:
        raise last_error
    return None

def run_part_c(data_dir: Path):
    tables = load_csvs(data_dir)
    ticket_pair = next(((n,d) for n,d in tables.items() if "ticket" in n), None)
    if ticket_pair is None:
        print("[Part C] No ticket CSV found; skipped.")
        return pd.DataFrame()
    name, df = ticket_pair
    text_col = resolve_column(df, "text", ALIASES)
    id_col = resolve_column(df, "ticket_id", ALIASES)
    if not text_col:
        raise ValueError(f"Ticket file {name} has no recognizable text/message column.")
    records, used_tokens = [], 0
    for i, row in df.iterrows():
        safe_text = mask_pii(row[text_col])
        cost = estimate_tokens(safe_text)
        if used_tokens + cost > TOKEN_BUDGET:
            result = {"category":"budget_exceeded","priority":"medium","sentiment":"unknown","queue":"manual_review"}
            source = "budget_guard"
        else:
            used_tokens += cost
            result = None
            try:
                response = call_with_retries(safe_text)
                if isinstance(response, dict):
                    # Accept only expected classification keys from mock output.
                    result = {k:response.get(k) for k in ("category","priority","sentiment","queue")}
                    if not result.get("category") or not result.get("priority"):
                        result = None
                    else:
                        result = {k:(v if v is not None else "unknown") for k,v in result.items()}
            except Exception:
                result = None
            if result is None:
                result = deterministic_classify(safe_text)
                source = "local_fallback"
            else:
                source = "mock_llm"
        # Routing safety overrides.
        if result.get("priority") == "urgent" or result.get("category") in {"payment","safety"}:
            result["queue"] = "human_priority_review"
        records.append({
            "ticket_id": row[id_col] if id_col else i,
            "masked_text": safe_text,
            **result,
            "classification_source":source,
            "estimated_tokens":cost,
            "pii_masked":safe_text != str(row[text_col]),
        })
    out = pd.DataFrame(records)
    out.to_csv(ARTIFACTS/"ticket_triage.csv", index=False)
    write_json(ARTIFACTS/"ticket_budget.json", {"budget":TOKEN_BUDGET,"estimated_tokens_used":used_tokens,
                                                "remaining":TOKEN_BUDGET-used_tokens,"tickets":len(out)})
    print(f"[Part C] Triaged {len(out)} tickets; estimated tokens={used_tokens}/{TOKEN_BUDGET}")
    return out
