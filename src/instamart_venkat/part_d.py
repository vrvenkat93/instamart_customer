import json
import re
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import POLICY_DIR, ARTIFACTS, DB_PATH
from .utils import write_json

MIN_SIMILARITY = 0.08

def load_policy_corpus(policy_dir: Path):
    docs = []
    for path in sorted(policy_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt",".md"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append({"source":path.name,"text":text})
    return docs

class PolicyRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = None
        self.matrix = None
        if docs:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
            self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def retrieve(self, query: str, top_k: int = 3):
        if not self.docs:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [{"source":self.docs[i]["source"],"text":self.docs[i]["text"],
                 "score":float(scores[i])} for i in indices if scores[i] >= MIN_SIMILARITY]

def refund_recommendation(ticket_text: str, evidence: list, order_context: dict | None = None):
    if not evidence:
        return {"decision":"human_review","reason":"No sufficiently relevant policy evidence was retrieved.","policy_sources":[]}
    combined = " ".join(x["text"].lower() for x in evidence)
    t = ticket_text.lower()
    # Conservative policy-aware heuristic; never claims a payment was made.
    mentions_refund = any(w in t for w in ("refund","money back","return","charged"))
    has_eligibility = any(w in combined for w in ("eligible","refund","return","within","days"))
    if not mentions_refund:
        decision = "human_review"
        reason = "Ticket does not clearly request a refund."
    elif has_eligibility:
        decision = "human_review"
        reason = "Potentially relevant policy found; verify order date, item condition and policy exceptions before approval."
    else:
        decision = "human_review"
        reason = "Policy evidence is insufficient to determine eligibility."
    return {"decision":decision,"reason":reason,"policy_sources":[x["source"] for x in evidence],
            "order_context_received":bool(order_context)}

def _init_state_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS agent_runs (
            run_id TEXT PRIMARY KEY, ticket_text TEXT NOT NULL, status TEXT NOT NULL,
            step INTEGER NOT NULL, trace_json TEXT NOT NULL, result_json TEXT
        )""")

def run_react(ticket_text: str, retriever: PolicyRetriever, run_id: str | None = None, resume: bool = False):
    _init_state_db()
    run_id = run_id or str(uuid.uuid4())
    trace = []
    with sqlite3.connect(DB_PATH) as conn:
        saved = conn.execute("SELECT step, trace_json, status FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
        if resume and saved:
            step, raw_trace, status = saved
            trace = json.loads(raw_trace)
            if status == "completed":
                result_row = conn.execute("SELECT result_json FROM agent_runs WHERE run_id=?", (run_id,)).fetchone()
                return json.loads(result_row[0]), trace, run_id
        else:
            conn.execute("INSERT OR REPLACE INTO agent_runs VALUES (?,?,?,?,?,?)",
                         (run_id, ticket_text, "running", 0, "[]", None))
    if not trace:
        trace.append({"type":"Thought","content":"Retrieve policy evidence before making a recommendation."})
    if len(trace) <= 1:
        trace.append({"type":"Action","content":"TF-IDF policy retrieval"})
        evidence = retriever.retrieve(ticket_text)
        trace.append({"type":"Observation","content":[{"source":e["source"],"score":e["score"]} for e in evidence]})
    else:
        evidence = retriever.retrieve(ticket_text)
    if not any(x["type"]=="Thought" and "decision" in x["content"].lower() for x in trace):
        trace.append({"type":"Thought","content":"Use retrieved evidence conservatively; uncertain refund cases require human review."})
    result = refund_recommendation(ticket_text, evidence)
    trace.append({"type":"Final","content":result})
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE agent_runs SET status='completed', step=?, trace_json=?, result_json=? WHERE run_id=?",
                     (len(trace), json.dumps(trace), json.dumps(result), run_id))
    return result, trace, run_id

def run_part_d():
    docs = load_policy_corpus(POLICY_DIR)
    retriever = PolicyRetriever(docs)
    # Exercise the assistant on a small sample of already-masked triage records.
    triage_path = ARTIFACTS/"ticket_triage.csv"
    tickets = pd.read_csv(triage_path) if triage_path.exists() else pd.DataFrame()
    outputs = []
    if not tickets.empty:
        for _, row in tickets.head(25).iterrows():
            result, trace, run_id = run_react(str(row["masked_text"]), retriever)
            outputs.append({"ticket_id":row.get("ticket_id"),"run_id":run_id,"result":result,"trace":trace})
    write_json(ARTIFACTS/"policy_assistant_results.json", {"policy_documents":len(docs),"results":outputs})
    print(f"[Part D] Policy docs={len(docs)}; agent runs={len(outputs)}")
    return outputs
