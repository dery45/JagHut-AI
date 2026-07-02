"""
query_logger.py — Structured per-query logging untuk Jaghut v0.1L

Setiap query menghasilkan satu log entry JSON di logs/queries.jsonl
Format JSONL (newline-delimited JSON) agar mudah di-grep, di-tail, dan
di-load ke pandas untuk analisis.

Fields per entry:
  ts            — ISO timestamp
  session_id    — ID sesi dari main.py
  query_id      — UUID unik per query
  question      — pertanyaan asli user
  rewritten     — hasil query rewriting (sama jika tidak ada history)
  scope_passed  — apakah lolos scope guard
  flags         — dict: is_plant, is_weather, has_history
  docs_retrieved — list ringkasan dokumen yang dipakai (source, score, snippet)
  answer_preview — 200 karakter pertama dari jawaban LLM
  eval          — dict hasil eval loop: faithfulness, relevance, flag
  latency_ms    — total waktu proses dalam milidetik
  error         — string error jika ada, null jika tidak
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR  = _ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "queries.jsonl"
EVAL_FILE = LOG_DIR / "eval_flags.jsonl"  # hanya entry yang flagged


def _ensure_log_dir():
    LOG_DIR.mkdir(exist_ok=True)


def new_query_trace(session_id: str) -> dict:
    """
    Buat trace baru untuk satu query. Panggil di awal,
    isi field-nya satu per satu, lalu panggil commit_trace() di akhir.
    """
    return {
        "ts":             datetime.now().isoformat(),
        "session_id":     session_id,
        "query_id":       str(uuid.uuid4())[:8],
        "question":       "",
        "rewritten":      "",
        "scope_passed":   None,
        "flags":          {},
        "docs_retrieved": [],
        "answer_preview": "",
        "eval":           {},
        "latency_ms":     0,
        "error":          None,
        "_start_ts":      time.monotonic(),   # internal, tidak disimpan
    }


def set_docs(trace: dict, docs: list) -> None:
    """
    Isi docs_retrieved dari list LangChain Document.
    Simpan source, chunk_index, dan 120 karakter pertama konten.
    """
    trace["docs_retrieved"] = [
        {
            "source":      doc.metadata.get("source", "unknown"),
            "sheet":       doc.metadata.get("sheet", ""),
            "chunk_index": doc.metadata.get("chunk_index", ""),
            "snippet":     doc.page_content[:120].replace("\n", " "),
        }
        for doc in docs
    ]


def commit_trace(trace: dict, error: Optional[str] = None) -> None:
    """
    Finalisasi dan tulis trace ke JSONL. Hapus field internal.
    """
    _ensure_log_dir()
    trace["latency_ms"] = int((time.monotonic() - trace.pop("_start_ts", 0)) * 1000)
    trace["error"]      = error

    # Tulis ke queries.jsonl
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # Kalau ada flag dari eval, tulis juga ke eval_flags.jsonl
    if trace.get("eval", {}).get("flag"):
        with open(EVAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    _print_trace_summary(trace)


def _print_trace_summary(trace: dict) -> None:
    """Cetak ringkasan singkat ke terminal setelah tiap query."""
    ev    = trace.get("eval", {})
    flag  = ev.get("flag", False)
    fth   = ev.get("faithfulness", "n/a")
    rel   = ev.get("relevance",    "n/a")
    lat   = trace.get("latency_ms", 0)
    docs  = len(trace.get("docs_retrieved", []))
    icon  = "🚩" if flag else "✅"
    print(f"\n{icon} [LOG {trace['query_id']}] "
          f"lat={lat}ms docs={docs} faith={fth} rel={rel}")
    if flag:
        print(f"   ⚠️  Flagged: {ev.get('reason', '')}")


# ─── Log reader helpers ───────────────────────────────────────────────────────

def tail_logs(n: int = 20) -> list[dict]:
    """Ambil n entry terakhir dari queries.jsonl."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]


def flagged_logs() -> list[dict]:
    """Ambil semua entry yang di-flag oleh eval loop."""
    if not EVAL_FILE.exists():
        return []
    lines = EVAL_FILE.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines]


def session_logs(session_id: str) -> list[dict]:
    """Ambil semua entry untuk satu session_id."""
    if not LOG_FILE.exists():
        return []
    results = []
    for line in LOG_FILE.read_text(encoding="utf-8").strip().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("session_id") == session_id:
                results.append(entry)
        except json.JSONDecodeError:
            continue
    return results


def print_debug_report(n: int = 10) -> None:
    """
    Cetak ringkasan debug ke terminal.
    Panggil dengan: python -c "from query_logger import print_debug_report; print_debug_report()"
    """
    entries  = tail_logs(n)
    flagged  = flagged_logs()
    total    = len(entries)
    n_flag   = sum(1 for e in entries if e.get("eval", {}).get("flag"))
    avg_lat  = int(sum(e.get("latency_ms", 0) for e in entries) / max(total, 1))
    avg_docs = sum(len(e.get("docs_retrieved", [])) for e in entries) / max(total, 1)

    print("\n" + "="*55)
    print(f"  SUGI Query Debug Report — last {total} queries")
    print("="*55)
    print(f"  Flagged by eval : {n_flag} / {total}")
    print(f"  Avg latency     : {avg_lat} ms")
    print(f"  Avg docs used   : {avg_docs:.1f}")
    print("-"*55)
    for e in entries:
        ev   = e.get("eval", {})
        icon = "🚩" if ev.get("flag") else "  "
        print(f"  {icon} [{e['query_id']}] {e['question'][:45]:<45} "
              f"faith={ev.get('faithfulness','?')} rel={ev.get('relevance','?')} "
              f"{e.get('latency_ms',0)}ms")
    print("="*55)
    if flagged:
        print(f"\n  Total flagged entries in eval_flags.jsonl: {len(flagged)}")
    print()