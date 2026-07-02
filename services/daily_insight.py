"""
daily_insight.py — Jaghut ESG Insight Engine
=============================================
Menghasilkan insight harian dari data RAG lokal (ChromaDB) dan mengirimkannya
ke MongoDB dalam koleksi terpisah per kategori.

Koleksi MongoDB:
  jaghut_insights.esg_insights         — skor ESG, metrik Forest IQ, kinerja perusahaan
  jaghut_insights.climate_insights     — risiko iklim & cuaca ekstrem
  jaghut_insights.company_insights     — wawasan sektoral ESG
  jaghut_insights.sector_insights      — tren ESG lintas sektor
  jaghut_insights.session_summaries    — ringkasan sesi percakapan dari main.py

Jadwal:
  - Selalu kirim saat pertama kali dijalankan
  - Setelah itu kirim otomatis setiap 12 jam (bisa ubah INTERVAL_HOURS)

Env (.env):
  MONGO_URI   — MongoDB connection string (wajib)
  CHROMA_HOST — Host ChromaDB (default: localhost)
  CHROMA_PORT — Port ChromaDB (default: 8000)

Cara pakai:
  python daily_insight.py               # jalankan & biarkan loop
  python daily_insight.py --once        # kirim sekali lalu exit
  python daily_insight.py --interval 6  # loop setiap 6 jam
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import ssl
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from typing import Any
import concurrent.futures

# Suppress pydantic V1 deprecation warnings on Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

# ── Env loading ───────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / "config" / ".env")
except ImportError:
    pass  # dotenv opsional

MONGO_URI      = os.getenv("MONGO_URI", "").strip()
CHROMA_HOST    = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT    = int(os.getenv("CHROMA_PORT", "8000"))
INTERVAL_HOURS = float(os.getenv("INSIGHT_INTERVAL_HOURS", "12"))
BATCH_SIZE     = 500
MAX_WORKERS    = 5

if not MONGO_URI:
    print("❌  MONGO_URI tidak di-set di .env atau environment variable.")
    print("    Isi MONGO_URI di file config/.env dan coba lagi.")
    sys.exit(1)

# ── Debug: print OpenSSL version ──────────────────────────────────────────────
print(f"🔒  OpenSSL version: {ssl.OPENSSL_VERSION}")

# ── MongoDB ───────────────────────────────────────────────────────────────────
try:
    import certifi
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import PyMongoError
except ImportError:
    print("❌  pymongo / certifi belum terinstall.")
    print("    Jalankan: pip install pymongo certifi")
    sys.exit(1)


def _build_mongo_client() -> MongoClient:
    """
    Buat MongoClient dengan konfigurasi TLS yang benar untuk MongoDB Atlas.

    - tlsCAFile=certifi.where()      → pakai CA bundle dari certifi (up-to-date)
    - tlsAllowInvalidCertificates=False → validasi sertifikat dengan benar (JANGAN True)
    - tls=True                       → aktifkan TLS secara eksplisit
    - retryWrites=True               → retry otomatis pada transient error
    """
    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=15_000,
        socketTimeoutMS=20_000,
        connectTimeoutMS=20_000,
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=False,   # ← WAJIB False agar handshake benar
        retryWrites=True,
        retryReads=True,
    )


import time as _time

_mongo_client = None
for _attempt in range(3):
    try:
        _mongo_client = _build_mongo_client()
        _mongo_client.admin.command("ping")
        print("✅  MongoDB Atlas terhubung.")
        break
    except Exception as e:
        if _attempt < 2:
            print(f"⚠️  MongoDB connection attempt {_attempt+1} failed: {e}")
            print(f"    Retrying in {3*(_attempt+1)}s...")
            _time.sleep(3 * (_attempt + 1))
        else:
            print(f"❌  Gagal koneksi ke MongoDB Atlas setelah 3 percobaan: {e}")
            print("    Pastikan:")
            print("    1. MONGO_URI benar di .env")
            print("    2. IP kamu sudah di-whitelist di Atlas Network Access")
            print("    3. pymongo & certifi sudah di-upgrade: pip install -U pymongo certifi")
            sys.exit(1)

if _mongo_client is None:
    sys.exit(1)

_db          = _mongo_client["jaghut_insights"]
COL_ESG      = _db["esg_insights"]
COL_CLIMATE  = _db["climate_insights"]
COL_SECTOR   = _db["sector_insights"]
COL_POLICY   = _db["policy_insights"]
COL_SESSION  = _db["session_summaries"]

# Buat index unik agar upsert cepat
for _col in [COL_ESG, COL_CLIMATE, COL_SECTOR, COL_POLICY, COL_SESSION]:
    try:
        _col.create_index("insight_id", unique=True, background=True)
    except PyMongoError:
        pass  # index mungkin sudah ada

# ── ChromaDB ──────────────────────────────────────────────────────────────────
try:
    import chromadb
except ImportError:
    print("❌  chromadb belum terinstall. Jalankan: pip install chromadb")
    sys.exit(1)

try:
    _chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _chroma.heartbeat()
    print(f"✅  ChromaDB terhubung di {CHROMA_HOST}:{CHROMA_PORT}")
except Exception as e:
    print(f"❌  Tidak bisa konek ke ChromaDB: {e}")
    sys.exit(1)

# ── Ollama LLM ────────────────────────────────────────────────────────────────
try:
    from langchain_ollama.llms import OllamaLLM
    _llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.4, repeat_penalty=1.1)
    print("✅  LLM (qwen2.5:1.5b) siap.")
except Exception as e:
    _llm = None
    print(f"⚠️   LLM tidak tersedia ({e}) — insight akan berupa ringkasan data mentah.")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_all_docs(collection_name: str) -> list[dict]:
    """
    Fetch semua dokumen dari koleksi ChromaDB dengan batching.
    Return list of {text, meta}.
    """
    try:
        col = _chroma.get_collection(collection_name)
    except Exception as e:
        print(f"   ⚠️  Koleksi '{collection_name}' tidak ditemukan: {e}")
        return []

    try:
        all_ids = col.get(include=[]).get("ids", [])
    except Exception as e:
        print(f"   ⚠️  Gagal fetch IDs dari '{collection_name}': {e}")
        return []

    if not all_ids:
        return []

    docs: list[dict] = []
    for offset in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[offset: offset + BATCH_SIZE]
        try:
            batch = col.get(ids=batch_ids, include=["documents", "metadatas"])
            for text, meta in zip(
                batch.get("documents", []),
                batch.get("metadatas", []),
            ):
                docs.append({"text": text or "", "meta": meta or {}})
        except Exception as e:
            print(f"   ⚠️  Batch {offset}–{offset + BATCH_SIZE} gagal: {e}")
            continue

    return docs


def _ask_llm(prompt: str, fallback: str = "") -> str:
    """Invoke LLM. Kembalikan fallback jika LLM tidak tersedia atau error."""
    if _llm is None:
        return fallback
    try:
        return _llm.invoke(prompt).strip()
    except Exception as e:
        print(f"   ⚠️  LLM error: {e}")
        return fallback


def _doc_hash(text: str, category: str, scope: str) -> str:
    """Buat ID unik deterministik dari konten + kategori + scope (tanggal/session)."""
    raw = f"{category}|{scope}|{text[:200]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _upsert_many(collection, docs: list[dict]) -> None:
    """
    Upsert batch ke MongoDB.
    Pakai $setOnInsert agar dokumen yang sudah ada tidak di-overwrite.
    Tambah retry sederhana (3x) untuk transient network error.
    """
    if not docs:
        return

    ops = [
        UpdateOne(
            {"insight_id": d["insight_id"]},
            {"$setOnInsert": d},
            upsert=True,
        )
        for d in docs
    ]

    for attempt in range(1, 4):
        try:
            result = collection.bulk_write(ops, ordered=False)
            inserted = result.upserted_count
            skipped  = len(docs) - inserted
            print(f"   ✅  {inserted} baru ditambahkan / {skipped} sudah ada")
            return
        except PyMongoError as e:
            print(f"   ⚠️  MongoDB upsert attempt {attempt}/3 gagal: {e}")
            if attempt < 3:
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s
            else:
                print(f"   ❌  Upsert gagal setelah 3 percobaan: {e}")


def _base_doc(category: str, text: str, date_str: str, **extra) -> dict:
    """Buat skeleton dokumen insight dengan field standar."""
    now = datetime.now(timezone.utc)
    return {
        "insight_id":     _doc_hash(text, category, date_str),
        "category":       category,
        "generated_at":   now,
        "generator_ver":  "jaghut-daily-insight-v1",
        "date":           date_str,
        "date_quarter":   _QUARTER_MAP.get(now.month, "Unknown"),
        "date_month":     now.strftime("%Y-%m"),
        "date_iso":       now.isoformat(),
        "raw_text":       text[:2000],
        "model_used":     "qwen2.5:1.5b" if _llm else None,
        "data_available": True,
        **extra,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Kuartal & sektor ESG
# ═════════════════════════════════════════════════════════════════════════════

_QUARTER_MAP: dict[int, str] = {
    1: "Q1",  2: "Q1",  3: "Q1",
    4: "Q2",  5: "Q2",  6: "Q2",
    7: "Q3",  8: "Q3",  9: "Q3",
    10: "Q4", 11: "Q4", 12: "Q4",
}

_ESG_SECTORS: list[str] = [
    "palm oil", "pulp and paper", "mining", "energy",
    "manufacturing", "agriculture", "forestry", "plantation",
    "banking", "infrastructure",
]


# ═════════════════════════════════════════════════════════════════════════════
# Insight Generators
# ═════════════════════════════════════════════════════════════════════════════

def generate_company_insights(date_str: str) -> list[dict]:
    """
    Ambil dokumen ESG dari ChromaDB, kelompokkan per perusahaan/sektor,
    minta LLM buat ringkasan insight per grup.
    """
    print("\n🏢  Generating company insights...")
    docs = _fetch_all_docs("main_dataset")

    esg_keywords = {
        "esg", "environmental", "social", "governance", "forest iq",
        "score", "rating", "emission", "carbon", "deforestation",
        "sustainability", "disclosure", "compliance", "risk",
    }
    esg_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in esg_keywords)
        and d["meta"].get("type") in ("esg", "price", "tabular", None)
    ]

    if not esg_docs:
        print("   ℹ️  Tidak ada dokumen ESG ditemukan.")
        return []

    groups: dict[str, list[str]] = {}
    for d in esg_docs:
        company = (
            d["meta"].get("company")
            or d["meta"].get("province")
            or d["meta"].get("region")
            or "Nasional"
        )
        groups.setdefault(company, []).append(d["text"])

    def _process(company: str, texts: list[str]) -> dict:
        combined = "\n".join(texts[:10])
        prompt = (
            f"Kamu adalah analis ESG untuk pasar Indonesia. "
            f"Berdasarkan data berikut untuk {company}, "
            f"buat insight singkat (3-5 kalimat) mencakup: "
            f"skor ESG terkini, metrik risiko lingkungan, "
            f"dan rekomendasi perbaikan.\n\n"
            f"Data:\n{combined[:1500]}\n\nInsight:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Data ESG tersedia untuk {company}: {combined[:300]}",
        )
        # Extract source files and data types from available metadata
        sources_seen = set()
        data_types_seen = set()
        for doc_text in texts:
            for line in doc_text.split("\n"):
                if line.startswith("source: "):
                    sources_seen.add(line[8:])
                if line.startswith("sheet: "):
                    pass
        return _base_doc(
            category      = "company_insight",
            text          = combined,
            date_str      = date_str,
            company       = company,
            company_normalized = company.lower().replace(" ", "_"),
            region        = "Indonesia",
            sector        = "unknown",
            insight       = insight,
            doc_count     = len(texts),
            data_types    = sorted(data_types_seen) if data_types_seen else ["esg"],
            source_files  = sorted(sources_seen) if sources_seen else ["main_dataset"],
            data_quality  = "sufficient" if len(texts) >= 5 else "limited",
            confidence    = "high" if len(texts) >= 10 else ("medium" if len(texts) >= 3 else "low"),
            has_esg_scores = any("score" in t.lower() or "rating" in t.lower() for t in texts),
            has_forest_iq  = any("forest iq" in t.lower() or "forestiq" in t.lower() for t in texts),
            has_emission_data = any("emission" in t.lower() or "karbon" in t.lower() or "carbon" in t.lower() for t in texts),
            entities_mentioned = [company],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(lambda kv: _process(*kv), groups.items()))

    print(f"   📊  {len(results)} company insights dari {len(esg_docs)} dokumen.")
    return results


def generate_climate_insights(date_str: str) -> list[dict]:
    """Ambil dokumen cuaca/iklim, buat insight per lokasi untuk ESG risk."""
    print("\n🌤️   Generating climate insights...")
    docs = _fetch_all_docs("weather_data")

    if not docs:
        print("   ℹ️  Tidak ada data iklim di weather_data collection.")
        return []

    groups: dict[str, list[str]] = {}
    for d in docs:
        loc = (
            d["meta"].get("location")
            or d["meta"].get("city")
            or d["meta"].get("region")
            or "Indonesia"
        )
        groups.setdefault(loc, []).append(d["text"])

    def _process(location: str, texts: list[str]) -> dict:
        combined = "\n".join(texts[:8])
        prompt = (
            f"Kamu adalah analis risiko iklim untuk ESG. "
            f"Berdasarkan data cuaca berikut untuk {location}, "
            f"buat insight singkat (3-4 kalimat) mencakup: "
            f"kondisi iklim terkini, risiko cuaca ekstrem, "
            f"dan dampaknya terhadap operasional dan rantai pasok.\n\n"
            f"Data:\n{combined[:1200]}\n\nInsight:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Data iklim tersedia untuk {location}: {combined[:300]}",
        )
        # Extract climate indicators from weather data
        extreme_events = []
        avg_temp = None
        total_rain = None
        for t in texts:
            if "extreme" in t.lower() or "alert" in t.lower():
                extreme_events.append(t[:200])
            # Try to extract temperature
            match = re.search(r"Temperature:.*avg\s+([\d.]+)", t)
            if match:
                avg_temp = float(match.group(1))
            match = re.search(r"Rainfall:\s*([\d.]+)\s*mm", t)
            if match:
                if total_rain is None:
                    total_rain = 0
                total_rain += float(match.group(1))
        return _base_doc(
            category        = "climate_insight",
            text            = combined,
            date_str        = date_str,
            location        = location,
            insight         = insight,
            doc_count       = len(texts),
            avg_temperature = avg_temp,
            total_rainfall  = total_rain,
            extreme_events  = extreme_events[:5],
            risk_level      = "high" if any("extreme" in e.lower() for e in extreme_events) else
                              ("medium" if any("alert" in e.lower() for e in extreme_events) else "low"),
            indicators      = {
                "temperature_available": any("temperature" in t.lower() for t in texts),
                "rainfall_available": any("rainfall" in t.lower() or "rain" in t.lower() for t in texts),
                "soil_moisture_available": any("soil moisture" in t.lower() for t in texts),
                "wind_data_available": any("wind" in t.lower() for t in texts),
                "has_extreme_alerts": len(extreme_events) > 0,
            },
            period          = {
                "start": date_str,
                "end": date_str,
                "days_covered": len(texts),
            },
            data_source     = "Open-Meteo API",
            has_alerts      = len(extreme_events) > 0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(lambda kv: _process(*kv), groups.items()))

    print(f"   🌦️   {len(results)} climate insights dari {len(docs)} dokumen.")
    return results


def generate_sector_insights(date_str: str) -> list[dict]:
    """Buat wawasan sektoral ESG per sektor berdasarkan data terkini."""
    print("\n📊  Generating sector insights...")
    docs = _fetch_all_docs("main_dataset")

    sector_keywords = {
        "esg", "sektor", "industri", "perusahaan", "emission", "carbon",
        "deforestation", "sustainability", "risk", "rating",
    }
    sector_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in sector_keywords)
    ]

    now         = datetime.now(timezone.utc)
    quarter     = _QUARTER_MAP.get(now.month, "Unknown")
    bulan_nama  = now.strftime("%B")

    def _process(sector: str) -> dict:
        relevant = [
            d["text"] for d in sector_docs
            if sector.lower() in d["text"].lower()
        ]
        combined = "\n".join(relevant[:6]) if relevant else "(tidak ada data spesifik)"
        prompt = (
            f"Kamu adalah analis ESG sektoral. "
            f"Buat wawasan praktis untuk sektor {sector} "
            f"di {bulan_nama} ({quarter}).\n"
            f"Wawasan harus mencakup: tren ESG terkini, risiko regulasi, "
            f"peluang keberlanjutan, dan peringatan kepatuhan. "
            f"Maksimal 5 poin singkat.\n\n"
            f"Data referensi:\n{combined[:1000]}\n\nWawasan:"
        )
        insight = _ask_llm(
            prompt,
            fallback=f"Wawasan sektor {sector} untuk {quarter}: data terbatas di database.",
        )
        has_data = len(relevant) > 0
        # Count companies mentioned in this sector
        company_keywords = ["wilmar", "april", "pertamina", "adaro", "indofood", "unilever",
                            "nestle", "cargill", "glencore", "freeport", "antam", "vale"]
        companies_in_sector = [c for c in company_keywords if c in combined.lower()]
        return _base_doc(
            category          = "sector_insight",
            text              = combined,
            date_str          = date_str,
            sector            = sector,
            sector_id         = sector.lower().replace(" ", "_"),
            quarter           = quarter,
            quarter_year      = f"{quarter}-{now.year}",
            bulan             = bulan_nama,
            insight           = insight,
            data_available    = has_data,
            doc_count_sector  = len(relevant),
            companies_tracked = companies_in_sector,
            company_count     = len(companies_in_sector),
            key_risks         = ["regulatory risk", "market risk"] if not has_data else None,
            recommendations   = ["Perbanyak data sektoral untuk analisis lebih akurat"] if not has_data else None,
            sector_summary    = {
                "name": sector,
                "data_points": len(relevant),
                "data_quality": "sufficient" if len(relevant) >= 3 else "limited",
            },
            tags              = [sector, "esg", quarter, "sektor"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(_process, _ESG_SECTORS))

    print(f"   🏭  {len(results)} sector insights ({quarter}).")
    return results


def generate_policy_insights(date_str: str) -> list[dict]:
    """Buat 3-5 insight tentang regulasi dan kebijakan ESG terkini."""
    print("\n📰  Generating policy insights...")
    docs = _fetch_all_docs("main_dataset")

    policy_keywords = {
        "kebijakan", "regulasi", "peraturan", "presiden", "menteri",
        "ojk", "idx", "pojk", "ekspor", "impor", "ketahanan pangan",
        "produksi", "produktivitas", "inovasi", "teknologi",
        "pajak karbon", "nett zero", "dekarbonisasi",
    }
    policy_docs = [
        d for d in docs
        if any(kw in d["text"].lower() for kw in policy_keywords)
    ]

    if not policy_docs:
        print("   ℹ️  Tidak ada dokumen kebijakan ESG ditemukan.")
        return []

    combined = "\n\n".join(d["text"] for d in policy_docs[:15])[:3000]
    prompt = (
        "Kamu adalah analis kebijakan ESG Indonesia. "
        "Berdasarkan dokumen-dokumen berikut, identifikasi "
        "3 sampai 5 insight terpenting terkait situasi ESG dan "
        "keberlanjutan di Indonesia saat ini. "
        "Setiap insight harus memiliki judul singkat dan penjelasan 2-3 kalimat. "
        "Format: **[Judul]** — [Penjelasan]\n\n"
        f"Dokumen:\n{combined}\n\nInsight:"
    )
    insight_text = _ask_llm(
        prompt,
        fallback=f"Insight kebijakan: {len(policy_docs)} dokumen tersedia di database.",
    )

    parts = re.split(r"\n(?=\*\*)", insight_text)
    if len(parts) <= 1:
        parts = [insight_text]

    # Extract regulation and policy keywords found in the data
    regulation_keywords_found = set()
    for kw in ["eudr", "ispo", "rspo", "ndpe", "ojk", "pojk", "amdal", "proper",
               "permentan", "carbon tax", "pajak karbon", "net zero", "dekarbonisasi",
               "uu lingkungan", "perpres", "permen", "pp"]:
        if any(kw in d["text"].lower() for d in policy_docs):
            regulation_keywords_found.add(kw)

    results = []
    for i, part in enumerate(parts, 1):
        m       = re.match(r"\*\*(.+?)\*\*\s*[—-]\s*(.*)", part, re.DOTALL)
        title   = m.group(1).strip() if m else f"Insight {i}"
        content = m.group(2).strip() if m else part.strip()
        if not content:
            continue
        # Categorize each insight
        insight_lower = (title + " " + content).lower()
        insight_type = "regulatory"
        if any(w in insight_lower for w in ["kebijakan", "policy", "regulation", "regulasi"]):
            insight_type = "regulatory"
        elif any(w in insight_lower for w in ["market", "pasar", "investasi", "investment"]):
            insight_type = "market"
        elif any(w in insight_lower for w in ["climate", "iklim", "carbon", "karbon", "emission"]):
            insight_type = "climate"
        elif any(w in insight_lower for w in ["social", "sosial", "community", "masyarakat"]):
            insight_type = "social"
        elif any(w in insight_lower for w in ["compliance", "kepatuhan", "enforcement"]):
            insight_type = "compliance"

        results.append(_base_doc(
            category            = "policy_insight",
            text                = combined[:500],
            date_str            = date_str,
            title               = title,
            insight             = content,
            index               = i,
            insight_type        = insight_type,
            doc_count           = len(policy_docs),
            regulations_matched = sorted(regulation_keywords_found),
            region              = "Indonesia",
            policy_area         = "ESG & Sustainability",
            has_carbon_topic    = any(w in insight_lower for w in ["carbon", "karbon", "emission", "emisi", "iklim", "climate"]),
            has_trade_topic     = any(w in insight_lower for w in ["eudr", "ekspor", "impor", "trade", "export", "import"]),
            tags                = [insight_type, "policy", "regulasi"] + ([
                "carbon" if "carbon" in insight_lower or "karbon" in insight_lower else "",
                "climate" if "climate" in insight_lower or "iklim" in insight_lower else "",
                "trade" if "eudr" in insight_lower or "ekspor" in insight_lower else "",
            ]),
        ))

    print(f"   📋  {len(results)} policy insights dari {len(policy_docs)} dokumen.")
    return results


def push_session_summaries(date_str: str) -> list[dict]:
    """
    Ambil ringkasan sesi dari conversation_memory ChromaDB
    dan kirim ke MongoDB session_summaries.
    """
    print("\n💬  Pushing session summaries...")
    docs = _fetch_all_docs("conversation_memory")

    if not docs:
        print("   ℹ️  Tidak ada session summary di conversation_memory.")
        return []

    # Common ESG entity keywords to detect in session text
    _ENTITY_SIGNALS = {
        "wilmar", "april", "sinar mas", "golden agri", "musim mas", "astra agro",
        "pertamina", "pln", "adaro", "freeport", "indofood", "unilever",
        "nestle", "cargill", "glencore", "eudr", "ispo", "rspo", "ndpe",
        "forest iq", "greenwashing", "deforestasi", "carbon", "karbon",
    }

    results = []
    now = datetime.now(timezone.utc)
    for d in docs:
        meta       = d["meta"]
        session_id = meta.get("session_id", "unknown")
        user_id    = meta.get("user_id", "unknown")
        timestamp  = meta.get("timestamp", date_str)
        content    = d["text"]

        # Detect mentioned entities in the summary
        content_lower = content.lower()
        entities_found = [e for e in _ENTITY_SIGNALS if e in content_lower]

        # Estimate question count from summary
        q_indicators = ["pertanyaan", "tanya", "question", "asked", "menanyakan"]
        est_questions = sum(1 for ind in q_indicators if ind in content_lower)

        # Categorize topics
        topics = []
        if any(w in content_lower for w in ["esg", "skor", "score", "rating", "environmental", "social", "governance"]):
            topics.append("ESG")
        if any(w in content_lower for w in ["forest iq", "forest", "deforestation", "deforestasi", "hutan"]):
            topics.append("Forest & Deforestation")
        if any(w in content_lower for w in ["carbon", "karbon", "emission", "emisi", "climate", "iklim", "net zero"]):
            topics.append("Climate & Carbon")
        if any(w in content_lower for w in ["greenwashing", "greenwash", "decoupling"]):
            topics.append("Greenwashing")
        if any(w in content_lower for w in ["eudr", "ispo", "rspo", "regulasi", "regulation", "kebijakan"]):
            topics.append("Regulation & Policy")
        if any(w in content_lower for w in ["supply chain", "rantai pasok", "sawit", "palm oil", "commodity"]):
            topics.append("Supply Chain")

        results.append({
            "insight_id":        _doc_hash(content, "session_summary", session_id),
            "category":          "session_summary",
            "session_id":        session_id,
            "user_id":           user_id,
            "platform":          "telegram" if "telegram" in session_id else "cli",
            "timestamp":         timestamp,
            "pushed_at":         now,
            "summary":           content,
            "char_count":        len(content),
            "word_count":        len(content.split()),
            "estimated_questions": max(est_questions, 1),
            "topics_discussed":  topics,
            "entities_mentioned": entities_found,
            "entity_count":      len(entities_found),
            "topic_count":       len(topics),
            "has_esg_content":   "ESG" in topics or "Forest & Deforestation" in topics,
            "has_climate_content": "Climate & Carbon" in topics,
        })

    print(f"   💾  {len(results)} session summaries akan di-push.")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main runner
# ═════════════════════════════════════════════════════════════════════════════

def run_once() -> None:
    """Generate semua insight dan kirim ke MongoDB."""
    now      = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    print("\n" + "═" * 60)
    print(f"🌿  Jaghut ESG Daily Insight — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═" * 60)

    try:
        tasks: list[tuple[Any, list[dict]]] = [
            (COL_ESG,     generate_company_insights(date_str)),
            (COL_CLIMATE, generate_climate_insights(date_str)),
            (COL_SECTOR,  generate_sector_insights(date_str)),
            (COL_POLICY,  generate_policy_insights(date_str)),
            (COL_SESSION, push_session_summaries(date_str)),
        ]

        total = 0
        for col, docs in tasks:
            if docs:
                _upsert_many(col, docs)
                total += len(docs)

        print(f"\n✅  Selesai — {total} dokumen diproses pada {now.strftime('%H:%M')} UTC")
        print("─" * 60)

    except Exception:
        print("❌  Error tidak terduga saat generate/kirim insight:")
        traceback.print_exc()


def run_loop(interval_hours: float = INTERVAL_HOURS) -> None:
    """Jalankan run_once() saat startup, lalu ulangi setiap interval_hours."""
    interval_sec = interval_hours * 3600
    print(f"⏰  Loop aktif — interval setiap {interval_hours:.1f} jam. "
          f"Ctrl+C untuk berhenti.\n")

    while True:
        run_once()
        next_run = datetime.now(timezone.utc).timestamp() + interval_sec
        remaining = next_run - time.time()
        print(f"⏳  Insight berikutnya dalam ~{remaining / 3600:.1f} jam "
              f"({interval_hours:.0f}h).\n")
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n👋  Loop dihentikan. Sampai jumpa!")
            sys.exit(0)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jaghut ESG Daily Insight — kirim insight ESG ke MongoDB"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Kirim insight sekali lalu exit (tanpa loop otomatis)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=INTERVAL_HOURS,
        metavar="JAM",
        help=f"Interval pengiriman dalam jam (default: {INTERVAL_HOURS})",
    )
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_loop(interval_hours=args.interval)
