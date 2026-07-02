"""
company_insight_engine.py — ForestIQ Company Insight Engine
============================================================
Menghasilkan insight perusahaan dari data assessment ForestIQ di MongoDB
(test.companies + test.assessments + test.commodities) dan diperkaya dengan
data RAG dari ChromaDB.

Alur:
  1. Baca semua companies, assessments, commodities dari DB test
  2. Join berdasarkan _id → companyId → commodityId
  3. Untuk setiap company, kumpulkan semua assessment lintas komoditas
  4. Generate insight via LLM (qwen2.5:1.5b) dengan konteks RAG
  5. Simpan ke test.company_insight + jaghut_insights.company_insight

Monitoring:
  - Cek perubahan pada companies/assessments/commodities via updatedAt
  - Update insight jika ada data baru

Integrasi:
  - Dipanggil dari start_all.py sebagai background service
"""

from __future__ import annotations

import argparse
import hashlib
import os
import ssl
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from typing import Any

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

try:
    from dotenv import load_dotenv
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_ROOT / "config" / ".env")
except ImportError:
    pass

MONGO_URI   = os.getenv("MONGO_URI", "").strip()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
POLL_SECS   = int(os.getenv("COMPANY_INSIGHT_POLL_SECS", "300"))

if not MONGO_URI:
    print("❌  MONGO_URI tidak di-set di .env")
    sys.exit(1)

print(f"🔒  OpenSSL version: {ssl.OPENSSL_VERSION}")

try:
    import certifi
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import PyMongoError
except ImportError:
    print("❌  pymongo / certifi belum terinstall.")
    sys.exit(1)

def _build_mongo_client() -> MongoClient:
    return MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=15_000,
        socketTimeoutMS=20_000,
        connectTimeoutMS=20_000,
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=False,
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
            sys.exit(1)

if _mongo_client is None:
    sys.exit(1)

_db_test    = _mongo_client["test"]
_db_insight = _mongo_client["jaghut_insights"]

COL_COMPANIES    = _db_test["companies"]
COL_COMMODITIES  = _db_test["commodities"]
COL_ASSESSMENTS  = _db_test["assessments"]
COL_OUT_TEST     = _db_test["company_insight"]
COL_OUT_INSIGHT  = _db_insight["company_insight"]

for _col in [COL_OUT_TEST, COL_OUT_INSIGHT]:
    try:
        _col.create_index("company_id", unique=True, background=True)
        _col.create_index("insight_id", unique=True, background=True)
    except PyMongoError:
        pass

try:
    import chromadb
except ImportError:
    print("❌  chromadb belum terinstall.")
    sys.exit(1)

try:
    _chroma = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _chroma.heartbeat()
    print(f"✅  ChromaDB terhubung di {CHROMA_HOST}:{CHROMA_PORT}")
except Exception as e:
    print(f"❌  Tidak bisa konek ke ChromaDB: {e}")
    sys.exit(1)

try:
    from langchain_ollama.llms import OllamaLLM
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    _embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    _llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.4, repeat_penalty=1.1)
    print(f"✅  LLM (qwen2.5:1.5b) + Embeddings ({EMBED_MODEL}) siap.")

    company_insight_store = Chroma(
        collection_name="company_insights",
        client=_chroma,
        embedding_function=_embeddings,
    )

    main_dataset_store = Chroma(
        collection_name="main_dataset",
        client=_chroma,
        embedding_function=_embeddings,
    )
    print("✅  ChromaDB collections ('company_insights', 'main_dataset') siap.")
except Exception as e:
    _llm = None
    _embeddings = None
    company_insight_store = None
    main_dataset_store = None
    print(f"⚠️  LangChain tidak tersedia ({e}) — insight hanya ke MongoDB.")


def _ask_llm(prompt: str, fallback: str = "") -> str:
    if _llm is None:
        return fallback
    try:
        return _llm.invoke(prompt).strip()
    except Exception as e:
        print(f"   ⚠️  LLM error: {e}")
        return fallback


def _fetch_rag_data(company_name: str) -> list[str]:
    """Cari dokumen RAG dari ChromaDB main_dataset yang relevan dengan company."""
    if main_dataset_store is None:
        return []
    try:
        docs = main_dataset_store.similarity_search(
            company_name,
            k=5,
        )
        return [d.page_content[:800] for d in docs]
    except Exception as e:
        print(f"   ⚠️  RAG fetch error: {e}")
        return []


def _score_label(score: int, max_score: int) -> str:
    if max_score == 0:
        return "N/A"
    ratio = score / max_score
    if ratio >= 0.8:
        return "High"
    if ratio >= 0.5:
        return "Medium"
    if ratio >= 0.25:
        return "Low"
    return "Very Low"


def _build_assessment_text(company: dict, assessments: list[dict], commodities_map: dict) -> str:
    lines = [f"Company: {company.get('name', 'Unknown')}",
             f"Headquarters: {company.get('headquarters', 'N/A')}",
             f"Sector: {company.get('sector', 'N/A')}",
             "",
             "=== Assessment Results ==="]

    for a in assessments:
        commodity = commodities_map.get(str(a.get("commodityId", "")), {}).get("name", "Unknown commodity")
        risk_level = commodities_map.get(str(a.get("commodityId", "")), {}).get("riskLevel", "Unknown")
        scores = a.get("scores", {})
        sub_scores = a.get("subScores", {})
        exposure = a.get("exposureData", {})
        commitments = a.get("commitmentsData", {})
        reporting = a.get("reportingData", {})
        gwr = a.get("greenwashingRiskCategory", "Unknown")
        gwi = a.get("greenwashingRiskIndex", 0)

        lines.append(f"")
        lines.append(f"--- Commodity: {commodity} (Risk: {risk_level}) ---")
        lines.append(f"Greenwashing Risk: {gwr} (Index: {gwi})")
        lines.append(f"")
        lines.append(f"ForestIQ Scores:")
        lines.append(f"  Deforestation Exposure: {scores.get('deforestationExposure', 0)}/5 ({_score_label(scores.get('deforestationExposure', 0), 5)})")
        lines.append(f"  Financial Materiality: {scores.get('financialMateriality', 0)}/5 ({_score_label(scores.get('financialMateriality', 0), 5)})")
        lines.append(f"  Commitments Strength: {scores.get('commitmentsStrength', 0)}/4 ({_score_label(scores.get('commitmentsStrength', 0), 4)})")
        lines.append(f"  Actions Reporting: {scores.get('actionsReporting', 0)}/6 ({_score_label(scores.get('actionsReporting', 0), 6)})")
        lines.append(f"  Performance Reporting: {scores.get('performanceReporting', 0)}/10 ({_score_label(scores.get('performanceReporting', 0), 10)})")
        lines.append(f"")
        lines.append(f"Sub-Scores:")
        lines.append(f"  No-Deforestation Commitment: {sub_scores.get('noDeforestationCommitment', 0)}/2")
        lines.append(f"  Human Rights Commitment: {sub_scores.get('humanRightsCommitment', 0)}/2")
        lines.append(f"  Upstream Reporting: {sub_scores.get('upstreamReporting', 0)}/3")
        lines.append(f"  Downstream Reporting: {sub_scores.get('downstreamReporting', 0)}/3")
        lines.append(f"  Human Rights Performance: {sub_scores.get('humanRightsPerformance', 0)}/3")

        ndp = commitments.get("noDeforestation", {})
        if ndp.get("hasCommitment"):
            lines.append(f"  No-Deforestation Commitment: Yes (type: {ndp.get('commitmentType', 'N/A')}, "
                         f"cut-off: {ndp.get('cutOffDate', 'N/A')})")

        hr = commitments.get("humanRights", {})
        if hr.get("hasFpic") or hr.get("respectsLandRights") or hr.get("respectsLabourRights"):
            lines.append(f"  Human Rights: FPIC={hr.get('hasFpic')}, LandRights={hr.get('respectsLandRights')}, "
                         f"LabourRights={hr.get('respectsLabourRights')}")

        lines.append(f"")
        lines.append(f"Exposure Data:")
        lines.append(f"  Exposed: {exposure.get('isExposed', 'N/A')}")
        lines.append(f"  Significant Player: {exposure.get('significantPlayer', 'N/A')}")
        lines.append(f"  Annual Volume: {exposure.get('totalAnnualVolumeTonnes', 0)} tonnes")
        lines.append(f"  Sourcing Countries: {', '.join(exposure.get('sourcingCountries', [])) or 'N/A'}")
        lines.append(f"  EXIOBASE Sector: {exposure.get('exiobaseSector', 'N/A')}")

        lines.append(f"")
        lines.append(f"Reporting Segments: {', '.join(a.get('reportingSegments', []))}")
        lines.append(f"Assessment Date: {a.get('assessmentDate', 'N/A')}")

    return "\n".join(lines)


import re as _re

def _strip_prefix(text: str, prefixes: list[str]) -> str:
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):]
    return text.strip()

def _cut_at_sentence(text: str, max_chars: int) -> str:
    """Potong pada batas kalimat terakhir sebelum max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Cari .!? terakhir yang diikuti spasi atau akhir string
    for sep in (". ", "!\n", "?\n", ".\n", "!", "?"):
        idx = cut.rfind(sep)
        if idx > max_chars * 0.4:  # minimal 40% dari max
            return cut[:idx + 1].strip()
    # fallback: potong di spasi terakhir
    idx = cut.rfind(" ")
    if idx > max_chars * 0.4:
        return cut[:idx].strip()
    return cut.strip()


def _generate_insight(company: dict, assessments: list[dict], commodities_map: dict) -> str:
    assessment_text = _build_assessment_text(company, assessments, commodities_map)
    company_name = company.get("name", "Unknown")
    extra_docs = _fetch_rag_data(company_name)
    extra_context = "\n\n".join(extra_docs) if extra_docs else "Tidak ada data tambahan."

    prompt = (
        f"Kamu adalah analis ESG senior. Buat insight singkat (3-4 kalimat pendek) "
        f"berdasarkan data ForestIQ berikut.\n\n"
        f"DATA ASSESSMENT:\n{assessment_text[:2500]}\n\n"
        f"DATA TAMBAHAN:\n{extra_context[:1200]}\n\n"
        f"PANDUAN:\n"
        f"1. Sebutkan nama perusahaan, sektor, dan lokasi.\n"
        f"2. Jelaskan komoditas yang dinilai dan tingkat risikonya.\n"
        f"3. Analisis skor ForestIQ: apa kekuatan dan kelemahan.\n"
        f"4. Gunakan Bahasa Indonesia profesional, ringkas.\n"
        f"5. Jangan sebut istilah teknis sistem.\n"
        f"6. Langsung ke isi, tanpa judul 'Insight:' atau 'Rekomendasi:' apapun.\n"
        f"7. Maksimal 4 kalimat pendek (sekitar 400 karakter).\n\n"
    )
    insight = _ask_llm(prompt, fallback=(
        f"{company_name} — {assessment_text[:500]}"
    ))
    insight = _strip_prefix(insight, ["Insight:", "insight:", "INSIGHT:", "Insight :", "insight :"])
    insight = _re.split(r'\n\s*Rekomendasi[:\s]', insight, maxsplit=1)[0]
    return _cut_at_sentence(insight.strip(), 500)


def _generate_recommendation(company: dict, assessments: list[dict], commodities_map: dict) -> str:
    assessment_text = _build_assessment_text(company, assessments, commodities_map)
    company_name = company.get("name", "Unknown")

    prompt = (
        f"Kamu adalah konsultan ESG. Berdasarkan data ForestIQ berikut, "
        f"buat rekomendasi perbaikan singkat (3-4 kalimat pendek) untuk {company_name}.\n\n"
        f"DATA ASSESSMENT:\n{assessment_text[:2500]}\n\n"
        f"PANDUAN:\n"
        f"1. Fokus pada tindakan konkret yang harus diperbaiki.\n"
        f"2. Sebutkan hal-hal yang harus dihindari perusahaan.\n"
        f"3. Berikan saran prioritas berdasarkan skor terendah.\n"
        f"4. Bahasa Indonesia profesional, ringkas, tanpa judul 'Rekomendasi:'.\n"
        f"5. Maksimal 4 kalimat pendek (sekitar 400 karakter).\n\n"
    )
    recommendation = _ask_llm(prompt, fallback=(
        f"Tingkatkan skor pada area dengan nilai terendah berdasarkan data ForestIQ yang tersedia."
    ))
    recommendation = _strip_prefix(recommendation, ["Rekomendasi:", "rekomendasi:", "Rekomendasi :"])
    return _cut_at_sentence(recommendation.strip(), 500)


def _company_insight_id(company_id: str) -> str:
    return hashlib.md5(f"company_insight:{company_id}".encode()).hexdigest()


def _needs_update(existing: dict | None, assessments: list[dict]) -> bool:
    """Cek apakah insight perlu diupdate berdasarkan updatedAt assessment."""
    if existing is None:
        return True
    existing_gen = existing.get("generated_at")
    if not existing_gen:
        return True
    if isinstance(existing_gen, str):
        try:
            existing_gen = datetime.fromisoformat(existing_gen.replace("Z", "+00:00"))
        except Exception:
            return True
    latest_assessment = max(
        (a.get("updatedAt") or a.get("createdAt") or datetime.now(timezone.utc)
         for a in assessments if a.get("updatedAt") or a.get("createdAt")),
        default=datetime.now(timezone.utc),
    )
    if isinstance(latest_assessment, str):
        try:
            latest_assessment = datetime.fromisoformat(latest_assessment.replace("Z", "+00:00"))
        except Exception:
            latest_assessment = datetime.now(timezone.utc)
    return latest_assessment > existing_gen


def process_all_companies() -> int:
    """
    Main processing loop:
    1. Fetch all companies, commodities, assessments
    2. Group assessments by company
    3. For each company, generate/update insight
    4. Store in both test.company_insight and jaghut_insights.company_insight
    Returns count of companies processed.
    """
    print("\n" + "=" * 60)
    print(f"🏢  Company Insight Engine — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    companies = list(COL_COMPANIES.find({"isActive": True}))
    if not companies:
        print("ℹ️  Tidak ada company aktif ditemukan.")
        return 0

    commodities = list(COL_COMMODITIES.find())
    commodities_map = {str(c["_id"]): c for c in commodities}

    all_assessments = list(COL_ASSESSMENTS.find())
    assessments_by_company: dict[str, list[dict]] = {}
    for a in all_assessments:
        cid = str(a.get("companyId", ""))
        assessments_by_company.setdefault(cid, []).append(a)

    print(f"📊  Companies: {len(companies)}, Assessments: {len(all_assessments)}, Commodities: {len(commodities)}")

    processed = 0
    for company in companies:
        cid = str(company["_id"])
        cname = company.get("name", "Unknown")
        c_assessments = assessments_by_company.get(cid, [])

        if not c_assessments:
            print(f"   ⏭️  {cname}: no assessments — skipping")
            continue

        insight_id = _company_insight_id(cid)
        existing = COL_OUT_TEST.find_one({"company_id": cid})

        if not _needs_update(existing, c_assessments):
            print(f"   ✅  {cname}: insight is current — no update needed")
            processed += 1
            continue

        print(f"   🔄  {cname}: generating insight ({len(c_assessments)} assessment(s))...")
        try:
            insight_text = _generate_insight(company, c_assessments, commodities_map)
            recommendation_text = _generate_recommendation(company, c_assessments, commodities_map)
        except Exception as e:
            print(f"   ❌  {cname}: Gagal generate insight: {e}")
            traceback.print_exc()
            continue

        now = datetime.now(timezone.utc)

        # Aggregate scores across all assessments
        score_summary = {}
        for key in ["deforestationExposure", "financialMateriality", "performanceReporting",
                     "commitmentsStrength", "actionsReporting"]:
            vals = [a.get("scores", {}).get(key, 0) for a in c_assessments if a.get("scores")]
            if vals:
                score_summary[key] = {
                    "avg": round(sum(vals) / len(vals), 1),
                    "max": max(vals),
                    "min": min(vals),
                }

        commodities_assessed = []
        for a in c_assessments:
            com = commodities_map.get(str(a.get("commodityId", "")), {})
            if com.get("name") and com["name"] not in commodities_assessed:
                commodities_assessed.append(com["name"])

        # Find assessments with greenwashing risk
        high_gwr = [a for a in c_assessments
                     if a.get("greenwashingRiskCategory", "").lower() in ("high", "very high")]

        doc = {
            "insight_id":          insight_id,
            "company_id":          cid,
            "company_name":        cname,
            "company_headquarters": company.get("headquarters", ""),
            "company_sector":      company.get("sector", ""),
            "commodities_assessed": commodities_assessed,
            "commodity_count":     len(commodities_assessed),
            "assessment_count":    len(c_assessments),
            "latest_assessment":   max((a.get("assessmentDate", "") for a in c_assessments if a.get("assessmentDate")), default=""),
            "scores_summary":      score_summary,
            "greenwashing_flagged": len(high_gwr) > 0,
            "greenwashing_high_risk_commodities": [
                f"{commodities_map.get(str(a.get('commodityId', '')), {}).get('name', 'Unknown')}"
                for a in high_gwr
            ],
            "insight":             insight_text,
            "Recomendation":       recommendation_text,
            "generated_at":        now,
            "generator_ver":       "company-insight-engine-v1",
            "model_used":          "qwen2.5:1.5b" if _llm else None,
            "rag_docs_found":      True,
        }

        # Upsert to both MongoDB collections
        for col in [COL_OUT_TEST, COL_OUT_INSIGHT]:
            try:
                col.update_one(
                    {"company_id": cid},
                    {"$set": doc},
                    upsert=True,
                )
            except PyMongoError as e:
                print(f"   ❌  Gagal upsert ke {col.name}: {e}")

        # Save to ChromaDB for RAG retrieval
        if company_insight_store is not None:
            try:
                # Remove old docs for this company
                old = company_insight_store.get(where={"company_id": cid})
                if old["ids"]:
                    company_insight_store.delete(ids=old["ids"])

                score_lines = "; ".join(
                    f"{k}: avg={v['avg']}/{v.get('max', 0)}"
                    for k, v in score_summary.items()
                ) if score_summary else "no scores"

                rag_text = (
                    f"=== ForestIQ Company Insight: {cname} ===\n"
                    f"Sector: {company.get('sector', 'N/A')} | "
                    f"Headquarters: {company.get('headquarters', 'N/A')}\n"
                    f"Commodities: {', '.join(commodities_assessed)}\n"
                    f"Greenwashing Risk: {'FLAGGED' if doc['greenwashing_flagged'] else 'Normal'}\n\n"
                    f"Scores: {score_lines}\n\n"
                    f"Insight:\n{insight_text}\n\n"
                    f"Recomendation:\n{recommendation_text}"
                )

                rag_doc = Document(
                    page_content=rag_text,
                    metadata={
                        "source":          "company_insight_engine",
                        "company_id":      cid,
                        "company_name":    cname,
                        "commodities":     ",".join(commodities_assessed),
                        "generated_at":    now.isoformat(),
                        "greenwashing":    doc["greenwashing_flagged"],
                    },
                    id=insight_id,
                )
                company_insight_store.add_documents(documents=[rag_doc], ids=[insight_id])
                print(f"   💾  Saved to ChromaDB 'company_insights' collection.")
            except Exception as e:
                print(f"   ⚠️  ChromaDB save error: {e}")

        print(f"   ✅  {cname}: insight saved (ID: {insight_id[:12]}...)")
        processed += 1

    print(f"\n✅  Selesai — {processed} company insight(s) diproses.\n")
    return processed


def run_once():
    """Bersihkan jaghut_insights.company_insight + ChromaDB lalu generate semua insight baru."""
    try:
        deleted = COL_OUT_INSIGHT.delete_many({})
        print(f"🗑️  jaghut_insights.company_insight dikosongkan ({deleted.deleted_count} dokumen dihapus).")
    except PyMongoError as e:
        print(f"⚠️  Gagal reset jaghut_insights.company_insight: {e}")

    # Also clear ChromaDB collection for a clean rebuild
    if company_insight_store is not None:
        try:
            all_old = company_insight_store.get(limit=100_000)
            if all_old["ids"]:
                company_insight_store.delete(ids=all_old["ids"])
                print(f"🗑️  ChromaDB 'company_insights' dikosongkan ({len(all_old['ids'])} dokumen dihapus).")
        except Exception as e:
            print(f"⚠️  Gagal reset ChromaDB company_insights: {e}")

    process_all_companies()


def run_loop(poll_seconds: int = POLL_SECS):
    """Jalankan process_all_companies() saat startup, lalu polling tiap poll_seconds.
    
    Tidak mereset data yang sudah ada — hanya membuat insight baru untuk
    perusahaan yang belum memiliki insight, dan memperbarui insight yang
    assessment-nya berubah (deteksi via updatedAt).
    """
    # First run: incremental — hanya create missing + update changed
    print("🏗️  First pass: membuat insight baru & memperbarui yang berubah...")
    process_all_companies()

    print(f"⏰  Monitoring aktif — cek perubahan setiap {poll_seconds}s. Ctrl+C untuk berhenti.\n")
    while True:
        time.sleep(poll_seconds)
        try:
            count = process_all_companies()
            if count == 0:
                print("   ℹ️  Tidak ada perubahan — menunggu siklus berikutnya...")
        except Exception:
            print("❌  Error saat monitoring:")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jaghut ForestIQ Company Insight Engine")
    parser.add_argument("--once", action="store_true", help="Reset & generate semua insight baru")
    parser.add_argument("--now", action="store_true", help="Jalankan sekali (incremental) tanpa loop")
    parser.add_argument("--poll", type=int, default=POLL_SECS, help=f"Interval polling detik (default: {POLL_SECS})")
    args = parser.parse_args()

    if args.once:
        run_once()
    elif args.now:
        process_all_companies()
    else:
        run_loop(poll_seconds=args.poll)
