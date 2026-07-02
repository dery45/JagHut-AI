"""
esg_api.py — ESG Entity API + Perenual Plant API (legacy support)

Dual function:
1. ESG Entity detection & lookup (companies, regulations, frameworks, commodities)
   — Data sourced from ChromaDB RAG (Forest IQ, company data, regulations)
2. Perenual Plant API (legacy) — biodiversity, plant species, conservation data
   — Kept for ESG biodiversity/ecosystem questions

Perubahan dari plant_api.py:
[NEW] ESG Entity search — mencari data perusahaan, skor ESG, Forest IQ metrics
[NEW] Entity cache di ChromaDB collection "entity_data"
[MAINTAINED] Perenual API tetap berfungsi untuk data tanaman/keanekaragaman hayati
"""

import os
import hashlib
import time
import random
import threading
import requests
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

PERENUAL_KEY     = os.getenv("PERENUAL_API_KEY", "")
PERENUAL_BASE    = "https://perenual.com/api/v2"
PERENUAL_BASE_V1 = "https://perenual.com/api"

# ─── Plant cache TTL ──────────────────────────────────────────────────────────
PLANT_CACHE_TTL_DAYS = 30

# ─── Entity cache TTL ─────────────────────────────────────────────────────────
ENTITY_CACHE_TTL_DAYS = 14

EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
# ─── ChromaDB server connection ──────────────────────────────────────────────
import chromadb as _chromadb
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

if not PERENUAL_KEY or PERENUAL_KEY == "sk-your-api-key-here":
    print("⚠️  PERENUAL_API_KEY not set in .env — plant API features disabled.")

_embeddings = OllamaEmbeddings(model=EMBED_MODEL)
plant_store = Chroma(
    collection_name="plant_data",
    client=_chroma_client,
    embedding_function=_embeddings,
)

entity_store = Chroma(
    collection_name="entity_data",
    client=_chroma_client,
    embedding_function=_embeddings,
)

# ─── API Key Validation on Startup ────────────────────────────────────────────
if PERENUAL_KEY and PERENUAL_KEY != "sk-your-api-key-here":
    try:
        print("🔍  Validating Perenual API key...")
        _test_resp = requests.get(
            f"{PERENUAL_BASE}/species-list",
            params={"key": PERENUAL_KEY},
            timeout=5
        )
        if _test_resp.status_code in (401, 403):
            print("   ❌ Invalid Perenual API Key! Plant features disabled.")
            PERENUAL_KEY = ""
        elif _test_resp.status_code == 429:
            print("   ⚠️  Perenual API 429 Rate Limit hit on startup — pausing 1 hour.")
            _api_rate_limit_until = time.time() + 3600
        else:
            print("   ✅ Perenual API key valid.")
    except Exception as _e:
        print(f"   ⚠️  API validation error: {_e}")


# ─── Token bucket rate limiter ────────────────────────────────────────────────
class _ApiQueue:
    """
    Token bucket yang membatasi laju request ke Perenual API.
    Default: max 1 request per detik (sesuai free tier ~60 req/menit).
    Thread-safe menggunakan Lock.
    """
    def __init__(self, min_interval: float = 1.1):
        self._min_interval = min_interval   # detik antar request
        self._last_call    = 0.0
        self._lock         = threading.Lock()

    def wait(self):
        """Blokir sampai boleh mengirim request berikutnya."""
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                sleep_for = self._min_interval - elapsed
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


_queue = _ApiQueue(min_interval=1.1)

# ─── Session-level rate limit flag ──────────────────────────────────────────
# Di-set ke waktu +1 jam saat _get() menyerah setelah semua retry habis karena 429.
# Fungsi API akan skip request jika waktu sekarang < flag ini.
_api_rate_limit_until = 0.0


# ─── HTTP helper dengan flat retry ───────────────────────────────────────────
_MAX_RETRIES  = 0      # max 0 retry (1 attempt total) — gagal cepat
_BACKOFF_FLAT = 1.5    # detik flat per retry — tidak eksponensial
_JITTER_MAX   = 0.3    # jitter kecil untuk menghindari thundering herd


def _get(url: str, params: dict) -> Optional[dict]:
    global _api_rate_limit_until
    if time.time() < _api_rate_limit_until:
        return None
    attempt = 0

    while attempt <= _MAX_RETRIES:
        _queue.wait()   # rate limit: tunggu giliran

        try:
            resp = requests.get(url, params=params, timeout=8)

            # 429 → flat retry
            if resp.status_code == 429:
                attempt += 1
                if attempt > _MAX_RETRIES:
                    _api_rate_limit_until = time.time() + 3600
                    print(f"   ❌ 429 after {_MAX_RETRIES} retries — Perenual API dimatikan selama 1 jam.")
                    print(f"   ℹ️  Jawaban akan sementara menggunakan data RAG lokal.")
                    return None

                # Cek header Retry-After dari server, tapi cap di 3 detik
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = min(float(retry_after), 3.0) if retry_after else _BACKOFF_FLAT
                except ValueError:
                    wait = _BACKOFF_FLAT

                jitter = random.uniform(0, _JITTER_MAX)
                total  = wait + jitter
                print(f"   ⏳ 429 — retry {attempt}/{_MAX_RETRIES} in {total:.1f}s...")
                time.sleep(total)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            attempt += 1
            wait = _BACKOFF_FLAT + random.uniform(0, _JITTER_MAX)
            print(f"   ⏳ Timeout — retry {attempt}/{_MAX_RETRIES} in {wait:.1f}s...")
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            # Error non-429 (500, network, dll.) — tidak di-retry
            print(f"   ⚠️  API error: {e}")
            return None

    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _doc_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def _entity_cache_expired(cached_at: str, ttl_days: int) -> bool:
    if not cached_at or ttl_days <= 0:
        return False
    from datetime import datetime as _dt
    try:
        age_days = (_dt.now() - _dt.fromisoformat(cached_at)).days
        return age_days > ttl_days
    except Exception:
        return False


def _already_cached(cache_key: str) -> bool:
    """
    Cek apakah cache_key sudah ada di ChromaDB dan belum expired (TTL).
    Kalau TTL habis → return False agar data di-fetch ulang dari Perenual.
    """
    result = plant_store.get(where={"cache_key": cache_key}, limit=1, include=["metadatas"])
    if not result["ids"]:
        return False
    if PLANT_CACHE_TTL_DAYS <= 0:
        return True   # TTL dimatikan — cache selalu valid
    # Cek usia cache
    meta      = result["metadatas"][0] if result["metadatas"] else {}
    cached_at = meta.get("cached_at", "")
    if not cached_at:
        return True   # entri lama tanpa timestamp — anggap masih valid
    try:
        from datetime import datetime as _dt
        age_days = (_dt.now() - _dt.fromisoformat(cached_at)).days
        if age_days > PLANT_CACHE_TTL_DAYS:
            print(f"   ♻️  Plant cache expired ({age_days} days old) — will re-fetch.")
            # Hapus entri lama agar tidak duplikat saat upsert
            plant_store.delete(ids=result["ids"])
            return False
    except Exception:
        pass   # parsing error — anggap masih valid
    return True


def _store_docs(documents: list[Document]):
    """Simpan dokumen ke ChromaDB dengan timestamp cached_at untuk TTL tracking."""
    if not documents:
        return
    from datetime import datetime as _dt
    now_iso = _dt.now().isoformat()
    # Inject cached_at dan truncating content agar tidak melebih context embedding (512 tokens ≈ 2000 chars)
    for doc in documents:
        doc.metadata["cached_at"] = now_iso
        if len(doc.page_content) > 2000:
            doc.page_content = doc.page_content[:2000] + "..."
    
    ids      = [doc.id for doc in documents]
    try:
        existing = plant_store.get(ids=ids)["ids"]
        new_docs = [d for d in documents if d.id not in existing]
        new_ids  = [d.id for d in new_docs]
        if new_docs:
            plant_store.add_documents(documents=new_docs, ids=new_ids)
            print(f"   💾 Stored {len(new_docs)} new plant docs to ChromaDB (cached_at: {now_iso}).")
    except Exception as e:
        print(f"   ⚠️  Plant store add error: {e}")


def _store_entity_docs(documents: list[Document]):
    """Simpan dokumen ESG entity ke entity_store ChromaDB."""
    if not documents:
        return
    from datetime import datetime as _dt
    now_iso = _dt.now().isoformat()
    for doc in documents:
        doc.metadata["cached_at"] = now_iso
        if len(doc.page_content) > 2000:
            doc.page_content = doc.page_content[:2000] + "..."
    
    ids = [doc.id for doc in documents]
    try:
        existing = entity_store.get(ids=ids)["ids"]
        new_docs = [d for d in documents if d.id not in existing]
        new_ids  = [d.id for d in new_docs]
        if new_docs:
            entity_store.add_documents(documents=new_docs, ids=new_ids)
            print(f"   💾 Stored {len(new_docs)} new entity docs to ChromaDB (cached_at: {now_iso}).")
    except Exception as e:
        print(f"   ⚠️  Entity store add error: {e}")


# ─── API 1: Species search + details ─────────────────────────────────────────
def _fetch_species_list(plant_name: str, page: int = 1) -> list[dict]:
    data = _get(f"{PERENUAL_BASE}/species-list", {
        "q": plant_name, "page": page, "key": PERENUAL_KEY
    })
    return data.get("data", []) if data else []


def _fetch_species_detail(plant_id: int) -> Optional[dict]:
    return _get(f"{PERENUAL_BASE}/species/details/{plant_id}", {"key": PERENUAL_KEY})


def _species_to_text(detail: dict) -> str:
    lines = [
        f"Plant: {detail.get('common_name', 'Unknown')}",
        f"Scientific name: {', '.join(detail.get('scientific_name', []))}",
        f"Family: {detail.get('family', '')}",
        f"Type: {detail.get('type', '')}",
        f"Cycle: {detail.get('cycle', '')}",
        f"Watering: {detail.get('watering', '')}",
        f"Sunlight: {', '.join(detail.get('sunlight', []))}",
        f"Care level: {detail.get('care_level', '')}",
        f"Growth rate: {detail.get('growth_rate', '')}",
        f"Propagation: {', '.join(detail.get('propagation', []))}",
        f"Soil: {', '.join(detail.get('soil', []))}",
        f"Origin: {', '.join(detail.get('origin', []))}",
        f"Indoor: {detail.get('indoor', '')}",
        f"Tropical: {detail.get('tropical', '')}",
        f"Drought tolerant: {detail.get('drought_tolerant', '')}",
        f"Edible fruit: {detail.get('edible_fruit', '')}",
        f"Harvest season: {detail.get('harvest_season', '')}",
        f"Medicinal: {detail.get('medicinal', '')}",
        f"Pest susceptibility: {', '.join(detail.get('pest_susceptibility', []))}",
    ]
    desc = detail.get("description", "")
    if desc:
        lines.append(f"Description: {desc}")
    return "\n".join(l for l in lines if not l.endswith(": "))


def fetch_plant_species(plant_name: str) -> list[Document]:
    cache_key = f"species:{plant_name.lower().strip()}"
    if _already_cached(cache_key):
        print(f"   ✅ Plant '{plant_name}' found in local cache.")
        results = plant_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    print(f"   🌐 Fetching species data for '{plant_name}' from Perenual...")
    species_list = _fetch_species_list(plant_name)
    if not species_list:
        print(f"   ℹ️  No species results found for '{plant_name}'.")
        return []

    documents = []
    for item in species_list[:5]:
        plant_id = item.get("id")
        if not plant_id:
            continue
        detail = _fetch_species_detail(plant_id)
        if not detail:
            continue
        text   = _species_to_text(detail)
        doc_id = _doc_id(f"species:{plant_id}")
        img_url = ""
        if detail.get("default_image"):
            img_url = detail["default_image"].get("regular_url", "")
        documents.append(Document(
            page_content=text,
            metadata={
                "source":      "perenual_species",
                "cache_key":   cache_key,
                "plant_id":    str(plant_id),
                "common_name": detail.get("common_name", ""),
                "image_url":   img_url,
            },
            id=doc_id,
        ))

    _store_docs(documents)
    return documents


# ─── API 2: Pest & Disease ────────────────────────────────────────────────────
def _disease_to_text(item: dict) -> str:
    lines = [
        f"Pest/Disease: {item.get('common_name', 'Unknown')}",
        f"Scientific name: {item.get('scientific_name', '')}",
        f"Host plants: {', '.join(item.get('host', []))}",
    ]
    for section in item.get("description", []):
        subtitle = section.get("subtitle", "")
        desc     = section.get("description", "")
        if subtitle and desc:
            lines.append(f"{subtitle}: {desc}")
    for section in item.get("solution", []):
        subtitle = section.get("subtitle", "")
        desc     = section.get("description", "")
        if subtitle and desc:
            lines.append(f"Solution - {subtitle}: {desc}")
    return "\n".join(l for l in lines if not l.endswith(": "))


def fetch_pest_disease(query: str) -> list[Document]:
    cache_key = f"disease:{query.lower().strip()}"
    if _already_cached(cache_key):
        print(f"   ✅ Disease '{query}' found in local cache.")
        results = plant_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    print(f"   🌐 Fetching pest/disease data for '{query}' from Perenual...")
    data = _get(f"{PERENUAL_BASE_V1}/pest-disease-list", {
        "q": query, "page": 1, "key": PERENUAL_KEY
    })
    if not data:
        return []

    items     = data.get("data", [])
    documents = []
    for item in items[:5]:
        text   = _disease_to_text(item)
        doc_id = _doc_id(f"disease:{item.get('id', query)}")
        documents.append(Document(
            page_content=text,
            metadata={
                "source":      "perenual_disease",
                "cache_key":   cache_key,
                "disease_id":  str(item.get("id", "")),
                "common_name": item.get("common_name", ""),
            },
            id=doc_id,
        ))

    _store_docs(documents)
    return documents


# ─── API 3: Care guides ───────────────────────────────────────────────────────
def _care_guide_to_text(species_id: int, guides: list[dict]) -> str:
    lines = [f"Care guides for species ID {species_id}:"]
    for guide in guides:
        for s in guide.get("section", []):
            subtitle = s.get("type", "")
            desc     = s.get("description", "")
            if subtitle and desc:
                lines.append(f"{subtitle}: {desc}")
    return "\n".join(lines)


def fetch_care_guides(species_id: int) -> list[Document]:
    cache_key = f"care:{species_id}"
    if _already_cached(cache_key):
        print(f"   ✅ Care guides for species {species_id} found in local cache.")
        results = plant_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    print(f"   🌐 Fetching care guides for species {species_id} from Perenual...")
    data = _get(f"{PERENUAL_BASE_V1}/species-care-guide-list", {
        "species_id": species_id, "page": 1, "key": PERENUAL_KEY
    })
    if not data:
        return []

    items = data.get("data", [])
    if not items:
        return []

    text   = _care_guide_to_text(species_id, items)
    doc_id = _doc_id(f"care:{species_id}")
    doc    = Document(
        page_content=text,
        metadata={
            "source":     "perenual_care_guide",
            "cache_key":  cache_key,
            "species_id": str(species_id),
        },
        id=doc_id,
    )
    _store_docs([doc])
    return [doc]


# ─── ESG Entity lookup ────────────────────────────────────────────────────────
def is_entity_cached(entity_name: str) -> bool:
    """Check if ESG entity data is in entity_store ChromaDB collection."""
    name = entity_name.lower().strip()
    cache_key = f"entity:{name}"
    result = entity_store.get(where={"cache_key": cache_key}, limit=1, include=["metadatas"])
    if not result["ids"]:
        return False
    if ENTITY_CACHE_TTL_DAYS <= 0:
        return True
    meta = result["metadatas"][0] if result["metadatas"] else {}
    cached_at = meta.get("cached_at", "")
    if not cached_at:
        return True
    if _entity_cache_expired(cached_at, ENTITY_CACHE_TTL_DAYS):
        entity_store.delete(ids=result["ids"])
        return False
    return True


def get_cached_entity_docs(entity_name: str, k: int = 6) -> list[Document]:
    """Retrieve cached ESG entity documents from entity_store."""
    try:
        return entity_store.similarity_search(entity_name, k=k)
    except Exception:
        return []


def search_entity_info(entity_name: str) -> list[Document]:
    """
    Main ESG entity search function.
    1. Checks entity_store for cached data
    2. Searches main_dataset ChromaDB for ESG documents about that entity
    3. Also tries Perenual if the entity looks like a plant/biodiversity term
    """
    all_docs: list[Document] = []
    name = entity_name.lower().strip()
    cache_key = f"entity:{name}"

    # Step 1: Check entity_store cache
    if is_entity_cached(entity_name):
        print(f"   ✅ Entity '{entity_name}' found in entity cache.")
        results = entity_store.get(
            where={"cache_key": cache_key}, include=["documents", "metadatas"]
        )
        cached_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
        all_docs.extend(cached_docs)
        return all_docs

    print(f"   🔍 Searching ESG data for entity '{entity_name}' in main_dataset...")

    # Step 2: Search main_dataset ChromaDB for ESG documents
    try:
        from langchain_chroma import Chroma as _Chroma
        _main_dataset = _Chroma(
            collection_name="main_dataset",
            client=_chroma_client,
            embedding_function=_embeddings,
        )
        esg_results = _main_dataset.similarity_search(entity_name, k=10)
        if esg_results:
            print(f"   ✅ Found {len(esg_results)} ESG documents for '{entity_name}' in main_dataset.")
            entity_docs = []
            for doc in esg_results:
                doc_id = _doc_id(f"entity:{name}:{doc.id}")
                entity_docs.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        "source":      "main_dataset",
                        "cache_key":   cache_key,
                        "entity_name": entity_name,
                        **doc.metadata,
                    },
                    id=doc_id,
                ))
            all_docs.extend(entity_docs)
            _store_entity_docs(entity_docs)
    except Exception as e:
        print(f"   ⚠️  main_dataset search error: {e}")

    # Step 3: Try Perenual for biodiversity/plant terms
    biodiversity_keywords = [
        "tree", "plant", "species", "forest", "flora", "fauna", "biodiversity",
        "mangrove", "rainforest", "conservation", "habitat", "ecosystem",
    ]
    if any(kw in name for kw in biodiversity_keywords) and PERENUAL_KEY:
        print(f"   🌿 Detected biodiversity term — also querying Perenual API for '{entity_name}'.")
        plant_docs = search_plant_info(entity_name)
        all_docs.extend(plant_docs)

    return all_docs


# ─── Public API ───────────────────────────────────────────────────────────────
def is_plant_cached(plant_name: str) -> bool:
    name = plant_name.lower().strip()
    return (
        _already_cached(f"species:{name}") and
        _already_cached(f"disease:{name}")
    )


def get_cached_plant_docs(plant_name: str, k: int = 6) -> list[Document]:
    try:
        return plant_store.similarity_search(plant_name, k=k)
    except Exception:
        return []


def search_plant_info(plant_name: str) -> list[Document]:
    if time.time() < _api_rate_limit_until:
        print("   ⏭️  Perenual API dinonaktifkan (429 sebelumnya) — pakai RAG lokal.")
        return []
    if not PERENUAL_KEY or PERENUAL_KEY == "sk-your-api-key-here":
        print("   ⚠️  PERENUAL_API_KEY not configured — skipping API fetch.")
        return []

    all_docs: list[Document] = []

    species_docs = fetch_plant_species(plant_name)
    all_docs.extend(species_docs)

    disease_docs = fetch_pest_disease(plant_name)
    all_docs.extend(disease_docs)

    if species_docs:
        species_id = species_docs[0].metadata.get("plant_id")
        if species_id:
            care_docs = fetch_care_guides(int(species_id))
            all_docs.extend(care_docs)

    return all_docs


def search_plant_rag(query: str, k: int = 5) -> list[Document]:
    try:
        return plant_store.similarity_search(query, k=k)
    except Exception:
        return []
