import hashlib
import os
import pickle
import time
import warnings
import pandas as pd

# Suppress pydantic V1 deprecation warnings on Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── Configuration ──────────────────────────────────────────────────────────
# ─── ChromaDB server connection ──────────────────────────────────────────────
# Ganti host/port sesuai setup kamu.
# Default: server jalan di mesin yang sama (localhost:8000)
# Remote server: ganti "localhost" dengan IP/hostname server
import chromadb as _chromadb
from pathlib import Path
from dotenv import load_dotenv
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

DATASET_DIR     = str(_ROOT / "data" / "raw_dataset")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "mxbai-embed-large")
BM25_CACHE_PATH = os.getenv("BM25_CACHE_PATH", str(_ROOT / "data" / "db" / "bm25_cache.pkl"))

# ─── ChromaDB server connection ──────────────────────────────────────────────
# Ganti host/port sesuai setup kamu.
import chromadb as _chromadb
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# ─── Per-type chunk strategies ───────────────────────────────────────────────
# Setiap strategi disesuaikan dengan karakteristik data:
#   - Data tabular/harga: chunk kecil agar satu baris = satu chunk (presisi tinggi)
#   - Data narasi/kebijakan: chunk besar agar konteks tidak terpotong
#   - Data ESG: medium chunk untuk metrik dan skor perusahaan
CHUNK_STRATEGIES = {
    "tabular":   {"chunk_size": 300,  "chunk_overlap": 30},
    "price":     {"chunk_size": 200,  "chunk_overlap": 20},
    "policy":    {"chunk_size": 600,  "chunk_overlap": 80},
    "esg":       {"chunk_size": 500,  "chunk_overlap": 50},
    "xlsx_default": {"chunk_size": 400, "chunk_overlap": 50},
    "csv_default":  {"chunk_size": 300, "chunk_overlap": 30},
}

# Keyword deteksi tipe data berdasarkan nama kolom (lowercase)
_ESG_COLS      = {"esg", "environmental", "social", "governance", "forest iq",
                  "score", "rating", "metric", "emission", "carbon",
                  "deforestation", "supply chain", "compliance", "risk",
                  "sustainability", "disclosure"}
_PRICE_COLS    = {"harga", "price", "nilai", "cost", "tarif", "rate", "rupiah", "rp", "usd"}
_POLICY_COLS   = {"kebijakan", "regulasi", "peraturan", "policy", "aturan", "ketentuan",
                  "pasal", "ayat", "keputusan", "permentan", "sk", "instruksi"}
_TABULAR_COLS  = {"tanggal", "date", "provinsi", "kabupaten", "kecamatan", "kode",
                  "id", "no", "nomor", "komoditas", "varietas", "musim", "panen"}


def _detect_sheet_type(df: pd.DataFrame) -> str:
    """
    Deteksi tipe data berdasarkan nama kolom.
    Return salah satu dari: esg, price, policy, tabular, xlsx_default.
    """
    cols = {c.lower().strip() for c in df.columns}

    # Cek overlap dengan keyword sets — ESG didahulukan
    if cols & _ESG_COLS:
        return "esg"
    if cols & _PRICE_COLS:
        return "price"
    if cols & _POLICY_COLS:
        return "policy"
    if cols & _TABULAR_COLS:
        return "tabular"
    return "xlsx_default"


def _get_splitter(data_type: str) -> RecursiveCharacterTextSplitter:
    """Buat splitter sesuai tipe data."""
    cfg = CHUNK_STRATEGIES.get(data_type, CHUNK_STRATEGIES["csv_default"])
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        add_start_index=True,
    )


# ─── Initialization ─────────────────────────────────────────────────────────
embeddings   = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma(
    collection_name="main_dataset",
    client=_chroma_client,
    embedding_function=embeddings
)


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def read_csv_safe(file_path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(file_path, dtype=str, encoding=enc).dropna(how="all")
        except (UnicodeDecodeError, Exception):
            continue
    return None


def read_xlsx_safe(file_path):
    try:
        xl     = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name, dtype=str).dropna(how="all")
                if not df.empty:
                    sheets[sheet_name] = df
            except Exception as e:
                print(f"  ⚠️  Skipping sheet '{sheet_name}': {e}")
        return sheets if sheets else None
    except Exception as e:
        print(f"  ❌ Failed to open XLSX: {e}")
        return None


def process_dataframe(df, file_name, sheet_name=None, forced_type=None):
    """
    Proses satu DataFrame menjadi list (Document, doc_id).

    Args:
        forced_type: override deteksi otomatis (untuk CSV yang tipenya sudah jelas)
    """
    results      = []
    source_label = f"{file_name} (sheet: {sheet_name})" if sheet_name else file_name

    # Deteksi tipe data
    if forced_type:
        data_type = forced_type
    elif sheet_name:
        data_type = _detect_sheet_type(df)
    else:
        # CSV: deteksi dari kolom, fallback ke csv_default
        detected = _detect_sheet_type(df)
        data_type = detected if detected != "xlsx_default" else "csv_default"

    text_splitter = _get_splitter(data_type)
    cfg           = CHUNK_STRATEGIES.get(data_type, CHUNK_STRATEGIES["csv_default"])

    print(f"   📐 [{data_type}] chunk_size={cfg['chunk_size']} overlap={cfg['chunk_overlap']}")

    for i, row in df.iterrows():
        row_dict = row.dropna().to_dict()
        if not row_dict:
            continue

        content_parts = [f"Source: {source_label}"]
        if sheet_name:
            content_parts.append(f"Sheet: {sheet_name}")
        for k, v in row_dict.items():
            content_parts.append(f"{k}: {v}")
        full_content = "\n".join(content_parts)

        base_metadata = {
            "source":        file_name,
            "row_id":        str(i),
            "data_type":     data_type,         # ← baru: untuk filtering
            **({"sheet": sheet_name} if sheet_name else {}),
        }

        chunks = text_splitter.split_text(full_content)
        for chunk_idx, chunk_text in enumerate(chunks):
            content      = normalize(chunk_text)
            combined_str = f"{content}_{file_name}_{sheet_name or ''}_{i}_{chunk_idx}"
            doc_id       = hashlib.md5(combined_str.encode()).hexdigest()
            metadata     = {**base_metadata, "chunk_index": chunk_idx}
            results.append((
                Document(page_content=content, metadata=metadata, id=doc_id),
                doc_id,
            ))

    return results


def invalidate_bm25_cache():
    if os.path.exists(BM25_CACHE_PATH):
        os.remove(BM25_CACHE_PATH)
        print("🗑️  BM25 cache invalidated — will rebuild on next main.py start.")


def index_file(file_path: str):
    """Index satu file CSV atau XLSX."""
    file_name = os.path.basename(file_path)
    ext       = os.path.splitext(file_name)[1].lower()
    documents = []
    ids       = []
    indexed   = False

    if ext == ".csv":
        existing = vector_store.get(where={"source": file_name}, limit=1)
        if existing["ids"]:
            print(f"⏭️  Already indexed: {file_name}")
            return
        df = read_csv_safe(file_path)
        if df is not None:
            print(f"📄 Indexing CSV: {file_name} ({len(df)} rows)")
            for doc, doc_id in process_dataframe(df, file_name):
                documents.append(doc)
                ids.append(doc_id)

    elif ext == ".xlsx":
        sheets = read_xlsx_safe(file_path)
        if sheets is None:
            return
        for sheet_name, df in sheets.items():
            existing = vector_store.get(
                where={"$and": [{"source": file_name}, {"sheet": sheet_name}]}, limit=1
            )
            if existing["ids"]:
                continue
            print(f"📄 New sheet: {file_name} → {sheet_name} ({len(df)} rows)")
            for doc, doc_id in process_dataframe(df, file_name, sheet_name=sheet_name):
                documents.append(doc)
                ids.append(doc_id)
    else:
        return

    if documents:
        total      = len(documents)
        batch_size = 10
        print(f"📦 Indexing {file_name}: {total} chunks...")
        for i in tqdm(range(0, total, batch_size), desc="Indexing", leave=False):
            vector_store.add_documents(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size],
            )
        print(f"✅ Indexed: {file_name}")
        indexed = True

    if indexed:
        invalidate_bm25_cache()


def index_all_existing():
    import glob
    all_files = (
        glob.glob(os.path.join(DATASET_DIR, "*.csv")) +
        glob.glob(os.path.join(DATASET_DIR, "*.xlsx"))
    )
    if not all_files:
        print(f"⚠️  No files found in {DATASET_DIR}/")
        return
    for file_path in tqdm(all_files, desc="Initial scan", leave=False):
        index_file(file_path)


# ─── Watchdog ────────────────────────────────────────────────────────────────
class DatasetHandler(FileSystemEventHandler):
    def _handle(self, event):
        if event.is_directory:
            return
        ext = os.path.splitext(event.src_path)[1].lower()
        if ext in (".csv", ".xlsx"):
            print(f"\n🔔 File event: {os.path.basename(event.src_path)}")
            time.sleep(0.5)
            index_file(event.src_path)

    def on_created(self, event):  self._handle(event)
    def on_modified(self, event): self._handle(event)


# ─── Retriever ───────────────────────────────────────────────────────────────
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.7},
)

# ─── Main Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"🚀 CSV/XLSX Vector Server started. Watching '{DATASET_DIR}/'...")
    print(f"📐 Chunk strategies active: {list(CHUNK_STRATEGIES.keys())}")

    index_all_existing()

    observer = Observer()
    observer.schedule(DatasetHandler(), path=DATASET_DIR, recursive=False)
    observer.start()
    print("👀 Watchdog active — waiting for new files...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Server stopped.")
    observer.join()