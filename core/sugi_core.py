"""
sugi_core.py — Core logic Sugi v0.1L (platform-agnostic)

Fitur:
  [1] Query Rewriting     — rule-based + Qwen LLM fallback
  [2] Plant & Weather     — deteksi query tanaman & cuaca
  [3] Plant API           — fetch + cache via Perenual API
  [4] Eval Loop           — faithfulness + relevance scoring
  [5] Answer Template     — panduan menjawab lengkap (3 section)
  [6] Debug Commands      — !debug / !flags / !session (CLI & Telegram)

Multi-user: setiap user_id punya session, history, dan memory sendiri.
Config    : semua dari .env di root proyek.
"""

import hashlib
import pickle
import os
import re
import threading
import configparser as _cp
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ─── Load .env dari root proyek ───────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

import chromadb as _chromadb

from services.vectorCSV import vector_store
from core.query_logger import (
    new_query_trace, set_docs, commit_trace,
    print_debug_report, flagged_logs, session_logs,
)

# ─────────────────────────────────────────────
# Config — semua dari .env
# ─────────────────────────────────────────────
EMBED_MODEL         = os.getenv("EMBED_MODEL",         "mxbai-embed-large")
LLM_MODEL           = os.getenv("LLM_MODEL",           "sugi-v0.1L")
UTILITY_MODEL       = os.getenv("UTILITY_MODEL",       "qwen2.5:1.5b")
CHROMA_HOST         = os.getenv("CHROMA_HOST",         "localhost")
CHROMA_PORT         = int(os.getenv("CHROMA_PORT",     "8000"))
BM25_CACHE_PATH     = os.getenv("BM25_CACHE_PATH",     "bm25_cache.pkl")
SCOPE_CONFIG_PATH   = os.getenv("SCOPE_CONFIG_PATH",   "config/settings/scope_config.ini")
REWRITER_CONFIG_PATH= os.getenv("REWRITER_CONFIG_PATH","config/settings/rewriter_config.ini")
PLANT_CONFIG_PATH   = os.getenv("PLANT_CONFIG_PATH",   "config/settings/plant_keywords.ini")
MEMORY_TTL_DAYS     = int(os.getenv("MEMORY_TTL_DAYS", "14"))

_chroma_client = _chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
embeddings     = OllamaEmbeddings(model=EMBED_MODEL)


# ═══════════════════════════════════════════════════════════════════════════════
# [1] QUERY REWRITING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_rewriter_config(path: str) -> tuple[set, set, list, dict]:
    cfg = _cp.ConfigParser(
        comment_prefixes     = (";", "#"),
        inline_comment_prefixes = (";",),
    )
    cfg.read(path, encoding="utf-8")

    def _lines(section, key):
        raw = cfg.get(section, key, fallback="")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    ref_words    = set(_lines("referential_words",    "words"))
    ref_suffixes = set(_lines("referential_suffixes", "suffixes"))
    followup_pats= _lines("followup_patterns",        "patterns")
    topic_kw     = {}
    if cfg.has_section("topic_keywords"):
        for k, v in cfg.items("topic_keywords"):
            topic_kw[k.strip()] = v.strip()
    return ref_words, ref_suffixes, followup_pats, topic_kw


def _word_match(pattern: str, text: str) -> bool:
    """Whole-word regex match with Indonesian suffix support."""
    escaped = re.escape(pattern)
    # Allows common suffixes from scope_config.ini ref-suffixes or similar
    # nya, mu, ku, lah, kah, pun, kan, i, an
    return bool(re.search(
        r"(?<![a-zA-Z])" + escaped + r"(?:nya|mu|ku|lah|kah|pun|kan|i|an)?(?![a-zA-Z])",
        text, re.IGNORECASE
    ))


# Kata-kata yang menandakan model menjawab bukan merewrite
_REWRITE_ANSWER_MARKERS = [
    "saya ", "aku ", "maaf", "tentu", "baik,", "iya,", "tidak,",
    "i am", "i can", "i don't", "sure", "of course",
    "pertama", "kedua", "berikut",
]

_QWEN_REWRITE_TEMPLATE = (
    "Tugas: ubah PERTANYAAN menjadi pertanyaan yang berdiri sendiri menggunakan RIWAYAT.\n\n"
    "Aturan KETAT:\n"
    "- Output HANYA satu pertanyaan singkat, tanpa penjelasan\n"
    "- Jangan jawab pertanyaannya — hanya reformulasi\n"
    "- Maksimal 20 kata\n"
    "- Jika pertanyaan sudah jelas sendiri, kembalikan apa adanya\n\n"
    "RIWAYAT:\n{history}\n\n"
    "PERTANYAAN: {question}\n\n"
    "PERTANYAAN STANDALONE:"
)


# ═══════════════════════════════════════════════════════════════════════════════
# [2] PLANT & WEATHER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _load_plant_config(path: str) -> tuple[set, set, dict]:
    cfg = _cp.ConfigParser(
        comment_prefixes        = (";", "#"),
        inline_comment_prefixes = (";",),
        strict = False,
    )
    try:
        if not cfg.read(path, encoding="utf-8"):
            print(f"⚠️  Plant config not found: {path}")
            return set(), set(), {}
    except _cp.Error as e:
        print(f"⚠️  Plant config parse error: {e}")
        return set(), set(), {}

    def _lines(section, key):
        raw = cfg.get(section, key, fallback="")
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    strong   = set(_lines("keywords_strong", "keywords"))
    weak     = set(_lines("keywords_weak",   "keywords"))
    name_map = {}
    if cfg.has_section("plant_name_map"):
        for k, v in cfg.items("plant_name_map"):
            name_map[k.strip()] = v.strip()
    return strong, weak, name_map


WEATHER_KEYWORDS = {
    "cuaca", "iklim", "hujan", "prakiraan cuaca", "curah hujan",
    "suhu", "suhu udara", "temperatur", "kelembaban", "angin",
    "kecepatan angin", "arah angin", "tekanan udara", "awan",
    "tutupan awan", "evapotranspirasi", "embun", "titik embun",
    "musim", "musim hujan", "musim kemarau", "kekeringan", "banjir",
    "el nino", "la nina", "perubahan iklim", "panas", "dingin",
    "kabut", "badai", "petir", "gerimis",
    "suhu tanah", "kelembaban tanah", "kadar air tanah",
    "prakiraan", "prediksi cuaca", "kondisi cuaca",
    "weather", "temperature", "humidity", "rainfall", "precipitation",
    "wind speed", "wind direction", "forecast", "climate",
    "cloud cover", "evapotranspiration", "dew point", "pressure",
    "soil temperature", "soil moisture", "drought", "flood",
    "storm", "sunny", "cloudy", "rain forecast",
}

_PLANTING_SUITABILITY_PHRASES = {
    "cocok untuk menanam", "cocok menanam", "waktu tanam",
    "saat tanam", "musim tanam", "kapan tanam", "kapan menanam",
    "apakah cocok", "apakah bagus", "waktu yang tepat",
    "suitable for planting", "good time to plant", "right time to plant",
    "cocok ditanam", "baik untuk menanam", "ideal untuk menanam",
}

_CULTIVATION_SIGNALS = {
    "merawat", "cara merawat", "cara menanam", "cara budidaya",
    "teknik", "panduan", "langkah", "tips", "bagaimana cara",
    "pemupukan", "penyiraman", "pemangkasan", "pengendalian hama",
    "penyakit tanaman", "hama tanaman", "perawatan", "budidaya",
    "how to grow", "how to care", "growing guide", "cultivation",
    "treatment", "fertilizer", "pruning", "irrigation method",
}


# ═══════════════════════════════════════════════════════════════════════════════
# [5] ANSWER TEMPLATE (detail, 3 section)
# ═══════════════════════════════════════════════════════════════════════════════

_ANSWER_TEMPLATE_BASE = (
    "Kamu adalah Sugi, asisten pertanian Indonesia yang ramah dan berpengetahuan luas.\n\n"
    "Informasi sesi saat ini:\n"
    "- Tanggal  : [[NOW_DATE]]\n"
    "- Hari     : [[NOW_DAY]]\n"
    "- Jam      : [[NOW_TIME]] WIB\n"
    "- Lokasi   : Jakarta, Indonesia\n\n"
    "Riwayat percakapan (gunakan sebagai konteks untuk pertanyaan lanjutan):\n"
    "{history}\n\n"
    "Data relevan dari database (pertanian, cuaca, tanaman):\n"
    "{data}\n\n"
    "PANDUAN MENJAWAB:\n"
    "1. SUMBER DATA & KONTEKS\n"
    "   - Jika pertanyaan merujuk ke jawaban sebelumnya (contoh: 'kenapa pupuk diatas', "
    "'bagaimana cara merawatnya', 'jelaskan lebih lanjut'), WAJIB gunakan riwayat "
    "percakapan di atas sebagai referensi utama.\n"
    "   - Jika data berisi informasi cuaca, WAJIB gunakan dan sebut angkanya "
    "(suhu, curah hujan, kelembaban, dll.).\n"
    "   - Jika data berisi informasi harga, sebutkan harga spesifik dan tanggalnya.\n"
    "   - Jika data berisi panduan budidaya, kutip langkah yang relevan saja.\n"
    "   - Jika data TIDAK relevan atau kosong, jawab dari pengetahuan umum dan "
    "beritahu user bahwa data spesifik tidak tersedia.\n"
    "   - Gunakan tanggal/hari/jam di atas kalau pertanyaan menyebut "
    "\"hari ini\", \"sekarang\", atau \"today\".\n\n"
    "2. FORMAT JAWABAN\n"
    "   - Langsung jawab tanpa sapaan pembuka (tidak perlu \"Halo!\" atau "
    "\"Tentu saja!\").\n"
    "   - Gunakan bullet points HANYA jika ada 3 item atau lebih yang perlu "
    "disebutkan (langkah, syarat, daftar).\n"
    "   - Untuk 1–2 poin: cukup gunakan kalimat biasa.\n"
    "   - Panjang ideal: 3–6 kalimat untuk pertanyaan sederhana, maksimal 10 "
    "baris untuk pertanyaan kompleks.\n"
    "   - Jangan mengulang kalimat yang sudah disebutkan.\n"
    "   - Akhiri jawaban tepat setelah selesai menjelaskan — tidak perlu penutup "
    "seperti \"Semoga membantu!\"\n"
    "   - Jangan tampilkan proses berpikir, langsung berikan jawaban final.\n\n"
    "3. BAHASA & NADA\n"
    "   - Gunakan Bahasa Indonesia yang jelas dan mudah dipahami petani.\n"
    "   - Jika user menulis dalam Bahasa Inggris, jawab dalam Bahasa Inggris.\n"
    "   - Nada: ramah tapi profesional. Tidak terlalu formal, tidak terlalu santai.\n\n"
    "Pertanyaan saat ini: {question}"
)


# ═══════════════════════════════════════════════════════════════════════════════
# SUGI CORE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class SugiCore:
    """
    Platform-agnostic Sugi engine.
    Dipakai bersama oleh CLI (main.py) dan Telegram (telegram_bot.py).
    Per-user state disimpan di self._sessions dict.
    """

    def __init__(self):
        print(f"⚙️  Config: LLM={LLM_MODEL} | Embed={EMBED_MODEL} | "
              f"Chroma={CHROMA_HOST}:{CHROMA_PORT}")

        self._session_lock = threading.Lock()

        # ── Models ────────────────────────────────────────────────────────────
        self.model = OllamaLLM(
            model          = LLM_MODEL,
            temperature    = 0.3,
            repeat_penalty = 1.15,
            keep_alive     = 300,
        )
        self.rewrite_model = OllamaLLM(
            model       = UTILITY_MODEL,
            temperature = 0,
            keep_alive  = 120,   # tetap loaded 2 menit — hindari reload tiap query
            timeout     = 30,    # max 30 detik untuk rewriting, fallback ke original jika timeout
        )

        # ── Memory store ──────────────────────────────────────────────────────
        self.memory_store = Chroma(
            collection_name    = "conversation_memory",
            client             = _chroma_client,
            embedding_function = embeddings,
        )
        self._purge_expired_memories()

        # ── Weather store (jika ada) ──────────────────────────────────────────
        try:
            from services.vectorWeather import weather_store as _ws
            self.weather_store = _ws
            print("🌤️  Weather store loaded.")
        except ImportError:
            self.weather_store = None
            print("⚠️  vectorWeather not found — weather features disabled.")

        # ── Plant store (jika ada) ────────────────────────────────────────────
        try:
            from core.plant_api import (
                search_plant_info, is_plant_cached,
                get_cached_plant_docs, plant_store,
            )
            self._search_plant_info    = search_plant_info
            self._is_plant_cached      = is_plant_cached
            self._get_cached_plant_docs= get_cached_plant_docs
            self.plant_store           = plant_store
            print("🌿  Plant API loaded.")
        except ImportError:
            self._search_plant_info     = None
            self._is_plant_cached       = None
            self._get_cached_plant_docs = None
            self.plant_store            = None
            print("⚠️  plant_api not found — plant features disabled.")

        # ── Eval loop (jika ada) ──────────────────────────────────────────────
        try:
            from core.eval_loop import evaluate as _evaluate
            self._evaluate = _evaluate
            print("🔬  Eval loop loaded.")
        except ImportError:
            self._evaluate = None
            print("⚠️  eval_loop not found — eval disabled.")

        # ── BM25 + Vector retrievers ──────────────────────────────────────────
        print("🔄  Initializing Retrievers...")
        self.bm25_retriever   = self._load_or_build_bm25()
        self.bm25_retriever.k = 4
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self.primary_ensemble = EnsembleRetriever(
            retrievers = [self.bm25_retriever, self.vector_retriever],
            weights    = [0.5, 0.5],
        )
        self.memory_retriever = self.memory_store.as_retriever(search_kwargs={"k": 2})

        if self.plant_store:
            self.plant_retriever = self.plant_store.as_retriever(search_kwargs={"k": 3})
        if self.weather_store:
            self.weather_retriever = self.weather_store.as_retriever(search_kwargs={"k": 8})

        # ── Reranker ──────────────────────────────────────────────────────────
        print("🧠  Loading Reranker...")
        _reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.compressor = CrossEncoderReranker(model=_reranker, top_n=5)

        # ── Config files ──────────────────────────────────────────────────────
        _scope_path = _ROOT / SCOPE_CONFIG_PATH
        self.greeting_patterns, self.allowed_kw, self.blocked_kw, self.refusal_msg = \
            self._load_scope_config(str(_scope_path))

        _rew_path = _ROOT / REWRITER_CONFIG_PATH
        (self._ref_words, self._ref_suffixes,
         self._followup_pats, self._topic_kw) = _load_rewriter_config(str(_rew_path))
        print(f"📝  Rewriter config: {len(self._ref_words)} ref-words, "
              f"{len(self._followup_pats)} patterns, {len(self._topic_kw)} topics.")

        _plant_path = _ROOT / PLANT_CONFIG_PATH
        (self._plant_strong, self._plant_weak,
         self._plant_name_map) = _load_plant_config(str(_plant_path))
        print(f"🌿  Plant config: {len(self._plant_strong)} strong, "
              f"{len(self._plant_weak)} weak, {len(self._plant_name_map)} name mappings.")

        # ── Qwen rewrite chain ────────────────────────────────────────────────
        self._qwen_rewrite_chain = (
            ChatPromptTemplate.from_template(_QWEN_REWRITE_TEMPLATE)
            | self.rewrite_model
            | StrOutputParser()
        )

        # ── Plant fallback chain ──────────────────────────────────────────────
        self._plant_fallback_chain = (
            ChatPromptTemplate.from_template(
                "You are a botanical assistant. Extract the English plant name.\n"
                "Return ONLY the name (1-3 words), or NONE if no specific plant.\n\n"
                "Question: {question}\n"
                "English plant name (or NONE):"
            )
            | self.rewrite_model
            | StrOutputParser()
        )

        # ── Per-user session state ────────────────────────────────────────────
        # { user_id: { "history": [...], "session_id": "..." } }
        self._sessions: dict[str, dict] = {}

        print("✅  SugiCore ready.\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def ask(self, user_id: str, question: str, platform: str = "cli") -> str:
        """
        Proses pertanyaan dari user dan kembalikan jawaban sebagai string.
        Dipakai oleh CLI (main.py) dan Telegram (telegram_bot.py).
        """
        session           = self._get_or_create_session(user_id)
        trace             = new_query_trace(session["session_id"])
        trace["question"] = question
        is_greeting_q     = self._is_greeting(question)

        full_response = ""
        error_msg     = None

        try:
            # ── [1] Scope check ───────────────────────────────────────────────
            print("🛡️  Checking scope...")
            in_scope = self._is_in_scope(question)
            
            history_text = self._format_history(session["history"])
            has_history  = len(session["history"]) > 0

            # ── [1.1] Query Rewriting ─────────────────────────────────────────
            standalone_query, rewrite_type = self._maybe_rewrite(question, history_text)
            
            if not in_scope and rewrite_type == "suffix":
                # Only trust suffix-based rewrites (e.g. "menanamnya" → "menanam semangka")
                # for scope bypass. Word replacements (e.g. "itu" → "musim tanam")
                # can inject agriculture keywords that weren't in the user's intent.
                print("   [scope] original failed — checking rewritten (suffix-only)...")
                in_scope = self._is_in_scope(standalone_query)
            
            trace["scope_passed"] = in_scope
            trace["rewritten"]    = standalone_query

            if not in_scope:
                print(f"🚫  Out of scope.")
                commit_trace(trace)
                return self.refusal_msg

            print("✅  Approved.")
            print(f"🔍  Query: {standalone_query}")

            # ── [2] Plant & Weather detection ─────────────────────────────────
            include_plant   = self._is_plant_query(question, standalone_query)
            include_weather = (
                self._is_weather_query(standalone_query) or
                self._is_weather_query(question)
            )

            trace["flags"] = {
                "is_plant":    include_plant,
                "is_weather":  include_weather,
                "has_history": has_history,
                "is_greeting": is_greeting_q,
            }

            # ── [3] Plant API ─────────────────────────────────────────────────
            plant_query_en = None
            if include_plant and self._search_plant_info:
                plant_query_en = self._extract_and_translate_plant(standalone_query)
                if plant_query_en is None:
                    print("🌿  No valid plant name — skipping plant API.")
                    include_plant = False
                elif self._is_plant_cached(plant_query_en):
                    cached_docs = self._get_cached_plant_docs(plant_query_en, k=5)
                    print(f"🌿  Cache hit: {len(cached_docs)} docs for '{plant_query_en}'.")
                else:
                    print(f"🌿  Fetching '{plant_query_en}' from Perenual API...")
                    fetched = self._search_plant_info(plant_query_en)
                    if fetched:
                        print(f"   ✅ Fetched {len(fetched)} plant docs.")
                    else:
                        print(f"   ⚠️  No data for '{plant_query_en}'.")
                        include_plant = False

            if include_weather:
                print("🌤️  Weather query detected.")

            # ── Retrieval ─────────────────────────────────────────────────────
            print("🗂️  Retrieving and reranking documents...")

            weather_docs = []
            if include_weather and self.weather_store:
                weather_docs = self.weather_store.similarity_search(standalone_query, k=8)
                print(f"🌤️  Weather docs: {len(weather_docs)}")

            retriever_obj = self._build_retriever(
                has_history     = has_history,
                include_plant   = include_plant,
                include_weather = False,   # weather diambil manual di atas
            )
            rag_docs = retriever_obj.invoke(standalone_query)

            # Dedup by content hash
            seen, all_docs = set(), []
            for d in (weather_docs + rag_docs):
                key = hash(d.page_content[:120])
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)

            # ── [Fix 1] Inject long-term memory docs for returning users ──────
            # Memory docs are prepended (low priority) to give the LLM past
            # conversation context while keeping the retrieval signal dominant.
            if has_history:
                try:
                    mem_docs = self.memory_store.similarity_search(
                        standalone_query, k=2,
                        filter={"user_id": user_id}
                    )
                    new_mem = [
                        d for d in mem_docs
                        if hash(d.page_content[:120]) not in seen
                    ]
                    if new_mem:
                        for d in new_mem:
                            seen.add(hash(d.page_content[:120]))
                        all_docs = new_mem + all_docs   # memory first
                        print(f"💾  Injected {len(new_mem)} memory doc(s) from long-term store.")
                except Exception as _mem_err:
                    print(f"   ⚠️  Memory recall error: {_mem_err}")

            context = self._format_docs(all_docs) if all_docs else "Tidak ada data relevan di database."
            print(f"📊  Found {len(all_docs)} chunks "
                  f"({len(weather_docs)} weather + {len(rag_docs)} RAG).")

            set_docs(trace, all_docs)

            # ── [5] Generate dengan answer template detail ────────────────────
            print("🤖  Generating answer:\n")
            _live_chain = self._get_answer_prompt() | self.model
            for chunk in _live_chain.stream({
                "data":     context,
                "question": standalone_query,
                "history":  history_text,
            }):
                full_response += chunk

            print(full_response)
            trace["answer_preview"] = full_response[:200]
            session["history"].append((question, full_response))
            # ── [Fix 3] Cap history list at 20 turns to prevent RAM bloat ────
            if len(session["history"]) > 20:
                session["history"] = session["history"][-20:]

            # Simpan memory setiap 5 pertanyaan
            if len(session["history"]) % 5 == 0:
                session_copy = {"history": list(session["history"]), "session_id": session["session_id"]}
                threading.Thread(target=self._save_session_memory, args=(user_id, session_copy), daemon=True).start()

            # ── [4] Eval Loop ─────────────────────────────────────────────────
            print("\n🔬  Running eval...")
            if is_greeting_q or not self._evaluate:
                eval_result = {
                    "faithfulness": "SKIP",
                    "relevance":    "SKIP",
                    "flag":         False,
                    "reason":       "greeting or eval disabled",
                    "method":       "skip",
                    "doc_count":    len(all_docs),
                }
                print("   ⏭️  Eval skipped.")
            else:
                eval_result = self._evaluate(
                    question = standalone_query,
                    docs     = all_docs,
                    context  = context,
                    answer   = full_response,
                    use_llm  = True,
                )
            trace["eval"] = eval_result

        except Exception as e:
            error_msg     = str(e)
            full_response = f"⚠️ Maaf, terjadi kesalahan: {e}"
            print(f"\n❌  SugiCore error for {user_id}: {e}")

        finally:
            commit_trace(trace, error=error_msg)

        return full_response

    def get_memory_summary(self, user_id: str) -> str:
        """Ambil ringkasan memory percakapan lama user dari ChromaDB."""
        try:
            results = self.memory_store.similarity_search(
                f"user:{user_id}", k=5, filter={"user_id": user_id}
            )
            if results:
                return "\n\n".join(doc.page_content for doc in results)
        except Exception:
            pass
        try:
            all_mem = self.memory_store.get(where={"user_id": user_id}, limit=5)
            if all_mem["documents"]:
                return "\n\n".join(all_mem["documents"])
        except Exception:
            pass
        return ""

    def clear_session(self, user_id: str):
        """Reset in-memory history untuk user (long-term memory di ChromaDB tetap ada)."""
        with self._session_lock:
            if user_id in self._sessions:
                self._save_session_memory(user_id, self._sessions[user_id])
                self._sessions[user_id]["history"]    = []
                self._sessions[user_id]["session_id"] = (
                    f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
        print(f"🗑️  Session cleared for {user_id}")

    # ── [6] Debug Commands ────────────────────────────────────────────────────

    def handle_debug_command(self, command: str, user_id: str) -> tuple[bool, str]:
        """
        Handle debug commands untuk CLI dan Telegram.
        Return: (handled: bool, output: str)
        """
        cmd      = command.strip().lower()
        session  = self._get_or_create_session(user_id)
        session_id = session["session_id"]

        if cmd == "!debug":
            # Redirect stdout ke string untuk Telegram compatibility
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_debug_report(n=10)
            return True, buf.getvalue()

        if cmd == "!flags":
            entries = flagged_logs()
            if not entries:
                return True, "✅ Tidak ada query yang di-flag."
            lines = [f"🚩 {len(entries)} query di-flag:\n"]
            for e in entries:
                lines.append(f"[{e['query_id']}] {e['question'][:60]}")
                lines.append(f"  → {e['eval'].get('reason', '')}")
                lines.append(f"  → docs: {[d['source'] for d in e.get('docs_retrieved', [])]}\n")
            return True, "\n".join(lines)

        if cmd == "!session":
            entries = session_logs(session_id)
            lines   = [f"📋 Log sesi {session_id}: {len(entries)} query\n"]
            for e in entries:
                flag = "🚩" if e.get("eval", {}).get("flag") else "✅"
                lines.append(
                    f"{flag} [{e['query_id']}] {e['question'][:50]} "
                    f"— {e.get('latency_ms', 0)}ms"
                )
            return True, "\n".join(lines)

        if cmd == "!memory":
            summary = self.get_memory_summary(user_id)
            if summary:
                return True, f"💾 Memory untuk {user_id}:\n\n{summary}"
            return True, "📭 Belum ada memory tersimpan untuk user ini."

        if cmd == "!stats":
            total_sessions = len(self._sessions)
            active_users   = [uid for uid, s in self._sessions.items() if s["history"]]
            lines = [
                f"📊 SugiCore Stats",
                f"Active sessions : {total_sessions}",
                f"Users with history: {len(active_users)}",
                f"LLM model       : {LLM_MODEL}",
                f"Embed model     : {EMBED_MODEL}",
                f"Utility model   : {UTILITY_MODEL}",
                f"Weather store   : {'✅' if self.weather_store else '❌'}",
                f"Plant API       : {'✅' if self._search_plant_info else '❌'}",
                f"Eval loop       : {'✅' if self._evaluate else '❌'}",
            ]
            return True, "\n".join(lines)

        return False, ""

    # ═══════════════════════════════════════════════════════════════════════════
    # [1] QUERY REWRITING — internal methods
    # ═══════════════════════════════════════════════════════════════════════════

    # Pola definitional: "apa itu X", "what is X", dll.
    # "itu" di sini bukan referensial, tapi bagian dari pola definitional.
    _DEFINITIONAL_PATTERNS = [
        re.compile(r"^apa\s+itu\b", re.IGNORECASE),
        re.compile(r"^apakah\s+itu\b", re.IGNORECASE),
        re.compile(r"^apa\s+yang\s+dimaksud\b", re.IGNORECASE),
        re.compile(r"^what\s+is\b", re.IGNORECASE),
        re.compile(r"^what\s+are\b", re.IGNORECASE),
    ]

    def _maybe_rewrite(self, question: str, history_text: str) -> tuple[str, str]:
        """
        Tiga lapis:
        1. Fast-path   — pertanyaan sudah self-contained, skip rewriting
        2. Rule-based  (0ms) — handle 85% kasus referensial
        3. Qwen LLM fallback (~5-30 detik) — hanya untuk kasus ambigu

        Returns:
            (rewritten_query, rewrite_type)
            rewrite_type: "none" | "suffix" | "word" | "followup" | "qwen"
        """
        if history_text == "Belum ada riwayat percakapan.":
            return question, "none"

        # ── Fast-path: definitional questions → skip rewriting ─────────────
        # "apa itu machine learning" → "itu" is NOT referential here
        q_stripped = question.strip()
        for pat in self._DEFINITIONAL_PATTERNS:
            if pat.match(q_stripped):
                print(f"   [rewrite] definitional question — skipping rewrite")
                return question, "none"

        has_ref     = self._has_referential(question)
        is_followup = self._is_implicit_followup(question)

        # ── Fast-path: tidak ada sinyal referensial → langsung return ──────────
        if not has_ref and not is_followup:
            return question, "none"

        # ── Fast-path: pertanyaan sudah self-contained (punya noun spesifik) ──
        # Contoh: "apa saja pupuk anorganik yang cocok?" sudah ada "pupuk anorganik"
        # → tidak perlu rewriting, cukup tambahkan konteks tanaman dari history
        q_lower = question.lower()
        all_known = list(self._plant_name_map.keys()) + list(self._topic_kw.keys())
        question_is_specific = any(_word_match(kw, q_lower) for kw in all_known)
        if question_is_specific and not has_ref:
            # Pertanyaan sudah mengandung subjek spesifik, tidak ambigu
            print(f"   [rewrite] question already specific — skipping rewrite")
            return question, "none"

        subject = self._extract_last_subject(history_text)
        if not subject:
            if has_ref or is_followup:
                # Hanya panggil Qwen jika benar-benar ambigu dan tidak ada subjek
                return self._qwen_rewrite(question, history_text), "qwen"
            return question, "none"

        q = question
        rewrite_type = "none"

        if has_ref:
            replaced = False
            for ref in sorted(self._ref_words, key=len, reverse=True):
                if _word_match(ref, q.lower()):
                    q = re.sub(
                        r"(?<![a-zA-Z])" + re.escape(ref) + r"(?![a-zA-Z])",
                        subject, q, count=1, flags=re.IGNORECASE,
                    )
                    replaced = True
                    rewrite_type = "word"
                    break
            if not replaced:
                old_q = q
                q = self._resolve_suffix_referential(q, subject)
                if q != old_q:
                    rewrite_type = "suffix"

        elif is_followup:
            subject_in_q = any(_word_match(kw, q.lower()) for kw in all_known)
            if not subject_in_q:
                q = q.rstrip("?").rstrip() + f" {subject}?"
                rewrite_type = "followup"

        if q != question:
            print(f"   [rewrite] rule-based: '{question}' → '{q}'")
            return q, rewrite_type

        # Qwen hanya dipanggil jika rule-based tidak bisa resolve DAN ada sinyal ambigu
        if has_ref or is_followup:
            return self._qwen_rewrite(question, history_text), "qwen"

        return q, rewrite_type

    def _has_referential(self, question: str) -> bool:
        q = question.lower().strip()
        for word in self._ref_words:
            if _word_match(word, q):
                return True
        for w in q.split():
            for suf in self._ref_suffixes:
                clean = suf.rstrip("?.").strip()
                if w.endswith(clean) and len(w) > len(clean) + 2:
                    return True
        return False

    def _is_implicit_followup(self, question: str) -> bool:
        q = question.lower().strip()
        for pattern in self._followup_pats:
            try:
                if re.match(pattern, q):
                    return True
            except re.error:
                continue
        return False

    def _extract_last_subject(self, history_text: str) -> str:
        lines = history_text.strip().splitlines()
        last_user_line = ""
        for line in reversed(lines):
            if line.startswith("User:"):
                last_user_line = line[5:].strip().lower()
                break
        if not last_user_line:
            return ""
        for kw in sorted(self._plant_name_map.keys(), key=len, reverse=True):
            if _word_match(kw, last_user_line):
                return kw
        for kw, label in sorted(self._topic_kw.items(), key=lambda x: len(x[0]), reverse=True):
            if _word_match(kw, last_user_line):
                return label
        return ""

    def _resolve_suffix_referential(self, question: str, subject: str) -> str:
        for suf in ["nya"]:
            pattern = r"(\w{3,})" + re.escape(suf) + r"\b"
            new = re.sub(pattern, lambda m: f"{m.group(1)} {subject}", question, flags=re.IGNORECASE)
            if new != question:
                return new
        return question

    def _qwen_rewrite(self, question: str, history_text: str) -> str:
        """
        Qwen LLM fallback untuk query rewriting.
        Dipanggil HANYA jika rule-based tidak bisa resolve.
        Timeout 30 detik — jika gagal/timeout, return original question.
        """
        print("   [rewrite] Qwen fallback (ambiguous query)...")
        try:
            raw = self._qwen_rewrite_chain.invoke(
                {"history": history_text, "question": question}
            )
            rewritten = raw.strip().split("\n")[0].strip()
            r_lower   = rewritten.lower()

            # Validasi: bukan jawaban, max 25 kata, max 1 tanda tanya
            invalid = (
                len(r_lower.split()) > 25
                or rewritten.count("?") > 1
                or any(r_lower.startswith(m) for m in _REWRITE_ANSWER_MARKERS)
                or not rewritten  # kosong
            )
            if not invalid:
                print(f"   [rewrite] Qwen OK: '{question}' → '{rewritten}'")
                return rewritten
            print(f"   [rewrite] Qwen invalid output — using original")
        except Exception as e:
            # Tangkap timeout, connection error, dll tanpa crash
            err_str = str(e)
            if "timeout" in err_str.lower() or "timed out" in err_str.lower():
                print(f"   [rewrite] Qwen TIMEOUT — using original question")
            else:
                print(f"   [rewrite] Qwen error: {err_str[:80]} — using original")
        return question

    # ═══════════════════════════════════════════════════════════════════════════
    # [2] PLANT & WEATHER DETECTION — internal methods
    # ═══════════════════════════════════════════════════════════════════════════

    def _is_plant_query(self, original: str, rewritten: str) -> bool:
        orig = original.lower()
        rew  = rewritten.lower()
        for kw in self._plant_strong:
            if _word_match(kw, rew) or _word_match(kw, orig):
                print(f"   [plant: strong '{kw}' → PLANT]")
                return True
        has_weak   = any(_word_match(kw, orig) for kw in self._plant_weak)
        has_strong = any(_word_match(kw, orig) or _word_match(kw, rew) for kw in self._plant_strong)
        if has_weak and has_strong:
            print(f"   [plant: weak+strong → PLANT]")
            return True
        if has_weak:
            print(f"   [plant: weak only → SKIP plant API]")
        return False

    def _is_weather_query(self, question: str) -> bool:
        q = question.lower()
        for kw in WEATHER_KEYWORDS:
            if _word_match(kw, q):
                return True
        has_suit = any(phrase in q for phrase in _PLANTING_SUITABILITY_PHRASES)
        if has_suit:
            has_cult = any(_word_match(sig, q) for sig in _CULTIVATION_SIGNALS)
            if not has_cult:
                return True
        return False

    def _is_greeting(self, question: str) -> bool:
        q = question.lower().strip()
        for p in self.greeting_patterns:
            if _word_match(p, q) and len(q.split()) <= 6:
                return True
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # [3] PLANT API — internal methods
    # ═══════════════════════════════════════════════════════════════════════════

    # Common non-plant words that Qwen sometimes returns as plant names
    _QWEN_PLANT_BLOCKLIST = {
        # Seasons / time
        "spring", "summer", "fall", "winter", "autumn", "season", "monsoon",
        # Weather
        "rain", "sun", "light", "water", "soil", "land", "air", "earth",
        # Generic nouns
        "plant", "tree", "crop", "seed", "fruit", "leaf", "root", "flower",
        "grain", "vegetable", "herb", "shrub", "bush", "weed", "grass",
        "food", "farm", "field", "garden", "nature", "agriculture", "harvest",
        # Error-like outputs
        "none", "n/a", "not applicable", "unknown", "no plant",
    }

    def _extract_and_translate_plant(self, question: str) -> str | None:
        q = question.lower()
        for kw in sorted(self._plant_name_map.keys(), key=len, reverse=True):
            if _word_match(kw, q):
                mapped = self._plant_name_map[kw]
                print(f"   [plant-map] '{kw}' → '{mapped}'")
                return mapped
        print("   [plant-map] no direct match — trying Qwen fallback...")
        try:
            result     = self._plant_fallback_chain.invoke({"question": question})
            plant_name = result.strip().split("\n")[0].lower().strip()
            invalid    = (
                plant_name == "none"
                or not plant_name
                or len(plant_name.split()) > 4
                or any(w in plant_name for w in ["question", "answer", "->", ":"])
                or plant_name in self._QWEN_PLANT_BLOCKLIST
                or any(plant_name == bl or plant_name.startswith(bl + " ")
                       for bl in self._QWEN_PLANT_BLOCKLIST)
            )
            if not invalid:
                print(f"   [plant-map] Qwen: '{plant_name}'")
                return plant_name
            print(f"   [plant-map] Qwen: no valid plant (got: '{plant_name}')")
        except Exception as e:
            print(f"   [plant-map] Qwen error: {e}")
        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # [5] ANSWER TEMPLATE — internal methods
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_answer_prompt(self) -> ChatPromptTemplate:
        """
        Buat ChatPromptTemplate segar tiap query dengan tanggal/waktu aktual.
        [[NOW_*]] adalah placeholder string biasa — tidak konflik dengan {data}/{question}.
        """
        now   = datetime.now()
        hari  = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][now.weekday()]
        bulan = ["Januari","Februari","Maret","April","Mei","Juni",
                 "Juli","Agustus","September","Oktober","November","Desember"][now.month - 1]
        filled = (
            _ANSWER_TEMPLATE_BASE
            .replace("[[NOW_DATE]]", f"{now.day} {bulan} {now.year}")
            .replace("[[NOW_DAY]]",  hari)
            .replace("[[NOW_TIME]]", now.strftime("%H:%M"))
        )
        return ChatPromptTemplate.from_template(filled)

    # ═══════════════════════════════════════════════════════════════════════════
    # RETRIEVER BUILDER
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_retriever(
        self,
        has_history:     bool,
        include_plant:   bool = False,
        include_weather: bool = False,
    ) -> ContextualCompressionRetriever:
        retrievers = [self.primary_ensemble]
        weights    = [0.60]

        if has_history:
            retrievers.append(self.memory_retriever)
            weights.append(0.10)

        if include_plant and self.plant_store:
            retrievers.append(self.plant_retriever)
            weights.append(0.15)

        if include_weather and self.weather_store:
            retrievers.append(self.weather_retriever)
            weights.append(0.15)

        total   = sum(weights)
        weights = [w / total for w in weights]

        base = (
            EnsembleRetriever(retrievers=retrievers, weights=weights)
            if len(retrievers) > 1
            else self.primary_ensemble
        )
        return ContextualCompressionRetriever(
            base_compressor = self.compressor,
            base_retriever  = base,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION & MEMORY — internal methods
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_or_create_session(self, user_id: str) -> dict:
        with self._session_lock:
            if user_id not in self._sessions:
                session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self._sessions[user_id] = {"history": [], "session_id": session_id}
                print(f"🆕  New session for {user_id}: {session_id}")
            return self._sessions[user_id]

    def _format_history(self, history: list, max_turns: int = 3) -> str:
        """
        Format riwayat percakapan untuk disisipkan ke prompt.
        - Ambil max_turns giliran terakhir
        - Truncate jawaban panjang agar tidak memenuhi context window
        - Jawaban dipotong di 600 karakter agar LLM tetap punya ruang untuk {data}
        """
        if not history:
            return "Belum ada riwayat percakapan."

        MAX_ANSWER_CHARS = 600  # potong jawaban panjang — cukup untuk konteks referensial
        lines = []
        for q, a in history[-max_turns:]:
            a_trimmed = a[:MAX_ANSWER_CHARS] + "..." if len(a) > MAX_ANSWER_CHARS else a
            lines.append(f"User: {q}\nSugi: {a_trimmed}")
        return "\n\n".join(lines)

    def _format_docs(self, docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def _save_session_memory(self, user_id: str, session: dict):
        history = session.get("history", [])
        if not history:
            return

        summarize_chain = (
            ChatPromptTemplate.from_template(
                "Kamu adalah asisten yang merangkum percakapan pertanian.\n"
                "Berikut percakapan antara pengguna dan Sugi:\n{conversation}\n\n"
                "Tugasmu:\n"
                "1. Ekstrak HANYA fakta penting tentang pertanian\n"
                "2. Format: poin-poin singkat, maks 10 poin\n"
                "3. Fokus: tanaman, teknik, hama/penyakit, harga, rekomendasi\n"
                "4. Bahasa Indonesia\n"
                "5. Jika tidak ada fakta penting, tulis: TIDAK ADA INFORMASI PENTING\n\n"
                "Rangkuman Fakta Sesi:"
            )
            | self.model
            | StrOutputParser()
        )

        try:
            conv_text = "\n\n".join(f"User: {q}\nSugi: {a}" for q, a in history[-10:])
            summary   = summarize_chain.invoke({"conversation": conv_text}).strip()

            if "TIDAK ADA INFORMASI PENTING" in summary.upper() or not summary:
                print("ℹ️   No important facts — session not saved.")
                return

            session_id = session["session_id"]
            doc_id     = hashlib.md5(f"{user_id}_{session_id}_{summary[:50]}".encode()).hexdigest()

            if self.memory_store.get(ids=[doc_id])["ids"]:
                return

            doc = Document(
                page_content = summary,
                metadata     = {
                    "source":     "conversation_memory",
                    "user_id":    user_id,
                    "session_id": session_id,
                    "timestamp":  datetime.now().isoformat(),
                },
                id = doc_id,
            )
            self.memory_store.add_documents(documents=[doc], ids=[doc_id])
            print(f"💾  Memory saved for {user_id} (session: {session_id[:14]}...)")

        except Exception as e:
            print(f"⚠️   Memory save error for {user_id}: {e}")

    def _purge_expired_memories(self, ttl_days: int = MEMORY_TTL_DAYS):
        if ttl_days <= 0:
            return
        try:
            all_docs = self.memory_store.get(include=["metadatas"])
            if not all_docs["ids"]:
                return
            cutoff    = datetime.now().timestamp() - (ttl_days * 86400)
            to_delete = [
                doc_id
                for doc_id, meta in zip(all_docs["ids"], all_docs["metadatas"])
                if meta.get("timestamp") and
                   datetime.fromisoformat(meta["timestamp"]).timestamp() < cutoff
            ]
            if to_delete:
                self.memory_store.delete(ids=to_delete)
                print(f"🗑️   Purged {len(to_delete)} expired memory docs.")
        except Exception as e:
            print(f"⚠️   Memory purge error: {e}")

    def _load_or_build_bm25(self) -> BM25Retriever:
        cache_path = _ROOT / BM25_CACHE_PATH if not os.path.isabs(BM25_CACHE_PATH) else Path(BM25_CACHE_PATH)
        hash_path = cache_path.with_suffix(".hash")

        # ── [Fix 2] BM25 cache invalidation based on document IDs hash ──────
        # If .hash sidecar exists, compare stored hash with live ChromaDB ids hash.
        # This handles subsets/deletions accurately while retaining fast lookup checks.
        live_ids = vector_store._collection.get(include=[])["ids"]
        live_hash = hashlib.md5(str(sorted(live_ids)).encode()).hexdigest()
        
        if cache_path.exists():
            try:
                stored_hash = hash_path.read_text().strip() if hash_path.exists() else ""
                if stored_hash == live_hash:
                    print(f"📂  Loading BM25 from cache (Hash matched)...")
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
                else:
                    print(f"🔄  Dataset changed — rebuilding BM25...", flush=True)
                    cache_path.unlink(missing_ok=True)
                    hash_path.unlink(missing_ok=True)
            except Exception as _e:
                print(f"⚠️  BM25 cache check error: {_e} — rebuilding.")
                cache_path.unlink(missing_ok=True)

        print("🔨  Building BM25 index...")
        BATCH_SIZE    = 500
        docs_for_bm25 = []
        all_ids       = live_ids
        total         = len(all_ids)
        print(f"   Found {total:,} documents...")

        for offset in range(0, total, BATCH_SIZE):
            batch = vector_store._collection.get(
                ids     = all_ids[offset:offset+BATCH_SIZE],
                include = ["documents", "metadatas"],
            )
            for doc_text, meta in zip(batch["documents"], batch["metadatas"]):
                docs_for_bm25.append(Document(page_content=doc_text, metadata=meta or {}))
            print(f"   Fetched {min(offset+BATCH_SIZE, total):,}/{total:,}...", end="\r")

        retriever   = BM25Retriever.from_documents(docs_for_bm25)
        retriever.k = 4
        with open(cache_path, "wb") as f:
            pickle.dump(retriever, f)
        # Save collection hash for invalidation check on next startup
        hash_path.write_text(live_hash)
        print(f"\n✅  BM25 built ({len(docs_for_bm25):,} docs) and cached.")
        return retriever

    def _load_scope_config(self, path: str):
        cfg = _cp.ConfigParser()
        cfg.read(path, encoding="utf-8")

        def _parse(section, key) -> set:
            raw = cfg.get(section, key, fallback="")
            return {l.strip().lower() for l in raw.splitlines() if l.strip()}

        greetings = _parse("greetings", "patterns")
        allowed   = set()
        for sec in cfg.sections():
            if sec.startswith("allowed_"):
                allowed |= _parse(sec, "keywords")
        blocked = _parse("blocked_topics", "keywords") if cfg.has_section("blocked_topics") else set()
        refusal = cfg.get("refusal", "message", fallback=(
            "Maaf, pertanyaan tersebut berada di luar bidang keahlian Sugi. "
            "Apakah ada pertanyaan seputar pertanian yang bisa Sugi bantu? 🌾"
        ))
        print(f"📋  Scope config: {len(greetings)} greetings, "
              f"{len(allowed)} allowed, {len(blocked)} blocked.")
        return greetings, allowed, blocked, refusal

    def _is_in_scope(self, question: str) -> bool:
        q = question.lower().strip()
        for p in self.greeting_patterns:
            if _word_match(p, q):
                print(f"   [scope: greeting '{p}' → ALLOWED]")
                return True
        for kw in self.allowed_kw:
            if _word_match(kw, q):
                print(f"   [scope: '{kw}' → ALLOWED]")
                return True
        print(f"   [scope: no match → BLOCKED]")
        return False