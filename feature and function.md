# JAGHUT v0.1L — ESG & Sustainability Intelligence Platform

# Feature & Function Documentation

---

## 1. Background

Perusahaan, lembaga keuangan, dan pemerintah di Indonesia dan Asia Tenggara menghadapi tekanan regulasi dan pasar yang semakin ketat terkait deforestasi, emisi karbon, hak asasi manusia, dan tata kelola lingkungan (ESG). Regulasi seperti **EUDR (European Union Deforestation Regulation)**, **ISPO (Indonesian Sustainable Palm Oil)** dan **RSPO** menuntut transparansi penuh dalam rantai pasok komoditas berisiko hutan.

Namun, data ESG tersebar di berbagai format — laporan PDF, spreadsheet Forest IQ, database internal, dan sumber terbuka — dan tidak ada sistem tunggal yang mampu mengintegrasikan, menganalisis, dan menyajikan intelijen ESG secara real-time dengan kemampuan tanya-jawab berbasis AI.

JAGHUT dibangun untuk menjembatani kesenjangan ini: sebuah platform intelijen ESG dan keberlanjutan yang menggunakan **Retrieval-Augmented Generation (RAG)** dengan model bahasa lokal untuk memberikan jawaban kontekstual, berbasis data, dan dapat diaudit.

---

## 2. Reason (Masalah yang Dipecahkan)

| Masalah | Solusi JAGHUT |
|---------|---------------|
| Data ESG tersebar di banyak format (CSV, PDF, database) | Satu sistem pencarian vektor terpadu (ChromaDB) yang mengindeks semua sumber data secara otomatis |
| Analisis ESG membutuhkan keahlian domain yang mendalam | AI dengan pengetahuan ESG spesifik (LLM `jaghut-v0.1L`) yang bisa menjawab pertanyaan kompleks |
| Sulit melacak perubahan skor Forest IQ antar waktu | Company Insight Engine yang memonitor perubahan assessment dan menghasilkan insight & rekomendasi otomatis |
| Greenwashing sulit dideteksi secara manual | NLP-driven evaluation dengan deteksi kesenjangan antara klaim dan data aktual |
| Regulasi berubah cepat (EUDR, ISPO, dll) | Sistem pembaruan data real-time dengan PDF/CSV watcher dan daily insight cron |
| Tim ESG membutuhkan akses cepat tanpa pelatihan teknis | Dua antarmuka: Telegram Bot (mudah, mobile) dan CLI (power user) |
| Data historis ESG tidak dimanfaatkan untuk pengambilan keputusan | Long-term memory dan session management yang menyimpan konteks percakapan |

---

## 3. Features

### 3.1 Core Platform

| Fitur | Deskripsi |
|-------|-----------|
| **Hybrid RAG Pipeline** | BM25 (sparse) + Dense Vector (mxbai-embed-large) + Cross-Encoder Reranker — akurasi hingga level kalimat |
| **3-Layer Query Rewriting** | Menangani pertanyaan referensial ("bagaimana dengan mereka?", "jelaskan lebih detail") tanpa kehilangan konteks |
| **Scope Guard** | Hanya menjawab topik ESG & lingkungan; menolak pertanyaan di luar domain dengan sopan |
| **Entity Detection** | Mendeteksi 199+ entitas ESG (perusahaan, regulasi, komoditas, framework) dengan keyword strong/weak dua tingkat |
| **Memory Jangka Panjang** | Ringkasan sesi percakapan disimpan di ChromaDB dengan TTL 14 hari |
| **Multi-Platform** | CLI (terminal) + Telegram Bot |
| **Streaming Output** | Jawaban ditampilkan real-time karakter per karakter |

### 3.2 Background Services

| Fitur | Deskripsi |
|-------|-----------|
| **CSV/XLSX Watcher** | Memonitor folder `data/raw_dataset/` — file baru langsung diindeks ke ChromaDB dengan chunk strategy spesifik per tipe data |
| **PDF Watcher** | Memonitor folder `data/raw_pdfs/` — mendeteksi tipe PDF (regulasi, jurnal, ESG, Forest IQ, climate) dan menerapkan chunk strategy berbeda |
| **Weather Crawler** | Mengambil data iklim real-time dari Open-Meteo (18 variabel cuaca, 92 hari historis + 16 hari prakiraan), menyimpan ringkasan harian |
| **Daily Insight Cron** | Setiap 12 jam, membaca semua data ChromaDB dan menghasilkan 5 jenis insight ESG ke MongoDB |
| **Company Insight Engine** | Membaca data assessment Forest IQ dari MongoDB, menghasilkan insight & rekomendasi per perusahaan (maks 500 karakter), menyimpan ke MongoDB + ChromaDB |

### 3.3 Forest IQ Integration

| Fitur | Deskripsi |
|-------|-----------|
| **Score Analysis** | Deforestation Exposure (0-5), Financial Materiality (0-5), Commitments Strength (0-4), Actions Reporting (0-6), Performance Reporting (0-10) |
| **Sub-Score Breakdown** | No-Deforestation Commitment, Human Rights Commitment, Upstream/Downstream Reporting, Human Rights Performance |
| **Greenwashing Detection** | Klasifikasi risiko greenwashing (Very Low s/d Very High) dengan indeks numerik |
| **Commodity Risk Mapping** | Pemetaan risiko per komoditas (Palm Oil, Pulp & Paper, Soy, Cocoa, Timber, dll) |
| **Incremental Updates** | Insight hanya dibuat ulang jika assessment berubah (deteksi via `updatedAt`) |

### 3.4 User Interface

| Fitur | Deskripsi |
|-------|-----------|
| **CLI Interface** | Chat terminal dengan user ID persisten, riwayat sesi, debugging commands |
| **Telegram Bot** | Bot multi-user dengan rate limiting, typing indicator, offline message catch-up, contact sharing |
| **Debug Commands** | `!debug` (10 query terakhir), `!flags` (flagged queries), `!session` (log sesi), `!memory` (memori tersimpan), `!stats` (statistik engine), `!companies` (trigger insight manual) |

### 3.5 Data Storage

| Fitur | Deskripsi |
|-------|-----------|
| **ChromaDB (Vector Store)** | 6 collections: `main_dataset`, `weather_data`, `plant_data`, `entity_data`, `conversation_memory`, `company_insights` |
| **MongoDB (Insight Store)** | 6 collections di `jaghut_insights`: `esg_insights`, `climate_insights`, `sector_insights`, `policy_insights`, `session_summaries`, `company_insight` |
| **MongoDB (Source Data)** | 4 collections di `test`: `companies`, `assessments`, `commodities`, `company_insight` |
| **Query Logging** | Semua query tercatat di `data/logs/queries.jsonl` dengan metadata lengkap + eval score |

---

## 4. Functions (Per Komponen)

### 4.1 `core/jaghut_core.py` — JaghutCore Engine

Fungsi utama mesin RAG. Memproses setiap pertanyaan melalui pipeline 8 langkah:

**Class: `JaghutCore`**

| Method | Fungsi |
|--------|--------|
| `__init__()` | Inisialisasi LLM (`jaghut-v0.1L` + `qwen2.5:1.5b`), embeddings (`mxbai-embed-large`), ChromaDB client, load semua file konfigurasi, inisialisasi retriever ensemble |
| `ask(user_id, question, platform)` | Pipeline utama: scope guard → query rewriting → entity detection → ESG API → hybrid retrieval → reranking → memory injection → answer generation → eval loop → session update → logging |
| `_is_in_scope(query, original)` | Cek apakah query termasuk topik yang diizinkan (ESG, iklim, keberlanjutan, atau sapaan) |
| `_maybe_rewrite(query, history, entities)` | 3-layer rewriting: fast-path → rule-based (0ms) → Qwen2.5 (30s timeout) |
| `_detect_entities(query)` | Deteksi entity ESG dua tingkat (strong/weak keywords) dari `esg_entities.ini` |
| `_detect_is_weather(query)` | Deteksi pertanyaan cuaca/iklim (80+ keywords) |
| `_build_retriever()` | Bangun ensemble retriever dengan bobot dinamis (BM25 30% + Dense Vector 30% + Memory 10% + Entity 10% + Company Insight 10% + Weather 10%) |
| `_get_or_create_session(user_id)` | Ambil atau buat sesi percakapan untuk user |
| `get_memory_summary(user_id)` | Ambil ringkasan memori jangka panjang untuk user |
| `clear_session(user_id)` | Hapus sesi aktif user |
| `handle_debug_command(command, user_id)` | Proses debug commands (!debug, !flags, !session, !memory, !stats) |

### 4.2 `core/esg_api.py` — ESG Entity & Perenual API

| Function | Fungsi |
|----------|--------|
| `search_entity_info(entity_name)` | Cari dokumen ESG tentang entity dari ChromaDB, cache di `entity_data` (TTL 14 hari) |
| `is_entity_cached(entity_name)` | Cek apakah entity sudah ada di cache dan belum expired |
| `get_cached_entity_docs(entity_name, k)` | Ambil dokumen entity yang sudah di-cache |
| `search_plant_info(species_name)` | Cari data tanaman dari Perenual API (species, hama, panduan perawatan), cache di `plant_data` (TTL 30 hari) |
| `fetch_plant_species(species_name)` | Panggil Perenual species search + detail endpoint |
| `fetch_pest_disease(query)` | Panggil Perenual pest/disease endpoint |
| `fetch_care_guides(species_id)` | Panggil Perenual care guide endpoint |

### 4.3 `core/eval_loop.py` — Faithfulness & Relevance Evaluation

| Function | Fungsi |
|----------|--------|
| `evaluate(answer, contexts, question)` | Evaluasi dua lapis: (1) lexical heuristic token overlap, (2) LLM-based evaluation jika heuristic tidak meyakinkan |
| `_lexical_heuristic(answer, contexts)` | Hitung token overlap antara jawaban dan konteks untuk deteksi halusinasi cepat |
| `_llm_eval(question, answer, contexts)` | Gunakan Qwen2.5 untuk menilai faithfulness (HIGH/MEDIUM/LOW) dan relevance (HIGH/MEDIUM/LOW) |

### 4.4 `core/query_logger.py` — Structured Query Logging

| Function | Fungsi |
|----------|--------|
| `log_query(**fields)` | Tulis satu record query ke `data/logs/queries.jsonl` |
| `tail_logs(n=10)` | Baca n query terakhir |
| `flagged_logs()` | Baca semua query yang di-flag (faithfulness/relevance LOW) |
| `session_logs(session_id)` | Baca log untuk sesi tertentu |
| `print_debug_report(n=10)` | Cetak laporan debug komprehensif ke terminal |

### 4.5 `core/user_store.py` — User Management

| Function | Fungsi |
|----------|--------|
| `get_or_create(user_id, platform, ...)` | Ambil atau buat profil user baru |
| `update_phone(user_id, phone)` | Simpan nomor telepon user |
| `update_last_seen(user_id)` | Update timestamp terakhir user online |
| `add_session(user_id, session_id)` | Tambahkan session ID ke riwayat user |
| `get(user_id)` | Ambil profil user |
| `get_display_name(user_id)` | Dapatkan nama tampilan user |
| `all_users()` | Daftar semua user |
| `stats()` | Statistik user (total user, aktif hari ini, dll) |

### 4.6 `services/vectorCSV.py` — CSV/XLSX Watcher & Indexer

| Function | Fungsi |
|----------|--------|
| `invalidate_bm25_cache()` | Tandai cache BM25 sebagai invalid (hash berubah, next startup rebuild) |
| **Auto-detection** | Deteksi tipe data dari header kolom: ESG → chunk 500/50, Price → 200/20, Policy → 600/80, Tabular → 300/30 |
| **Watchdog Loop** | Monitor `data/raw_dataset/` untuk file baru/modifikasi, skip yang sudah diindex |

### 4.7 `services/vectorpdf.py` — PDF Watcher & Indexer

| Function | Fungsi |
|----------|--------|
| **Auto-detection** | Deteksi tipe PDF dari nama file: regulasi → chunk 800/100, jurnal → 600/80, harga → 250/25, esg → 700/80, forest_iq → 500/50, climate → 600/80, carbon → 500/50 |
| **Dedup** | Skip file yang sudah pernah diindex (cek source filename) |
| **Watchdog Loop** | Monitor `data/raw_pdfs/` untuk PDF baru |

### 4.8 `services/vectorWeather.py` — Climate Data Crawler

| Function | Fungsi |
|----------|--------|
| `fetch_weather_data()` | Ambil data iklim 18 variabel dari Open-Meteo API, konversi ke daily summary, simpan ke ChromaDB `weather_data` |
| **Alert Generation** | Deteksi 7 jenis agronomic alerts: drought, extreme rain, heat stress, cold risk, disease risk, high wind, high water demand |
| **Polling** | Cek perubahan hari setiap 300 detik, re-fetch jika tanggal berganti |

### 4.9 `services/daily_insight.py` — ESG Insight Generator (12h Cron)

| Function | Fungsi |
|----------|--------|
| `_generate_company_insight()` | Group data `main_dataset` per perusahaan, generate insight ESG 3-5 kalimat via Qwen2.5, simpan ke `esg_insights` |
| `_generate_climate_insight()` | Group data `weather_data` per lokasi, generate analisis risiko iklim, simpan ke `climate_insights` |
| `_generate_sector_insight()` | Per sektor (palm oil, pulp & paper, mining, dll), generate insight tren ESG, simpan ke `sector_insights` |
| `_generate_policy_insight()` | Ekstrak 3-5 insight regulasi dari `main_dataset`, kategorikan (regulatory, market, climate, social, compliance), simpan ke `policy_insights` |
| `_generate_session_summary()` | Baca `conversation_memory`, deteksi topik yang dibahas, simpan ringkasan ke `session_summaries` |
| `_build_mongo_client()` | Buat koneksi MongoDB Atlas dengan TLS/certifi, retry 3x dengan exponential backoff |

### 4.10 `services/company_insight_engine.py` — ForestIQ Company Insight Engine

| Function | Fungsi |
|----------|--------|
| `_build_assessment_text(company, assessments, commodities_map)` | Format data assessment ForestIQ menjadi teks terstruktur untuk prompt LLM |
| `_fetch_rag_data(company_name)` | Cari dokumen tambahan dari ChromaDB `main_dataset` relevan dengan company (similarity search) |
| `_generate_insight(company, assessments, commodities_map)` | Generate insight 3-4 kalimat (400-500 karakter) — analisis skor ForestIQ, kekuatan, kelemahan |
| `_generate_recommendation(company, assessments, commodities_map)` | Generate rekomendasi 3-4 kalimat (400-500 karakter) — tindakan perbaikan berdasarkan skor terendah |
| `_needs_update(existing, assessments)` | Cek apakah insight perlu di-update berdasarkan timestamp `updatedAt` assessment |
| `process_all_companies()` | Main loop: baca semua company aktif → group assessment → generate insight & recommendation → save ke MongoDB (test + jaghut_insights) + ChromaDB |
| `run_once()` | Reset & generate semua insight dari awal |
| `run_loop(poll_seconds)` | Incremental loop: generate missing + update changed, polling setiap N detik |

### 4.11 `interfaces/cli/main.py` — CLI Interface

| Function | Fungsi |
|----------|--------|
| `get_or_create_cli_user_id()` | Buat/ambil user ID persisten dari `data/cli_user.txt` |
| `main()` | REPL loop: input pertanyaan → panggil `JaghutCore.ask()` → print streaming response. Hotkeys: `q` (quit), `clear` (reset sesi), `history` (riwayat), `!` commands |

### 4.12 `interfaces/telegram/telegram_bot.py` — Telegram Bot

| Function | Fungsi |
|----------|--------|
| `/start` | Sambutan + panduan awal |
| `/help` | Daftar command |
| `/about` | Informasi tentang Jaghut |
| `/history` | Riwayat percakapan user |
| `/clear` | Reset sesi user |
| `/contact` | Bagikan nomor telepon |
| `/debug` | Panel debug commands |
| `handle_message()` | Proses pesan: rate limit → user store → scope guard → `JaghutCore.ask()` → split panjang → kirim ke Telegram |
| `handle_contact()` | Simpan nomor telepon dari contact share |
| `_send_long()` | Split pesan >4000 karakter menjadi beberapa pesan |
| `_keep_typing()` | Kirim typing indicator selama LLM memproses |

### 4.13 `start_all.py` — Master Startup Script

| Function | Fungsi |
|----------|--------|
| **Startup** | Cek readiness ChromaDB (port 8000) → launch 6 background service subprocesses + Telegram Bot |
| **Health Check** | Periksa semua service berjalan dalam 5 detik pertama |
| **Auto-Restart** | Jika service crash, restart maksimal 3 kali |
| **Clean Shutdown** | Handle Ctrl+C → terminate semua subprocess |

---

## 5. Alur Data Lengkap

```
FOREST IQ (CSV)            
    │                              
    ▼                        
┌──────────────────────┐    
│ vectorCSV.py         │    
│ (Watchdog: .csv/.xls)│    
└──────────┬───────────┘    
           │               
           ▼               
┌──────────────────────┐    
│ ChromaDB              │    
│ main_dataset          │    
│ (6 collections total) │    
└──────────────────────┘    
           │               
           ├──────────────────────────┐
           │                          │
           ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐
│ Company Insight      │   │ JaghutCore            │
│ Engine (300s poll)   │   │ (RAG Pipeline)        │
│                       │   │                       │
│ MongoDB test DB       │   │ User Question         │
│ → companies           │   │ → Scope Guard         │
│ → assessments         │   │ → Query Rewriting     │
│ → commodities         │   │ → Entity Detection    │
│                       │   │ → Hybrid Retrieval    │
│ Insight + Recommend   │   │ → Reranker            │
│ → MongoDB (2 DBs)     │   │ → LLM                 │
│ → ChromaDB            │   │ → Eval Loop           │
│ company_insights      │   │ → Response            │
└──────────────────────┘   └──────────────────────┘
                                    │
                                    ▼
                           ┌──────────────────────┐
                           │ Daily Insight Cron    │
                           │ (12h)                 │
                           │                       │
                           │ → 5 insight types     │
                           │ → MongoDB jaghut_      │
                           │   insights (6 coll)    │
                           └──────────────────────┘
```

---

## 6. Teknologi yang Digunakan

| Komponen | Teknologi |
|----------|-----------|
| **Primary LLM** | Llama 3.2 custom-tuned → `jaghut-v0.1L` (via Ollama, temp=0.3) |
| **Utility LLM** | `qwen2.5:1.5b` — query rewriting, entity extraction, eval loop, insight generation |
| **Embedding Model** | `mxbai-embed-large` (1024-dim) via Ollama |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector Database** | ChromaDB Server (localhost:8000, HTTP client) |
| **Insight Database** | MongoDB Atlas (pymongo + certifi TLS) |
| **Sparse Retriever** | BM25Okapi via `rank_bm25` |
| **Framework** | LangChain (OllamaLLM, OllamaEmbeddings, Chroma, EnsembleRetriever, ContextualCompressionRetriever) |
| **Filesystem Watchdog** | `watchdog` library (inotify-style) |
| **Climate API** | Open-Meteo (18 variables, 92-day history + 16-day forecast) |
| **Biodiversity API** | Perenual (species, pest/disease, care guides) |
| **Telegram** | `python-telegram-bot>=21.0` (async) |
| **PDF Processing** | `pypdf` (PyPDFLoader) |
| **Runtime** | Python 3.14, Windows/Linux |

---

## 7. Keterbatasan & Catatan

- **Bahasa**: Primer Bahasa Indonesia, sekunder English
- **LLM Lokal**: Semua model berjalan via Ollama lokal — tidak ada API call ke cloud LLM
- **Data Forest IQ**: Bergantung pada dataset Forest IQ yang diupload ke `data/raw_dataset/` dan MongoDB `test` database
- **Koneksi Internet**: Diperlukan untuk MongoDB Atlas, Open-Meteo API, dan Perenual API (LLM dan ChromaDB lokal)
- **ChromaDB**: Harus berjalan di port 8000 sebelum service apa pun dimulai

---

*Dokumen ini diperbarui: Juli 2026 · JAGHUT v0.1L*
