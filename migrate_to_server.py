"""
migrate_to_server.py — Salin data dari embedded ChromaDB ke server (FIXED)

Fix dari versi sebelumnya:
  - ChromaDB server v1.x tidak menerima metadata={} (dict kosong)
  - Sekarang metadata=None kalau kosong atau None
  - Tambah verifikasi jumlah dokumen setelah migrasi

Jalankan:
  1. Pastikan ChromaDB server sudah jalan:
       python -m chromadb.cli.cli run --path ./chrome_longchain_db --port 8000
     ATAU (kalau chroma sudah di PATH):
       chroma run --path ./chrome_longchain_db --port 8000

  2. Buka terminal KEDUA, jalankan:
       python migrate_to_server.py

  3. Verifikasi output — setiap collection harus menunjukkan jumlah docs sama
"""

import chromadb

OLD_PATH  = "chrome_longchain_db"   # folder embedded DB lama
NEW_HOST  = "localhost"
NEW_PORT  = 8000

print(f"Membuka embedded DB lama di '{OLD_PATH}'...")
old_client = chromadb.PersistentClient(path=OLD_PATH)

print(f"Menghubungkan ke server {NEW_HOST}:{NEW_PORT}...")
new_client = chromadb.HttpClient(host=NEW_HOST, port=NEW_PORT)

# Test koneksi server
try:
    new_client.heartbeat()
    print("Koneksi server OK\n")
except Exception as e:
    print(f"GAGAL terhubung ke server: {e}")
    print("Pastikan server sudah jalan terlebih dahulu.")
    exit(1)

collections = old_client.list_collections()
print(f"Ditemukan {len(collections)} collection: {[c.name for c in collections]}\n")

total_migrated = 0

for col in collections:
    print(f"Migrasi collection: '{col.name}'...")
    old_col = old_client.get_collection(col.name)

    # Ambil semua data
    result = old_col.get(include=["documents", "metadatas", "embeddings"])
    total  = len(result["ids"])

    if total == 0:
        print(f"  Kosong — skip.\n")
        continue

    # FIX: metadata={} tidak diterima server v1.x → pakai None
    col_metadata = col.metadata if col.metadata else None

    try:
        new_col = new_client.get_or_create_collection(
            name=col.name,
            metadata=col_metadata,   # None kalau kosong
        )
    except Exception as e:
        print(f"  ERROR buat collection: {e}\n")
        continue

    # Upload dalam batch
    batch_size = 100
    uploaded   = 0

    # Cek keberadaan field sekali di luar loop — hindari numpy truth-value error
    has_docs = result.get("documents") is not None and len(result["documents"]) > 0
    has_meta = result.get("metadatas") is not None and len(result["metadatas"]) > 0
    has_emb  = result.get("embeddings") is not None and len(result["embeddings"]) > 0

    for i in range(0, total, batch_size):
        batch_ids  = result["ids"][i:i+batch_size]
        batch_docs = result["documents"][i:i+batch_size] if has_docs else None
        batch_meta = result["metadatas"][i:i+batch_size] if has_meta else None
        batch_emb  = result["embeddings"][i:i+batch_size] if has_emb  else None

        kwargs = {"ids": batch_ids}
        if batch_docs:
            kwargs["documents"] = batch_docs
        if batch_meta:
            cleaned_meta = [m if m else {} for m in batch_meta]
            kwargs["metadatas"] = cleaned_meta
        if batch_emb is not None:
            kwargs["embeddings"] = batch_emb

        try:
            new_col.upsert(**kwargs)
            uploaded += len(batch_ids)
            print(f"  Progress: {uploaded}/{total}", end="\r")
        except Exception as e:
            print(f"\n  ERROR batch {i}-{i+batch_size}: {e}")
            continue

    # Verifikasi
    server_count = new_col.count()
    status = "OK" if server_count == total else f"MISMATCH (expect {total})"
    print(f"  Selesai: {total} docs → server {server_count} docs [{status}]\n")
    total_migrated += uploaded

print(f"Migrasi selesai. Total {total_migrated} dokumen dipindahkan.")
print("\nVerifikasi collections di server:")
for col in new_client.list_collections():
    c = new_client.get_collection(col.name)
    print(f"  {col.name}: {c.count()} docs")