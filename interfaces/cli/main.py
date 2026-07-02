"""
main.py — Jaghut v0.1L CLI Interface

Perubahan:
  [UPGRADE] Menggunakan JaghutCore (platform-agnostic engine)
  [NEW]     UserStore — persistent user ID & session
  [NEW]     Debug commands: !debug / !flags / !session / !memory / !stats
  [NEW]     Riwayat percakapan lama bisa diresume dari ChromaDB
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from core.jaghut_core import JaghutCore
from core.user_store import UserStore

# ─────────────────────────────────────────────
# CLI User ID — persistent lintas run
# ─────────────────────────────────────────────
CLI_USER_FILE = _ROOT / "data" / "cli_user.txt"


def get_or_create_cli_user_id() -> str:
    CLI_USER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CLI_USER_FILE.exists():
        user_id = CLI_USER_FILE.read_text(encoding="utf-8").strip()
        if user_id:
            return user_id
    import hashlib, datetime
    raw     = f"cli_{os.environ.get('USERNAME', 'user')}_{datetime.datetime.now().strftime('%Y%m%d')}"
    user_id = "cli_" + hashlib.md5(raw.encode()).hexdigest()[:12]
    CLI_USER_FILE.write_text(user_id, encoding="utf-8")
    print(f"🆔  CLI user ID dibuat: {user_id}")
    return user_id


# ─────────────────────────────────────────────
# Main CLI Loop
# ─────────────────────────────────────────────
def main():
    jaghut    = JaghutCore()
    users   = UserStore()
    user_id = get_or_create_cli_user_id()

    user_info   = users.get_or_create(
        user_id  = user_id,
        platform = "cli",
        username = os.environ.get("USERNAME", "user"),
    )
    visit_count = user_info.get("visit_count", 1)
    session     = jaghut._get_or_create_session(user_id)
    session_id  = session["session_id"]

    print(f"\n🌍  Jaghut v0.1L CLI siap! (user: {user_id}, kunjungan ke-{visit_count})")

    # Tampilkan memory sesi sebelumnya jika ada
    if visit_count > 1:
        memory = jaghut.get_memory_summary(user_id)
        if memory:
            print(f"\n📚  Konteks dari percakapan sebelumnya:\n{'-'*45}")
            print(memory[:500] + ("..." if len(memory) > 500 else ""))
            print(f"{'-'*45}\n")

    print(
        "Tips: ketik 'q' untuk keluar, 'clear' untuk reset sesi,\n"
        "      'history' untuk riwayat, '!companies' untuk generate insight,\n"
        "      '!debug' / '!flags' / '!session' / '!memory' / '!stats' untuk debug\n"
    )

    while True:
        print("\n-------------------------------------------")
        try:
            question = input("Tanyakan pertanyaan anda: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        print("-------------------------------------------\n")

        if not question:
            print("⚠️   Pertanyaan tidak boleh kosong.")
            continue

        # ── Shortcut commands ──────────────────────────────────────────────────
        if question.lower() == "q":
            jaghut.clear_session(user_id)
            users.update_last_seen(user_id)
            print("Sampai jumpa! 🌍")
            break

        if question.lower() == "clear":
            jaghut.clear_session(user_id)
            print("🗑️   Sesi direset. Mulai percakapan baru!")
            continue

        if question.lower() == "history":
            memory = jaghut.get_memory_summary(user_id)
            if memory:
                print(f"\n📚  Riwayat percakapanmu:\n{memory}")
            else:
                print("📭  Belum ada riwayat tersimpan.")
            continue

        # ── [6] Debug commands ────────────────────────────────────────────────
        if question.startswith("!"):
            if question.strip() == "!companies":
                print("🏢  Menjalankan Company Insight Engine (incremental)...")
                import subprocess as _sp
                result = _sp.run(
                    [sys.executable, str(_ROOT / "services" / "company_insight_engine.py"), "--now"],
                    capture_output=True, text=True, timeout=600,
                )
                print(result.stdout)
                if result.stderr:
                    print(f"⚠️  Stderr:\n{result.stderr}")
                continue
            handled, output = jaghut.handle_debug_command(question, user_id)
            if handled:
                print(output)
            else:
                print(f"⚠️   Command tidak dikenal: {question}")
                print("     Tersedia: !companies / !debug / !flags / !session / !memory / !stats")
            continue

        # ── Proses pertanyaan ─────────────────────────────────────────────────
        print("🤖  Generating answer:\n")
        response = jaghut.ask(user_id=user_id, question=question, platform="cli")
        # response sudah di-print secara streaming di dalam ask()
        # print ulang hanya jika tidak ada output (error)
        if response.startswith("⚠️"):
            print(response)
        print()


if __name__ == "__main__":
    main()