"""
telegram_bot.py — Jaghut v0.1L Telegram Bot Integration

Fitur:
  - Multi-user session via user_id Telegram
  - Riwayat percakapan persistent di ChromaDB
  - Nomor HP tersimpan jika user share contact
  - Context percakapan dilanjutkan di sesi berikutnya
  - Debug commands: !debug / !flags / !session / !memory / !stats
  - Command: /start /help /history /clear /contact /about /debug
  - Offline message catch-up via persistent offset (Option B)
        → Semua pesan yang masuk saat bot mati akan dijawab saat bot hidup lagi
        → Offset terakhir disimpan di data/telegram_offset.json
        → Telegram menyimpan pesan hingga ~24 jam

Fixes:
  - Pesan offline tidak dikirim dua kali (get_updates ack sebelum run_polling)
  - Tidak ada Markdown parse error dari respons LLM (kirim plain text, tanpa parse_mode)
  - Import warning Pyrefly diabaikan dengan # type: ignore (false alarm — venv beda interpreter)

Setup:
  pip install "python-telegram-bot>=21.0"
  Isi TELEGRAM_BOT_TOKEN di .env
"""

import sys
import os
import json
import asyncio
import time
from pathlib import Path

# ─── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / "config" / ".env")

# type: ignore comments di bawah ini untuk suppress false alarm Pyrefly.
# python-telegram-bot SUDAH terinstall di venv — Pyrefly salah baca interpreter.
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove  # type: ignore[import]
from telegram.ext import (  # type: ignore[import]
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ChatAction  # type: ignore[import]

from core.jaghut_core import JaghutCore
from core.user_store import UserStore

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
MAX_MESSAGE_LENGTH = 4000   # Telegram limit 4096, buffer 96 chars
RATE_LIMIT_SECS    = 3.0    # Minimum detik antar request per user

# Path file penyimpan offset terakhir yang diproses
OFFSET_FILE = _ROOT / "data" / "telegram_offset.json"

# User yang diizinkan pakai debug commands (kosong = semua user bisa)
DEBUG_ALLOWED_USERS: set[str] = set(
    os.getenv("DEBUG_ALLOWED_USERS", "").split(",")
) - {""}


# ─────────────────────────────────────────────────────────────────────────────
# Offset persistence helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_offset() -> int:
    """Baca update_id terakhir yang berhasil diproses dari disk."""
    try:
        if OFFSET_FILE.exists():
            data = json.loads(OFFSET_FILE.read_text(encoding="utf-8"))
            return int(data.get("offset", 0))
    except Exception as e:
        print(f"⚠️  Gagal membaca offset file: {e}")
    return 0


def save_offset(offset: int) -> None:
    """Simpan update_id terakhir ke disk agar persistent antar restart."""
    try:
        OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
        OFFSET_FILE.write_text(
            json.dumps({"offset": offset}, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"⚠️  Gagal menyimpan offset file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Bot class
# ─────────────────────────────────────────────────────────────────────────────
class JaghutTelegramBot:

    def __init__(self):
        if not TELEGRAM_TOKEN:
            raise ValueError(
                "❌ TELEGRAM_BOT_TOKEN tidak ditemukan di .env!\n"
                "Isi nilai TELEGRAM_BOT_TOKEN di file .env"
            )
        self.jaghut  = JaghutCore()
        self.users = UserStore()
        self._last_request: dict[str, float] = {}
        print("🤖  Jaghut Telegram Bot initialized.")

    # ─────────────────────────────────────────────────────────────────────────
    # Offline catch-up: proses pesan yang masuk saat bot mati
    # ─────────────────────────────────────────────────────────────────────────
    async def _process_offline_backlog(self, app: Application) -> None:
        """
        Dipanggil sekali saat startup via post_init.

        1. Baca offset terakhir dari disk
        2. Fetch semua update pending di Telegram sejak offset itu
        3. Proses setiap update teks (bukan command) → kirim jawaban plain text
        4. Simpan offset per-update (crash-safe), lalu acknowledge ke Telegram
           dengan satu get_updates(offset=final) agar run_polling tidak replay ulang
        """
        saved_offset = load_offset()
        print(f"📬  Memeriksa pesan offline sejak offset {saved_offset}...")

        try:
            updates = await app.bot.get_updates(
                offset          = saved_offset if saved_offset > 0 else None,
                limit           = 100,
                timeout         = 10,
                allowed_updates = ["message"],
            )
        except Exception as e:
            print(f"⚠️  Gagal mengambil update offline: {e}")
            return

        if not updates:
            print("✅  Tidak ada pesan offline yang tertunda.")
            return

        print(f"📨  Ditemukan {len(updates)} update offline. Memproses...")

        for update in updates:
            try:
                await self._handle_offline_update(app, update)
            except Exception as e:
                print(f"⚠️  Error memproses update {update.update_id}: {e}")
            finally:
                # Simpan progress per-update: jika crash di tengah, tidak repeat dari awal
                save_offset(update.update_id + 1)

        final_offset = updates[-1].update_id + 1
        save_offset(final_offset)

        # KUNCI anti-double-send: acknowledge semua update yang sudah diproses ke
        # server Telegram. Tanpa ini, run_polling() akan menerima update yang sama
        # lagi dan handle_message akan mengirim respons kedua kalinya.
        try:
            await app.bot.get_updates(offset=final_offset, limit=1, timeout=3)
        except Exception:
            pass

        print(f"✅  Selesai memproses {len(updates)} pesan offline. Polling mulai dari offset {final_offset}.")

    async def _handle_offline_update(self, app: Application, update: Update) -> None:
        """
        Proses satu update offline dan kirim balasan ke user.

        PENTING: Tidak menggunakan parse_mode=Markdown sama sekali.
        Respons dari LLM sering mengandung karakter *, _, ** yang tidak
        berpasangan, menyebabkan Telegram error "Can't parse entities".
        Solusinya: kirim semua pesan sebagai plain text biasa.
        """
        msg = update.message
        if not msg:
            return  # Bukan pesan (edited_message, dll)

        chat_id = msg.chat_id
        user    = msg.from_user
        user_id = str(user.id) if user else None
        text    = (msg.text or "").strip()

        if not user_id or not text:
            return

        # Lewati commands (/start, /help, dll) — tidak relevan di-replay
        if text.startswith("/"):
            print(f"   ↪ Lewati command offline: {text[:40]}")
            return

        # Lewati debug commands (!debug, dll)
        if text.startswith("!"):
            print(f"   ↪ Lewati debug command offline: {text[:40]}")
            return

        # Lewati tombol keyboard
        if text == "⏭️ Lewati":
            return

        print(f"   ↪ Menjawab pesan offline dari user {user_id}: {text[:60]}...")

        # Pastikan user ada di store
        self.users.get_or_create(
            user_id   = user_id,
            platform  = "telegram",
            username  = getattr(user, "username", "") or "",
            full_name = getattr(user, "full_name", "") or "",
        )

        # Dapatkan jawaban dari JaghutCore
        response = await asyncio.to_thread(
            self.jaghut.ask,
            user_id  = user_id,
            question = text,
            platform = "telegram",
        )

        # ── Kirim header (plain text, tidak ada parse_mode) ───────────────────
        header = f"📭 Pesan kamu saat bot offline telah diproses:\n\"{text[:200]}\""
        await app.bot.send_message(chat_id=chat_id, text=header)
        await asyncio.sleep(0.3)

        # ── Kirim jawaban tanpa parse_mode ────────────────────────────────────
        # Alasan: LLM menghasilkan *, _, ** yang tidak berpasangan → Telegram
        # error "Can't parse entities" jika parse_mode diaktifkan.
        if not response:
            await app.bot.send_message(
                chat_id = chat_id,
                text    = "⚠️ Tidak ada respons dari Jaghut untuk pertanyaan ini.",
            )
            return

        for i in range(0, len(response), MAX_MESSAGE_LENGTH):
            await app.bot.send_message(
                chat_id = chat_id,
                text    = response[i : i + MAX_MESSAGE_LENGTH],
            )
            if i + MAX_MESSAGE_LENGTH < len(response):
                await asyncio.sleep(0.3)

        # Throttle antar user agar tidak kena flood limit Telegram
        await asyncio.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # /start
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user      = update.effective_user
        user_id   = str(user.id)
        user_info = self.users.get_or_create(
            user_id   = user_id,
            platform  = "telegram",
            username  = user.username or "",
            full_name = user.full_name or "",
        )

        is_returning = user_info.get("visit_count", 1) > 1
        if is_returning:
            last_visit = user_info.get("last_seen", "sebelumnya")
            greeting = (
                f"Halo lagi, {user.first_name}! 👋\n"
                f"Selamat datang kembali di Jaghut v0.1L.\n"
                f"Kunjungan terakhirmu: {last_visit}\n\n"
                f"Ada yang bisa Jaghut bantu hari ini? 🌍"
            )
        else:
            greeting = (
                f"Halo, {user.first_name}! 👋\n\n"
                f"Saya Jaghut v0.1L, asisten ESG dan kecerdasan keberlanjutan digital.\n\n"
                f"Saya siap membantu kamu seputar:\n"
                f"🌍 ESG & analisis keberlanjutan\n"
                f"🌲 Deforestasi & konservasi hutan\n"
                f"🌤 Risiko iklim & perubahan iklim\n"
                f"📊 Forest IQ & metrik perusahaan\n"
                f"🔍 Greenwashing detection & OSINT\n"
                f"📈 Regulasi lingkungan & kepatuhan\n\n"
                f"Ketik pertanyaanmu langsung, atau /help untuk panduan."
            )

        contact_btn = KeyboardButton("📱 Bagikan Nomor HP", request_contact=True)
        skip_btn    = KeyboardButton("⏭️ Lewati")
        markup = ReplyKeyboardMarkup(
            [[contact_btn], [skip_btn]],
            resize_keyboard=True,
            one_time_keyboard=True,
        )
        await update.message.reply_text(greeting, reply_markup=markup)

    # ─────────────────────────────────────────────────────────────────────────
    # /help
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "📖 Panduan Jaghut Bot\n\n"
            "Command:\n"
            "/start   — Mulai atau restart percakapan\n"
            "/help    — Panduan ini\n"
            "/history — Riwayat percakapan sebelumnya\n"
            "/clear   — Reset sesi berjalan\n"
            "/contact — Perbarui nomor HP\n"
            "/about   — Info tentang Jaghut\n"
            "/debug   — Panel debug (stats & logs)\n\n"
            "Cara bertanya:\n"
            "Ketik pertanyaan langsung, contoh:\n"
            "• Bandingkan skor Forest IQ Wilmar dan APRIL\n"
            "• Apa itu EUDR dan dampaknya pada sawit Indonesia?\n"
            "• Analisis greenwashing laporan keberlanjutan terbaru\n\n"
            "⚠️ Jaghut hanya menjawab topik ESG & keberlanjutan."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # /history
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        summary = self.jaghut.get_memory_summary(user_id)
        if not summary:
            await update.message.reply_text(
                "📭 Belum ada riwayat percakapan tersimpan.\n"
                "Mulailah bertanya, dan Jaghut akan mengingat topik-topik penting!"
            )
            return
        await self._send_long(update, f"📚 Riwayat Percakapan Kamu:\n\n{summary}")

    # ─────────────────────────────────────────────────────────────────────────
    # /clear
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.jaghut.clear_session(str(update.effective_user.id))
        await update.message.reply_text(
            "🗑️ Riwayat sesi ini telah dihapus.\n"
            "Percakapan baru dimulai! Ada yang ingin kamu tanyakan pada Jaghut? 🌍"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # /about
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_about(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🌍 Jaghut v0.1L\n\n"
            "Asisten ESG & Kecerdasan Keberlanjutan:\n"
            "• Model LLM lokal (Ollama)\n"
            "• RAG (Retrieval-Augmented Generation)\n"
            "• Hybrid search (BM25 + Vector)\n"
            "• Forest IQ Integration\n"
            "• Climate intelligence\n"
            "• Greenwashing detection\n"
            "• Eval loop (faithfulness + relevance)\n\n"
            "Dikembangkan oleh BU Team."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # /contact
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        contact_btn = KeyboardButton("📱 Bagikan Nomor HP", request_contact=True)
        markup = ReplyKeyboardMarkup(
            [[contact_btn]],
            resize_keyboard=True,
            one_time_keyboard=True,
        )
        await update.message.reply_text(
            "Silakan bagikan nomor HPmu agar Jaghut bisa memberikan layanan lebih personal.",
            reply_markup=markup,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # /debug
    # ─────────────────────────────────────────────────────────────────────────
    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if DEBUG_ALLOWED_USERS and user_id not in DEBUG_ALLOWED_USERS:
            await update.message.reply_text("⛔ Akses debug tidak diizinkan.")
            return
        await update.message.reply_text(
            "🔧 Debug Panel\n\n"
            "Kirim salah satu command berikut:\n"
            "!debug       — Laporan 10 query terakhir\n"
            "!flags       — Query yang di-flag eval\n"
            "!session     — Log sesi ini\n"
            "!memory      — Memory tersimpan untukmu\n"
            "!stats       — Statistik JaghutCore\n"
            "!offset      — Lihat offset tersimpan saat ini\n"
            "!companies   — Generate company insight manual"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Handle contact share
    # ─────────────────────────────────────────────────────────────────────────
    async def handle_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        contact = update.message.contact
        user_id = str(update.effective_user.id)
        if contact:
            self.users.update_phone(user_id, contact.phone_number)
            await update.message.reply_text(
                f"✅ Nomor HP {contact.phone_number} berhasil disimpan.\n"
                f"Terima kasih! Sekarang kamu bisa langsung bertanya. 🌾"
            )
        else:
            await update.message.reply_text("⚠️ Tidak dapat membaca nomor HP.")

    # ─────────────────────────────────────────────────────────────────────────
    # Handle tombol "Lewati"
    # ─────────────────────────────────────────────────────────────────────────
    async def handle_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Baik! Langsung saja ketik pertanyaanmu. 🌾",
            reply_markup=ReplyKeyboardRemove(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Handle pesan teks biasa (online / real-time)
    # ─────────────────────────────────────────────────────────────────────────
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user     = update.effective_user
        user_id  = str(user.id)
        question = update.message.text.strip()

        # Rate limiting
        now = time.time()
        if user_id in self._last_request and now - self._last_request[user_id] < RATE_LIMIT_SECS:
            await update.message.reply_text("⏳ Harap tunggu beberapa detik sebelum bertanya lagi.")
            return
        self._last_request[user_id] = now

        # Pastikan user ada di store & update last_seen
        self.users.get_or_create(
            user_id   = user_id,
            platform  = "telegram",
            username  = user.username or "",
            full_name = user.full_name or "",
        )
        self.users.update_last_seen(user_id)

        # Simpan offset setiap pesan online agar backlog tidak replay ulang
        save_offset(update.update_id + 1)

        # ── Debug commands via chat ───────────────────────────────────────────
        if question.startswith("!"):
            if DEBUG_ALLOWED_USERS and user_id not in DEBUG_ALLOWED_USERS:
                await update.message.reply_text("⛔ Akses debug tidak diizinkan.")
                return

            # !offset — cek offset tersimpan
            if question.strip() == "!offset":
                current = load_offset()
                await update.message.reply_text(
                    f"Offset tersimpan saat ini: {current}\nFile: {OFFSET_FILE}"
                )
                return

            # !companies — jalankan company insight engine
            if question.strip() == "!companies":
                await update.message.reply_text("🏢  Menjalankan Company Insight Engine...")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable,
                        str(Path(__file__).resolve().parent.parent.parent / "services" / "company_insight_engine.py"),
                        "--now",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
                    output = stdout.decode() if stdout else ""
                    if stderr:
                        output += f"\n⚠️  Stderr:\n{stderr.decode()}"
                    if not output.strip():
                        output = "✅  Selesai (tidak ada output detail)."
                except asyncio.TimeoutError:
                    output = "⏱️  Timeout — Company Insight Engine memakan waktu terlalu lama."
                    proc.terminate()
                except Exception as e:
                    output = f"❌  Error: {e}"
                await self._send_long(update, output)
                return

            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            handled, output = await asyncio.to_thread(
                self.jaghut.handle_debug_command, question, user_id
            )
            if handled:
                await self._send_long(update, output or "✅ Done.")
            else:
                await update.message.reply_text(
                    f"⚠️ Command tidak dikenal: {question}\n"
                    f"Tersedia: !debug / !flags / !session / !memory / !stats / !offset / !companies"
                )
            return

        # ── Pertanyaan biasa ──────────────────────────────────────────────────
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            self._keep_typing(update.effective_chat.id, context.bot, stop_typing)
        )

        try:
            response = await asyncio.to_thread(
                self.jaghut.ask,
                user_id  = user_id,
                question = question,
                platform = "telegram",
            )
        finally:
            stop_typing.set()
            await typing_task

        await self._send_long(update, response)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: typing indicator loop
    # ─────────────────────────────────────────────────────────────────────────
    async def _keep_typing(self, chat_id, bot, stop_event: asyncio.Event):
        while not stop_event.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            for _ in range(8):
                if stop_event.is_set():
                    break
                await asyncio.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: kirim pesan panjang dengan chunking (plain text, tanpa parse_mode)
    # ─────────────────────────────────────────────────────────────────────────
    async def _send_long(self, update: Update, text: str):
        if not text:
            await update.message.reply_text("⚠️ Tidak ada respons dari Jaghut.")
            return
        for i in range(0, len(text), MAX_MESSAGE_LENGTH):
            await update.message.reply_text(text[i : i + MAX_MESSAGE_LENGTH])

    # ─────────────────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        app = (
            Application.builder()
            .token(TELEGRAM_TOKEN)
            .build()
        )

        # ── Register handlers ─────────────────────────────────────────────────
        app.add_handler(CommandHandler("start",   self.cmd_start))
        app.add_handler(CommandHandler("help",    self.cmd_help))
        app.add_handler(CommandHandler("history", self.cmd_history))
        app.add_handler(CommandHandler("clear",   self.cmd_clear))
        app.add_handler(CommandHandler("about",   self.cmd_about))
        app.add_handler(CommandHandler("contact", self.cmd_contact))
        app.add_handler(CommandHandler("debug",   self.cmd_debug))

        app.add_handler(MessageHandler(filters.CONTACT,                  self.handle_contact))
        app.add_handler(MessageHandler(filters.Regex("^⏭️ Lewati$"),     self.handle_skip))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  self.handle_message))

        # ── Offline catch-up: jalankan sebelum polling dimulai ────────────────
        async def post_init(application: Application) -> None:
            await self._process_offline_backlog(application)

        app.post_init = post_init

        print(f"🚀  Jaghut Telegram Bot berjalan... (token: {TELEGRAM_TOKEN[:10]}...)")
        app.run_polling(
            allowed_updates      = Update.ALL_TYPES,
            drop_pending_updates = False,
        )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    JaghutTelegramBot().run()