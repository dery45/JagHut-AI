"""
user_store.py — Persistent user management untuk Jaghut v0.1L

Menyimpan data user ke users.json (lokal, ringan, tidak butuh DB tambahan).
Data yang disimpan per user:
  - user_id       : ID unik (Telegram user_id atau generated untuk CLI)
  - platform      : "telegram" | "cli"
  - username      : Telegram @username (jika ada)
  - full_name     : Nama lengkap
  - phone_number  : Nomor HP (jika user share contact di Telegram)
  - first_seen    : Timestamp pertama kali interaksi
  - last_seen     : Timestamp interaksi terakhir
  - visit_count   : Jumlah sesi
  - session_ids   : List session ID yang pernah dibuat
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


_ROOT = Path(__file__).resolve().parent.parent
USER_DB_PATH = _ROOT / "data" / "users.json"


class UserStore:
    def __init__(self, path: Path = USER_DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save(self):
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_or_create(
        self,
        user_id:   str,
        platform:  str = "cli",
        username:  str = "",
        full_name: str = "",
    ) -> dict:
        """
        Ambil data user jika ada, buat baru jika belum ada.
        Update visit_count dan last_seen setiap kali dipanggil.
        """
        now = datetime.now().isoformat()

        if user_id not in self._data:
            self._data[user_id] = {
                "user_id":      user_id,
                "platform":     platform,
                "username":     username,
                "full_name":    full_name,
                "phone_number": None,
                "first_seen":   now,
                "last_seen":    now,
                "visit_count":  1,
                "session_ids":  [],
            }
            print(f"👤 New user registered: {user_id} ({platform})")
        else:
            self._data[user_id]["visit_count"] += 1
            self._data[user_id]["last_seen"]    = now
            # Update username/nama jika berubah
            if username:
                self._data[user_id]["username"]   = username
            if full_name:
                self._data[user_id]["full_name"]  = full_name

        self._save()
        return self._data[user_id]

    def update_phone(self, user_id: str, phone: str):
        """Simpan nomor HP user."""
        if user_id in self._data:
            self._data[user_id]["phone_number"] = phone
            self._save()
            print(f"📱 Phone updated for {user_id}: {phone}")

    def update_last_seen(self, user_id: str):
        if user_id in self._data:
            self._data[user_id]["last_seen"] = datetime.now().isoformat()
            self._save()

    def add_session(self, user_id: str, session_id: str):
        """Tambahkan session_id ke riwayat user."""
        if user_id in self._data:
            sessions = self._data[user_id].setdefault("session_ids", [])
            if session_id not in sessions:
                sessions.append(session_id)
                # Simpan hanya 50 session terakhir
                self._data[user_id]["session_ids"] = sessions[-50:]
                self._save()

    def get(self, user_id: str) -> Optional[dict]:
        return self._data.get(user_id)

    def get_display_name(self, user_id: str) -> str:
        user = self._data.get(user_id, {})
        return (
            user.get("full_name")
            or user.get("username")
            or user_id
        )

    def all_users(self) -> list[dict]:
        return list(self._data.values())

    def stats(self) -> dict:
        users    = self.all_users()
        telegram = sum(1 for u in users if u.get("platform") == "telegram")
        cli      = sum(1 for u in users if u.get("platform") == "cli")
        with_phone = sum(1 for u in users if u.get("phone_number"))
        return {
            "total":      len(users),
            "telegram":   telegram,
            "cli":        cli,
            "with_phone": with_phone,
        }