"""Isolate API tests on a throwaway SQLite file so schema changes do not hit ./securemail.db."""
from __future__ import annotations

import os
from pathlib import Path

_db = Path(__file__).resolve().parent / "_test.db"
if _db.exists():
    _db.unlink()
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_db}"
