# utils/db.py
import sqlite3, time
from typing import Dict, Optional, Tuple

_CONN = None

def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect("app.db", check_same_thread=False)
        _CONN.execute("""
          CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            pass_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            fav_dance TEXT NOT NULL,
            fav_color_name TEXT NOT NULL,
            fav_color_hex TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
          )
        """)
        _CONN.execute("""
          CREATE TABLE IF NOT EXISTS prefs (
            user_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (user_id, key)
          )
        """)
        _CONN.commit()
    return _CONN

def create_user(username: str, pass_hash: str, salt: str,
                fav_dance: str, fav_color_name: str, fav_color_hex: str) -> bool:
    """Returns True on success, False if username already exists."""
    now = time.time()
    try:
        _conn().execute(
            "INSERT INTO users(username, pass_hash, salt, fav_dance, fav_color_name, fav_color_hex, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (username, pass_hash, salt, fav_dance, fav_color_name, fav_color_hex, now, now)
        )
        _conn().commit()
        return True
    except sqlite3.IntegrityError:
        return False

def update_user_login(username: str, pass_hash: str, salt: str,
                      fav_dance: str, fav_color_name: str, fav_color_hex: str):
    import time as _time
    _conn().execute(
        "UPDATE users SET pass_hash=?, salt=?, fav_dance=?, fav_color_name=?, fav_color_hex=?, updated_at=? "
        "WHERE username=?",
        (pass_hash, salt, fav_dance, fav_color_name, fav_color_hex, _time.time(), username),
    )
    _conn().commit()


def get_user(username: str) -> Optional[Tuple]:
    cur = _conn().cursor()
    cur.execute("SELECT username, pass_hash, salt, fav_dance, fav_color_name, fav_color_hex FROM users WHERE username=?",
                (username,))
    return cur.fetchone()

def touch_user(username: str):
    _conn().execute("UPDATE users SET updated_at=? WHERE username=?", (time.time(), username))
    _conn().commit()

def get_prefs(uid: str) -> Dict[str, str]:
    cur = _conn().cursor()
    cur.execute("SELECT key, value FROM prefs WHERE user_id=?", (uid,))
    return {k: v for k, v in cur.fetchall()}

def set_prefs(uid: str, d: Dict[str, str]):
    cur = _conn().cursor()
    for k, v in d.items():
        cur.execute(
            "INSERT INTO prefs(user_id,key,value) VALUES(?,?,?) "
            "ON CONFLICT(user_id,key) DO UPDATE SET value=excluded.value",
            (uid, k, str(v)),
        )
    _conn().commit()
