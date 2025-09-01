# NOTE: macOS only. Requires Full Disk Access to read ~/Library/Messages/chat.db.
# Sending uses AppleScript to control the Messages app.

import os
import sys
import sqlite3
import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
import re
import textwrap
import json
import urllib.request
import urllib.error

IMSG_DB_DEFAULT = os.path.expanduser("~/Library/Messages/chat.db")
STATE_DEFAULT = os.path.expanduser("~/.reply_state.json")
LEGACY_STATE = os.path.expanduser("~/.imreply_state.json")
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

COMMON_TAPBACKS = ["❤️", "👍", "👎", "😂", "!!", "?"]

FULL_HELP = textwrap.dedent("""
    reply — interactive iMessage/SMS triage CLI for macOS
    =====================================================

    Overview
    --------
    Rank and triage threads that likely need a response, see recent context,
    reply from the terminal (or copy to clipboard), react with emoji tapbacks,
    mark resolved until next inbound, ignore temporarily/forever, refresh without restarting,
    generate OpenAI-assisted drafts, resolve names from Contacts, and set your own
    persistent aliases for handles.

    Quick start
    -----------
      # List likely-to-reply threads with context and names
      python3 reply.py --context 6 --resolve-names

      # Interactive triage
      python3 reply.py --interactive --context 6 --resolve-names --days 14 --limit 40

      # Resume where you left off
      python3 reply.py --interactive --resume

      # No truncation
      python3 reply.py --interactive --no-truncate

      # Export list
      python3 reply.py --export ~/Desktop/needs_reply.json

    Command-line options
    --------------------
      --db PATH                Path to Messages chat.db (default: ~/Library/Messages/chat.db)
      --days N                 Only consider threads active within the last N days (0 = no limit). Default: 30
      --limit N                Max threads to consider (after scoring/sorting). Default: 40
      --include-groups         Include group chats (default: only 1:1)
      --context N              Show last N messages of context. Default: 6
      --resolve-names          Resolve names using the Contacts app
      --names-timeout SECS     Seconds to wait when reading Contacts. Default: 12
      --state PATH             Path to JSON state file. Default: ~/.reply_state.json
      --interactive            Enter interactive triage mode
      --no-truncate            Do not truncate message text in rendering
      --resume                 Start where you left off (uses saved position/GUID in state)
      --llm-model NAME         OpenAI model for drafting (env OPENAI_MODEL also supported). Default: gpt-4o-mini
      --applescript-timeout S  Seconds to wait for AppleScript actions. Default: 20
      --export FILE            Export the (non-interactive) list to .json or .csv and exit
      --help-full              Show this extended help (same as `reply.py help`) and exit

    Interactive keys
    ----------------
      n  next           p  previous         j <#>  jump to item #
      s  skip (session) i  ignore Ndays     f      ignore forever
      z  mark resolved (until next inbound) u      unresolve (clear marker)
      r  reply (send/copy/both)            t      tapback reaction (pick #)
      g  LLM draft (accept/edit/copy/both) a      alias name (persist)
      o  open in Messages                  R      refresh threads from DB
      c  clear ignore on this thread       h      help
      q  quit (saves position & GUID; use --resume to start there next time)

    OpenAI drafting (optional)
    --------------------------
      export OPENAI_API_KEY=sk-...
      export OPENAI_MODEL=gpt-4o-mini   # optional; can also pass --llm-model

      In interactive mode press `g` to generate a short draft using recent context
      plus your notes. You can accept/send, edit, copy, or discard the draft.

    State file
    ----------
      Default: ~/.reply_state.json (override with --state). If not present, the tool
      will transparently read your old ~/.imreply_state.json (legacy) if it exists.
      Fields stored:
        - ignored_forever: [chat GUID, ...]
        - ignored_until: { GUID: ISO8601 timestamp, ... }
        - resolved_until: { GUID: ISO8601 last-incoming we considered resolved, ... }
        - drafts: { GUID: {text, t}, ... }
        - history: [action log]
        - position: last index viewed
        - last_guid: last viewed GUID (used by --resume)
        - overrides: { normalized-handle -> alias name }  # e.g., tel:15551234567, email:user@example.com

    Permissions (macOS)
    -------------------
      • Full Disk Access for reading ~/Library/Messages/chat.db (Terminal/iTerm).
      • Automation permission to control Messages (for sending) and Contacts (for name resolution).
      • SMS relay requires your iPhone with Text Message Forwarding enabled.

    Security & privacy notes
    ------------------------
      • The script never writes to chat.db; sending is done via the Messages app (AppleScript).
      • State is stored locally in JSON. Delete it if you want to reset.
      • If you use LLM drafting, the included context and notes are sent to OpenAI.
        Consider limiting --context or skipping LLM for sensitive threads.
""")

# ---------------------- Utilities ----------------------

def apple_time_to_dt(apple_time: Optional[int]) -> Optional[datetime]:
    if apple_time is None:
        return None
    try:
        t = int(apple_time)
    except Exception:
        return None
    if t <= 0:
        return None
    if t > 10**12:
        delta = timedelta(seconds=t / 1_000_000_000)
    else:
        delta = timedelta(seconds=t)
    return APPLE_EPOCH + delta

def human_timedelta(dt: Optional[datetime]) -> str:
    if dt is None:
        return "unknown"
    now = datetime.now(timezone.utc)
    delta = now - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours = mins // 60
    if hours < 48:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def prompt(msg: str) -> str:
    try:
        return input(msg)
    except EOFError:
        return ""

# ---------------------- Data classes ----------------------

@dataclass
class ThreadInfo:
    chat_id: int
    guid: str
    display_name: Optional[str]
    chat_identifier: Optional[str]
    last_incoming_dt: Optional[datetime]
    last_outgoing_dt: Optional[datetime]
    last_message_dt: Optional[datetime]
    last_incoming_text: Optional[str]
    participants: List[str]
    consecutive_incoming_since_last_outgoing: int
    needs_reply: bool
    score: float

# ---------------------- DB ----------------------

def open_ro(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)

def list_participants(conn: sqlite3.Connection, chat_id: int) -> List[str]:
    cur = conn.execute(
        """
        SELECT h.id
        FROM chat_handle_join chj
        JOIN handle h ON h.ROWID = chj.handle_id
        WHERE chj.chat_id = ?
        ORDER BY h.id
        """,
        (chat_id,),
    )
    return [row[0] for row in cur.fetchall()]

def get_threads_core(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        WITH msgs AS (
            SELECT cmj.chat_id AS chat_id,
                   m.ROWID     AS msg_id,
                   m.is_from_me AS is_from_me,
                   m.date      AS date,
                   m.text      AS text
            FROM chat_message_join cmj
            JOIN message m ON m.ROWID = cmj.message_id
        ),
        per_chat AS (
            SELECT chat_id,
                   MAX(CASE WHEN is_from_me = 0 THEN date END) AS last_incoming,
                   MAX(CASE WHEN is_from_me = 1 THEN date END) AS last_outgoing,
                   MAX(date) AS last_message
            FROM msgs
            GROUP BY chat_id
        ),
        last_in_msg AS (
            SELECT m.chat_id, m.msg_id, m.text, m.date
            FROM msgs m
            JOIN (
                SELECT chat_id, MAX(date) AS maxdate
                FROM msgs
                WHERE is_from_me = 0
                GROUP BY chat_id
            ) w ON w.chat_id = m.chat_id AND w.maxdate = m.date
            WHERE m.is_from_me = 0
        )
        SELECT
            c.ROWID AS chat_id,
            c.guid,
            c.display_name,
            c.chat_identifier,
            pc.last_incoming,
            pc.last_outgoing,
            pc.last_message,
            lim.text AS last_incoming_text,
            lim.date AS last_incoming_date
        FROM chat c
        LEFT JOIN per_chat pc ON pc.chat_id = c.ROWID
        LEFT JOIN last_in_msg lim ON lim.chat_id = c.ROWID
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def count_consecutive_incoming(conn: sqlite3.Connection, chat_id: int, last_outgoing_date: Optional[int]) -> int:
    pivot = 0 if last_outgoing_date is None else int(last_outgoing_date)
    cur = conn.execute(
        """
        SELECT m.is_from_me
        FROM chat_message_join cmj
        JOIN message m ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id = ? AND m.date > ?
        ORDER BY m.date DESC
        """,
        (chat_id, pivot),
    )
    streak = 0
    for (is_from_me,) in cur.fetchall():
        if is_from_me == 0:
            streak += 1
        else:
            break
    return streak

def compute_score(last_incoming_dt, last_outgoing_dt, last_message_dt, last_incoming_text, consecutive_incoming) -> float:
    now = now_utc()
    if last_incoming_dt is None:
        urgency = 0.0
    else:
        hours = (now - last_incoming_dt).total_seconds() / 3600.0
        urgency = 1.0 / (1.0 + (hours / 8.0))
    replied_after = bool(last_incoming_dt and last_outgoing_dt and last_outgoing_dt > last_incoming_dt)
    base = 0.15 if replied_after else 0.65
    streak = min(consecutive_incoming, 5) / 5.0
    expects = 0.0
    txt = (last_incoming_text or "").strip().lower()
    if txt:
        qm = txt.count("?")
        expects += min(qm, 2) * 0.25
        if any(k in txt for k in ["can you","could you","do you","are you","when","what time","free","available","ok?"]):
            expects += 0.25
        if len(txt) < 12 and any(k in txt for k in ["ping","?","u there","you there","yo","hey"]):
            expects += 0.1
        expects = min(expects, 0.8)
    score = base * urgency + 0.25 * streak + 0.35 * expects
    return round(score, 4)

def build_threads(conn: sqlite3.Connection, within_days: int, include_groups: bool) -> List[ThreadInfo]:
    raw = get_threads_core(conn)
    now = now_utc()
    cutoff = now - timedelta(days=within_days) if within_days > 0 else None
    threads: List[ThreadInfo] = []
    for row in raw:
        chat_id = row["chat_id"]
        last_in_dt = apple_time_to_dt(row["last_incoming"])
        last_out_dt = apple_time_to_dt(row["last_outgoing"])
        last_msg_dt = apple_time_to_dt(row["last_message"])
        last_in_text = row["last_incoming_text"]
        parts = list_participants(conn, chat_id)
        if not last_msg_dt:
            continue
        if cutoff and last_msg_dt < cutoff:
            continue
        if not include_groups and len(parts) > 1:
            continue
        needs = bool(last_in_dt and (last_out_dt is None or last_in_dt > last_out_dt))
        streak = count_consecutive_incoming(conn, chat_id, row["last_outgoing"])
        score = compute_score(last_in_dt, last_out_dt, last_msg_dt, last_in_text, streak)
        threads.append(ThreadInfo(
            chat_id=chat_id,
            guid=row["guid"],
            display_name=row["display_name"],
            chat_identifier=row["chat_identifier"],
            last_incoming_dt=last_in_dt,
            last_outgoing_dt=last_out_dt,
            last_message_dt=last_msg_dt,
            last_incoming_text=last_in_text,
            participants=parts,
            consecutive_incoming_since_last_outgoing=streak,
            needs_reply=needs,
            score=score,
        ))
    threads.sort(key=lambda t: (not t.needs_reply, -t.score, t.last_message_dt or APPLE_EPOCH))
    return threads

def fetch_last_messages(conn: sqlite3.Connection, chat_id: int, limit: int) -> List[dict]:
    cur = conn.execute(
        """
        SELECT m.ROWID, m.guid, m.is_from_me, m.date, m.text, m.cache_has_attachments,
               m.associated_message_type, h.id as sender_id
        FROM chat_message_join cmj
        JOIN message m ON m.ROWID = cmj.message_id
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        WHERE cmj.chat_id = ?
        ORDER BY m.date DESC
        LIMIT ?
        """,
        (chat_id, limit),
    )
    msgs: List[dict] = []
    for row in cur.fetchall():
        rowid, guid, is_from_me, date_raw, text, has_att, assoc_type, sender_id = row
        msgs.append(dict(
            rowid=rowid,
            guid=guid,
            when=apple_time_to_dt(date_raw),
            is_from_me=is_from_me,
            sender_id=sender_id,
            text=text,
            has_attachments=has_att or 0,
            assoc_type=assoc_type,
        ))
    msgs.reverse()
    return msgs

# ---------------------- Contacts & Overrides ----------------------

def normalize_phone(s: str) -> str:
    return re.sub(r"\D+", "", s or "")

def key_for_identifier(identifier: Optional[str]) -> Optional[str]:
    if not identifier:
        return None
    s = identifier.strip()
    if not s:
        return None
    if "@" in s:
        return "email:" + s.lower()
    d = normalize_phone(s)
    if d:
        return "tel:" + d
    return "raw:" + s

class NameResolver:
    def __init__(self, enabled: bool = False, timeout: int = 12, overrides: Optional[Dict[str,str]] = None):
        self.enabled = enabled
        self.timeout = timeout
        self.by_email: Dict[str, str] = {}
        self.by_digits: Dict[str, str] = {}
        self.user_overrides: Dict[str, str] = dict(overrides or {})
        if enabled:
            try:
                self._load_contacts()
            except Exception as e:
                print(f"[names] Could not load Contacts: {e}")
                self.enabled = False

    def _load_contacts(self):
        osa = """
        on joinList(L, sep)
          set {T, text item delimiters} to {text item delimiters, sep}
          try
            set s to L as text
          on error
            set s to ""
          end try
          set text item delimiters to T
          return s
        end joinList

        tell application "Contacts"
          set thePeople to people
          set outText to ""
          repeat with p in thePeople
            set personName to (name of p as text)
            try
              set emailVals to value of emails of p
            on error
              set emailVals to { }
            end try
            try
              set phoneVals to value of phones of p
            on error
              set phoneVals to { }
            end try
            set outText to outText & personName & tab & (my joinList(emailVals, ";")) & tab & (my joinList(phoneVals, ";")) & linefeed
          end repeat
          return outText
        end tell
        """
        res = subprocess.run(["osascript", "-e", osa], capture_output=True, text=True, timeout=self.timeout)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or "osascript error")
        for line in res.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].replace("\t", " ").replace("\n", " ").strip()
            emails = [e.strip().lower() for e in parts[1].split(";") if e.strip()]
            phones = [p.strip() for p in parts[2].split(";") if p.strip()]
            for e in emails:
                self.by_email[e] = name
            for p in phones:
                d = normalize_phone(p)
                if not d:
                    continue
                self.by_digits[d] = name
                if len(d) >= 7:
                    self.by_digits[d[-7:]] = name

    def override_lookup(self, identifier: Optional[str]) -> Optional[str]:
        k = key_for_identifier(identifier)
        if not k:
            return None
        if k in self.user_overrides:
            return self.user_overrides[k]
        if k.startswith("tel:"):
            d = k[4:]
            if d.startswith("1") and len(d) > 10 and ("tel:" + d[1:]) in self.user_overrides:
                return self.user_overrides["tel:" + d[1:]]
            if len(d) >= 7 and ("tel:" + d[-7:]) in self.user_overrides:
                return self.user_overrides["tel:" + d[-7:]]
        return None

    def add_override(self, identifier: str, name: str):
        k = key_for_identifier(identifier)
        if not k:
            return
        self.user_overrides[k] = name

    def resolve(self, identifier: Optional[str]) -> Optional[str]:
        ov = self.override_lookup(identifier)
        if ov:
            return ov
        if not self.enabled:
            return None
        if not identifier:
            return None
        s = identifier.strip()
        if "@" in s:
            return self.by_email.get(s.lower())
        d = normalize_phone(s)
        if not d:
            return None
        if d in self.by_digits:
            return self.by_digits[d]
        d_alt = d.lstrip("0")
        if d_alt.startswith("1") and len(d_alt) > 10 and d_alt[1:] in self.by_digits:
            return self.by_digits[d_alt[1:]]
        if d_alt in self.by_digits:
            return self.by_digits[d_alt]
        if len(d) >= 7 and d[-7:] in self.by_digits:
            return self.by_digits[d[-7:]]
        return None

# ---------------------- Messaging & Clipboard ----------------------

def send_via_messages(chat_guid: str, text: str, timeout: int = 20) -> Tuple[bool, str]:
    osa = """
    on run argv
      set chatID to item 1 of argv
      set msgText to item 2 of argv
      tell application "Messages"
        if it is not running then launch
        try
          set theChat to chat id chatID
          send msgText to theChat
          return "OK"
        on error errMsg number errNum
          return "ERR: " & errNum & " " & errMsg
        end try
      end tell
    end run
    """
    try:
        res = subprocess.run(["osascript", "-e", osa, "--", chat_guid, text], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "osascript not found (must run on macOS)."
    except subprocess.TimeoutExpired:
        return False, "Timed out asking Messages to send."
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if res.returncode == 0 and out == "OK":
        return True, "Sent."
    if out.startswith("ERR:"):
        return False, out
    return False, (err or out or f"osascript exited with code {res.returncode}.")

def tapback_via_messages(chat_guid: str, message_guid: str, emoji: str, timeout: int = 20) -> Tuple[bool, str]:
    osa = """
    on run argv
      set chatID to item 1 of argv
      set msgID to item 2 of argv
      set tbEmoji to item 3 of argv
      tell application "Messages"
        if it is not running then launch
        try
          set theChat to chat id chatID
          set theMsg to message id msgID of theChat
          set tapback of theMsg to tbEmoji
          return "OK"
        on error errMsg number errNum
          return "ERR: " & errNum & " " & errMsg
        end try
      end tell
    end run
    """
    try:
        res = subprocess.run(["osascript", "-e", osa, "--", chat_guid, message_guid, emoji], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "osascript not found (must run on macOS)."
    except subprocess.TimeoutExpired:
        return False, "Timed out asking Messages to react."
    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()
    if res.returncode == 0 and out == "OK":
        return True, "Reacted."
    if out.startswith("ERR:"):
        return False, out
    return False, (err or out or f"osascript exited with code {res.returncode}.")

def copy_to_clipboard(text: str) -> Tuple[bool, str]:
    try:
        res = subprocess.run(["pbcopy"], input=text, text=True, capture_output=True)
    except FileNotFoundError:
        return False, "pbcopy not found (must run on macOS)."
    if res.returncode == 0:
        return True, "Copied to clipboard."
    return False, (res.stderr or "pbcopy error")

# ---------------------- LLM draft (OpenAI only) ----------------------

def call_openai(messages: List[Dict[str, str]], model: str, api_key: Optional[str]) -> str:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = "https://api.openai.com/v1/chat/completions"
    body = {"model": model, "messages": messages, "temperature": 0.4, "top_p": 1.0}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        raise RuntimeError(f"OpenAI HTTP {e.code}: {err}")
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError("Unexpected OpenAI response format")

def llm_draft_reply(context_msgs: List[dict], notes: str, my_name_hint: Optional[str], model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    sys_prompt = (
        "You're helping craft a short, clear SMS/iMessage reply in the user's tone.\n"
        "Constraints:\n"
        " - Keep it brief (<= 2 sentences unless more is necessary).\n"
        " - Be concrete; answer questions directly.\n"
        " - Respect the user's intent in NOTES.\n"
        " - Do not include placeholders; output only the reply text.\n"
    )
    if my_name_hint:
        sys_prompt += f" The user's name is {my_name_hint}.\n"
    convo = []
    convo.append({"role": "system", "content": sys_prompt})
    transcript = []
    for m in context_msgs[-12:]:
        who = "You" if m["is_from_me"] == 1 else "Them"
        text = (m["text"] or "[non-text]").replace("\n", " ")
        transcript.append(f"{who}: {text}")
    convo.append({"role": "user", "content": "RECENT MESSAGES:\n" + "\n".join(transcript)})
    if notes:
        convo.append({"role": "user", "content": f"NOTES/GUIDANCE:\n{notes}"})
    draft = call_openai(convo, model=model, api_key=api_key)
    return draft

# ---------------------- State ----------------------

def load_state(path: str) -> Dict[str, Any]:
    p = os.path.expanduser(path)
    # Legacy fallback only when using default state path
    if not os.path.exists(p) and os.path.abspath(p) == os.path.abspath(STATE_DEFAULT) and os.path.exists(LEGACY_STATE):
        p = LEGACY_STATE
    if not os.path.exists(p):
        return {"ignored_forever": [], "ignored_until": {}, "resolved_until": {}, "drafts": {}, "history": [], "position": 0, "last_guid": None, "overrides": {}}
    try:
        with open(p, "r", encoding="utf-8") as f:
            st = json.load(f)
        st.setdefault("ignored_forever", [])
        st.setdefault("ignored_until", {})
        st.setdefault("resolved_until", {})
        st.setdefault("drafts", {})
        st.setdefault("history", [])
        st.setdefault("position", 0)
        st.setdefault("last_guid", None)
        st.setdefault("overrides", {})
        return st
    except Exception:
        return {"ignored_forever": [], "ignored_until": {}, "resolved_until": {}, "drafts": {}, "history": [], "position": 0, "last_guid": None, "overrides": {}}

def save_state(st: Dict[str, Any], path: str):
    p = os.path.expanduser(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def is_ignored(guid: str, st: Dict[str, Any]) -> bool:
    if guid in st.get("ignored_forever", []):
        return True
    until_str = st.get("ignored_until", {}).get(guid)
    if not until_str:
        return False
    try:
        until = datetime.fromisoformat(until_str.replace("Z","+00:00"))
    except Exception:
        return False
    return now_utc() < until

def is_resolved_for_now(thread: ThreadInfo, st: Dict[str, Any]) -> bool:
    """Return True if the thread was marked resolved and has NO newer incoming since then."""
    ru_str = st.get("resolved_until", {}).get(thread.guid)
    if not ru_str:
        return False
    try:
        ru = datetime.fromisoformat(ru_str.replace("Z", "+00:00"))
    except Exception:
        return False
    # If there's no incoming at all, treat as not resolved (surface it if heuristic says so)
    if not thread.last_incoming_dt:
        return False
    # Hide only if the last incoming we see is not newer than the resolved marker
    return thread.last_incoming_dt <= ru

def set_ignore_days(guid: str, days: int, st: Dict[str, Any], state_path: str):
    until = now_utc() + timedelta(days=days)
    st.setdefault("ignored_until", {})[guid] = until.isoformat()
    st["history"].append({"t": now_utc().isoformat(), "guid": guid, "action": "ignore_days", "days": days})
    save_state(st, state_path)

def set_ignore_forever(guid: str, st: Dict[str, Any], state_path: str):
    if guid not in st.setdefault("ignored_forever", []):
        st["ignored_forever"].append(guid)
    st["history"].append({"t": now_utc().isoformat(), "guid": guid, "action": "ignore_forever"})
    save_state(st, state_path)

def clear_ignore(guid: str, st: Dict[str, Any], state_path: str):
    st.get("ignored_until", {}).pop(guid, None)
    if guid in st.get("ignored_forever", []):
        st["ignored_forever"] = [g for g in st["ignored_forever"] if g != guid]
    st["history"].append({"t": now_utc().isoformat(), "guid": guid, "action": "clear_ignore"})
    save_state(st, state_path)

def record_draft(guid: str, text: str, st: Dict[str, Any], state_path: str):
    st.setdefault("drafts", {})[guid] = {"text": text, "t": now_utc().isoformat()}
    st["history"].append({"t": now_utc().isoformat(), "guid": guid, "action": "draft"})
    save_state(st, state_path)

def set_position(idx: int, guid: Optional[str], st: Dict[str, Any], state_path: str):
    st["position"] = idx
    if guid is not None:
        st["last_guid"] = guid
    save_state(st, state_path)

def set_override(identifier: str, name: str, st: Dict[str, Any], state_path: str):
    k = key_for_identifier(identifier)
    if not k:
        return
    st.setdefault("overrides", {})[k] = name
    st["history"].append({"t": now_utc().isoformat(), "action": "override_set", "handle": identifier, "key": k, "name": name})
    save_state(st, state_path)

def set_resolved_until(guid: str, dt: datetime, st: Dict[str, Any], state_path: str):
    st.setdefault("resolved_until", {})[guid] = (dt or now_utc()).isoformat()
    st["history"].append({"t": now_utc().isoformat(), "guid": guid, "action": "resolved_until_set", "until": st["resolved_until"][guid]})
    save_state(st, state_path)

def clear_resolved(guid: str, st: Dict[str, Any], state_path: str):
    st.get("resolved_until", {}).pop(guid, None)
    st["history"].append({"t": now_utc().isoformat(), "guid": guid, "action": "resolved_until_clear"})
    save_state(st, state_path)

# ---------------------- Presentation ----------------------

def build_display_name(thread: ThreadInfo, resolver: 'NameResolver') -> Tuple[str, str]:
    parts = thread.participants
    if thread.display_name:
        name = thread.display_name
    else:
        if len(parts) == 1:
            person = resolver.resolve(parts[0])
            name = person or parts[0]
        else:
            resolved = []
            for p in parts[:3]:
                rp = resolver.resolve(p)
                resolved.append(rp or p)
            name = ", ".join(resolved) + ("…" if len(parts) > 3 else "")
    summary = []
    for p in parts[:3]:
        rp = resolver.resolve(p)
        summary.append(rp or p)
    part_str = ", ".join(summary) + ("…" if len(parts) > 3 else "")
    return name, part_str

def format_context(msgs: List[dict], resolver: 'NameResolver', no_truncate: bool = False) -> List[str]:
    out = []
    for m in msgs:
        ts = human_timedelta(m["when"])
        who = "You" if m["is_from_me"] == 1 else (resolver.resolve(m["sender_id"]) or (m["sender_id"] or "Other"))
        text = (m["text"] or "").replace("\n", " ").strip()
        if not text:
            if m["assoc_type"] and m["assoc_type"] != 0:
                text = "[reaction]"
            elif m["has_attachments"]:
                text = "[attachment]"
            else:
                text = "[message]"
        if not no_truncate and len(text) > 140:
            text = text[:137] + "…"
        out.append(f"      {ts:>6}  {who}: {text}")
    return out

def render_thread(idx: int, total: int, t: ThreadInfo, resolver: 'NameResolver', conn: sqlite3.Connection, context_n: int, no_truncate: bool):
    name, parts_short = build_display_name(t, resolver)
    last_r = human_timedelta(t.last_message_dt)
    last_in = human_timedelta(t.last_incoming_dt)
    need = "YES" if t.needs_reply else "no"
    header = f"[{idx+1}/{total}] {name}  [{parts_short}]"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    preview = (t.last_incoming_text or "").replace("\n", " ").strip()
    if preview and (not no_truncate) and len(preview) > 100:
        preview = preview[:97] + "…"
    if t.last_incoming_text:
        print(f"Last incoming: “{preview}”")
    print(f"Last msg: {last_r} ago | Last incoming: {last_in} ago | Needs reply: {need} | Score: {t.score:.2f} | Streak: {t.consecutive_incoming_since_last_outgoing}")
    if context_n > 0:
        msgs = fetch_last_messages(conn, t.chat_id, context_n)
        lines = format_context(msgs, resolver, no_truncate=no_truncate)
        if lines:
            print("Context (oldest → newest):")
            for line in lines:
                print(line)

def open_in_messages(chat_guid: str) -> Tuple[bool, str]:
    osa = """
    on run argv
      set chatID to item 1 of argv
      tell application "Messages"
        if it is not running then launch
        try
          set theChat to chat id chatID
          activate
          return "OK"
        on error errMsg number errNum
          return "ERR: " & errNum & " " & errMsg
        end try
      end tell
    end run
    """
    try:
        res = subprocess.run(["osascript", "-e", osa, "--", chat_guid], capture_output=True, text=True, timeout=15)
    except Exception as e:
        return False, str(e)
    out = (res.stdout or "").strip()
    if res.returncode == 0 and out == "OK":
        return True, "Opened."
    return False, out

def read_multiline_from_editor(initial_text: str = "") -> Optional[str]:
    editor = os.environ.get("EDITOR") or "nano"
    with tempfile.NamedTemporaryFile(prefix="reply_", suffix=".txt", delete=False) as tf:
        path = tf.name
        tf.write(initial_text.encode("utf-8"))
    try:
        subprocess.run([editor, path])
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

# ---------------------- Interactive loop ----------------------

def interactive_loop(threads: List[ThreadInfo], conn: sqlite3.Connection, resolver: NameResolver, st: Dict[str, Any], state_path: str, context_n: int, llm_model: str, applescript_timeout: int, within_days: int, include_groups: bool, limit: int, no_truncate: bool, resume: bool):
    def build_active(ths: List[ThreadInfo]) -> List[ThreadInfo]:
        return [t for t in ths if not is_ignored(t.guid, st) and not is_resolved_for_now(t, st)]

    active = build_active(threads)
    if not active:
        print("No threads to triage (everything is ignored/resolved or filtered).")
        return

    idx = 0
    if resume:
        last_guid = st.get("last_guid")
        if last_guid:
            for i, tt in enumerate(active):
                if tt.guid == last_guid:
                    idx = i
                    break
        else:
            pos = st.get("position", 0)
            if 0 <= pos < len(active):
                idx = pos
        print(f"(resuming at {idx+1}/{len(active)})")

    session_skips = set()

    def find_next(i: int, step: int) -> int:
        j = i + step
        if j < 0: j = 0
        if j >= len(active): j = len(active) - 1
        return j

    while True:
        t = active[idx]
        if t.guid in session_skips:
            nxt = idx + 1
            while nxt < len(active) and active[nxt].guid in session_skips:
                nxt += 1
            if nxt >= len(active):
                nxt = 0
                while nxt < len(active) and active[nxt].guid in session_skips:
                    nxt += 1
                if nxt >= len(active):
                    print("All remaining threads skipped for this session.")
                    break
            idx = nxt
            continue

        render_thread(idx, len(active), t, resolver, conn, context_n, no_truncate=no_truncate)
        print("\nCommands: [n]ext  [p]rev  [j]ump#  [s]kip  [i]gnore Ndays  [f]orever  [z] resolve  [u]nresolve  [r]eply  [t]apback  [g]enerate  [a]lias  [o]pen  [R]efresh  [c]lear ignore  [h]elp  [q]uit")
        cmd = prompt("> ").strip()

        if cmd == "q":
            set_position(idx, active[idx].guid if 0 <= idx < len(active) else None, st, state_path)
            print("Bye.")
            break

        elif cmd in ("h", "?"):
            print("n: next | p: previous | j 5: jump to #5 | s: skip (session only)")
            print("i: ignore for N days | f: ignore forever | c: clear ignore for this thread")
            print("z: mark resolved (hide until a NEW incoming) | u: clear resolved marker")
            print("r: reply (send/copy/both, with inline or $EDITOR) | t: tapback reaction (pick #) | g: LLM draft (accept/edit/copy/both) | o: open in Messages")
            print("a: alias a participant handle to a custom name (persisted)")
            print("R: refresh threads from DB with current filters | q: quit")

        elif cmd.startswith("j"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                j = int(parts[1]) - 1
                if 0 <= j < len(active):
                    idx = j
                else:
                    print(f"Out of range 1..{len(active)}")
            else:
                print("Usage: j 7")

        elif cmd == "n":
            idx = find_next(idx, +1)
        elif cmd == "p":
            idx = find_next(idx, -1)

        elif cmd == "R":
            cur_guid = active[idx].guid if active else None
            new_threads = build_threads(conn, within_days=within_days, include_groups=include_groups)
            if limit and limit > 0:
                new_threads = new_threads[:limit]
            active = build_active(new_threads)
            if not active:
                print("After refresh, no threads to triage (ignored/resolved or filtered).")
                break
            if cur_guid:
                try:
                    idx = next(i for i, tt in enumerate(active) if tt.guid == cur_guid)
                except StopIteration:
                    idx = 0
            else:
                idx = 0
            print(f"Refreshed. Now {len(active)} active threads.")

        elif cmd == "s":
            session_skips.add(t.guid)
            idx = find_next(idx, +1)

        elif cmd.startswith("i"):
            parts = cmd.split()
            if len(parts) == 2 and parts[1].isdigit():
                days = int(parts[1])
            else:
                ans = prompt("Ignore for how many days? [default 2] ").strip()
                days = int(ans) if ans.isdigit() else 2
            set_ignore_days(t.guid, days, st, state_path)
            print(f"Ignored for {days} day(s).")
            active = [x for x in active if x.guid != t.guid]
            if not active:
                print("No more threads.")
                break
            if idx >= len(active): idx = len(active) - 1

        elif cmd == "f":
            set_ignore_forever(t.guid, st, state_path)
            print("Ignored forever.")
            active = [x for x in active if x.guid != t.guid]
            if not active:
                print("No more threads.")
                break
            if idx >= len(active): idx = len(active) - 1

        elif cmd == "c":
            clear_ignore(t.guid, st, state_path)
            print("Cleared ignore flags for this thread.")

        elif cmd == "o":
            ok, detail = open_in_messages(t.guid)
            print(detail)

        elif cmd == "a":
            parts = t.participants
            if not parts:
                print("No participants found to alias.")
                continue
            if len(parts) == 1:
                handle = parts[0]
                current = resolver.resolve(handle) or handle
                print(f"Current name for {handle}: {current}")
                new = prompt("New name (leave empty to cancel): ").strip()
                if not new:
                    print("Canceled.")
                    continue
                set_override(handle, new, st, state_path)
                resolver.add_override(handle, new)
                print(f"Saved alias: {handle} → {new}")
            else:
                print("Participants:")
                for i, h in enumerate(parts, start=1):
                    show = resolver.resolve(h) or h
                    print(f"  {i}. {show} [{h}]")
                pick = prompt(f"Choose 1..{len(parts)} (empty cancels): ").strip()
                if not pick or not pick.isdigit():
                    print("Canceled.")
                    continue
                k = int(pick)
                if not (1 <= k <= len(parts)):
                    print("Out of range.")
                    continue
                handle = parts[k-1]
                current = resolver.resolve(handle) or handle
                print(f"Current name for {handle}: {current}")
                new = prompt("New name (leave empty to cancel): ").strip()
                if not new:
                    print("Canceled.")
                    continue
                set_override(handle, new, st, state_path)
                resolver.add_override(handle, new)
                print(f"Saved alias: {handle} → {new}")
            render_thread(idx, len(active), t, resolver, conn, context_n, no_truncate=no_truncate)

        elif cmd == "z":
            # Mark resolved until next inbound
            mark_dt = t.last_incoming_dt or t.last_message_dt or now_utc()
            set_resolved_until(t.guid, mark_dt, st, state_path)
            print("Marked resolved (will re-appear on next incoming).")
            # Remove from active now
            active = [x for x in active if x.guid != t.guid]
            if not active:
                print("No more threads.")
                break
            if idx >= len(active): idx = len(active) - 1

        elif cmd == "u":
            clear_resolved(t.guid, st, state_path)
            print("Cleared resolved marker for this thread.")

        elif cmd == "r":
            msg = prompt("Reply text (leave empty to open $EDITOR): ")
            if not msg:
                msg = read_multiline_from_editor("")
            if not msg:
                print("No message; canceled.")
                continue
            print(f"Reply: {msg!r}")
            mode = prompt("Choose: (s)end  (c)opy  (b)oth  (n)cancel > ").strip().lower()
            if mode not in ("s","c","b"):
                print("Canceled.")
                continue
            did_send = False
            did_copy = False
            if mode in ("c","b"):
                ok, detail = copy_to_clipboard(msg)
                print("📋 " + (detail if ok else f"Copy failed: {detail}"))
                did_copy = ok
            if mode in ("s","b"):
                ok, detail = send_via_messages(t.guid, msg, timeout=applescript_timeout)
                print("✉️  " + ("Sent." if ok else f"Not sent: {detail}"))
                did_send = ok
            if did_send or did_copy:
                st["history"].append({"t": now_utc().isoformat(), "guid": t.guid, "action": "reply", "sent": did_send, "copied": did_copy, "chars": len(msg)})
                save_state(st, state_path)
                idx = find_next(idx, +1)

        elif cmd == "t":
            msgs = fetch_last_messages(conn, t.chat_id, max(12, context_n or 6))
            lines = format_context(msgs, resolver, no_truncate=no_truncate)
            for i, line in enumerate(lines, 1):
                print(f"{i:2d}. {line.strip()}")
            pick = prompt("React to which message #? ").strip()
            if not pick.isdigit() or not (1 <= int(pick) <= len(msgs)):
                print("Canceled.")
                continue
            target = msgs[int(pick) - 1]
            print("Pick tapback:")
            for i, tb in enumerate(COMMON_TAPBACKS, 1):
                print(f"{i}. {tb}")
            epick = prompt("Tapback #: ").strip()
            if not epick.isdigit() or not (1 <= int(epick) <= len(COMMON_TAPBACKS)):
                print("Canceled.")
                continue
            emoji = COMMON_TAPBACKS[int(epick) - 1]
            ok, detail = tapback_via_messages(t.guid, target["guid"], emoji, timeout=applescript_timeout)
            print(f"{emoji} " + ("Reacted." if ok else f"Not reacted: {detail}"))
            if ok:
                st["history"].append({"t": now_utc().isoformat(), "guid": t.guid, "action": "tapback", "emoji": emoji, "msg_guid": target["guid"]})
                save_state(st, state_path)
                idx = find_next(idx, +1)

        elif cmd == "g":
            notes = prompt("Notes/guidance for LLM (optional): ")
            try:
                ctx = fetch_last_messages(conn, t.chat_id, max(12, context_n or 6))
                draft = llm_draft_reply(ctx, notes, my_name_hint=None, model=llm_model)
                print("\n--- DRAFT ---")
                print(draft)
                print("-------------")
                record_draft(t.guid, draft, st, state_path)
                sub = prompt("(a)ccept & send  (e)dit then send  (c)opy  (b)oth send+copy  (d)iscard  > ").strip().lower()
                if sub == "a":
                    ok, detail = send_via_messages(t.guid, draft, timeout=applescript_timeout)
                    print("✉️  " + ("Sent." if ok else f"Not sent: {detail}"))
                    if ok:
                        st["history"].append({"t": now_utc().isoformat(), "guid": t.guid, "action": "reply", "sent": True, "copied": False, "chars": len(draft)})
                        save_state(st, state_path)
                        idx = find_next(idx, +1)
                elif sub == "e":
                    edited = read_multiline_from_editor(draft)
                    if edited:
                        mode = prompt("Choose: (s)end  (c)opy  (b)oth  (n)cancel > ").strip().lower()
                        if mode in ("c","b"):
                            ok, detail = copy_to_clipboard(edited)
                            print("📋 " + (detail if ok else f"Copy failed: {detail}"))
                            copied = ok
                        else:
                            copied = False
                        if mode in ("s","b"):
                            ok, detail = send_via_messages(t.guid, edited, timeout=applescript_timeout)
                            print("✉️  " + ("Sent." if ok else f"Not sent: {detail}"))
                            sent = ok
                        else:
                            sent = False
                        if sent or copied:
                            st["history"].append({"t": now_utc().isoformat(), "guid": t.guid, "action": "reply", "sent": sent, "copied": copied, "chars": len(edited)})
                            save_state(st, state_path)
                            if sent:
                                idx = find_next(idx, +1)
                    else:
                        print("No text; canceled.")
                elif sub == "c":
                    ok, detail = copy_to_clipboard(draft)
                    print("📋 " + (detail if ok else f"Copy failed: {detail}"))
                    if ok:
                        st["history"].append({"t": now_utc().isoformat(), "guid": t.guid, "action": "reply", "sent": False, "copied": True, "chars": len(draft)})
                        save_state(st, state_path)
                elif sub == "b":
                    okc, detc = copy_to_clipboard(draft)
                    print("📋 " + (detc if okc else f"Copy failed: {detc}"))
                    oks, dets = send_via_messages(t.guid, draft, timeout=applescript_timeout)
                    print("✉️  " + ("Sent." if oks else f"Not sent: {dets}"))
                    if okc or oks:
                        st["history"].append({"t": now_utc().isoformat(), "guid": t.guid, "action": "reply", "sent": oks, "copied": okc, "chars": len(draft)})
                        save_state(st, state_path)
                        if oks:
                            idx = find_next(idx, +1)
                else:
                    print("Draft discarded.")
            except Exception as e:
                print(f"(LLM error) {e}")

        else:
            print("Unknown command. Type 'h' for help.")

# ---------------------- Non-interactive listing ----------------------

def print_threads(conn: sqlite3.Connection, threads: List[ThreadInfo], limit: int, context_n: int, resolver: NameResolver, no_truncate: bool, st: Dict[str, Any]):
    # Filter out ignored/resolved
    filtered = [t for t in threads if not is_ignored(t.guid, st) and not is_resolved_for_now(t, st)]
    shown = filtered[:limit] if limit else filtered
    print("\nThreads that likely need a reply (showing {} of {} after filters):".format(len(shown), len(filtered)))
    print("-" * 100)
    for idx, t in enumerate(shown, start=1):
        name, parts_short = build_display_name(t, resolver)
        last_r = human_timedelta(t.last_message_dt)
        last_in = human_timedelta(t.last_incoming_dt)
        need = "YES" if t.needs_reply else "no"

        preview = (t.last_incoming_text or "").replace("\n", " ").strip()
        if preview and (not no_truncate) and len(preview) > 80:
            preview = preview[:77] + "…"

        print(f"{idx:>2}. {name}  [{parts_short}]")
        print(f"    last msg: {last_r} ago   last incoming: {last_in} ago   needs reply: {need}   score: {t.score:.2f}   streak: {t.consecutive_incoming_since_last_outgoing}")
        if t.last_incoming_text:
            print(f"    last incoming: “{preview}”")

        if context_n > 0:
            try:
                msgs = fetch_last_messages(conn, t.chat_id, context_n)
                lines = format_context(msgs, resolver, no_truncate=no_truncate)
                if lines:
                    print("    context: (oldest → newest)")
                    for line in lines:
                        print(line)
            except Exception as e:
                print(f"    (context error: {e})")
        print("-" * 100)

def export_threads(threads: List[ThreadInfo], path: str, resolver: NameResolver):
    ext = os.path.splitext(path)[1].lower()
    rows = []
    for t in threads:
        display, _ = build_display_name(t, resolver)
        rows.append({
            "chat_id": t.chat_id,
            "guid": t.guid,
            "display_name": t.display_name,
            "resolved_display": display,
            "chat_identifier": t.chat_identifier,
            "participants": t.participants,
            "last_incoming_text": t.last_incoming_text,
            "last_incoming_iso": t.last_incoming_dt.isoformat() if t.last_incoming_dt else None,
            "last_outgoing_iso": t.last_outgoing_dt.isoformat() if t.last_outgoing_dt else None,
            "last_message_iso": t.last_message_dt.isoformat() if t.last_message_dt else None,
            "consecutive_incoming_since_last_outgoing": t.consecutive_incoming_since_last_outgoing,
            "needs_reply": t.needs_reply,
            "score": t.score,
        })
    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    elif ext == ".csv":
        import csv
        keys = list(rows[0].keys()) if rows else []
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        raise ValueError("Unsupported export extension. Use .json or .csv")
    print(f"Exported {len(rows)} rows to {path}")

# ---------------------- Main ----------------------

def main():
    # Pre-parse: `help` subcommand prints extended help and exits.
    if len(sys.argv) > 1 and sys.argv[1] == "help":
        print(FULL_HELP)
        sys.exit(0)

    ap = argparse.ArgumentParser(
        description="Interactive iMessage/SMS triage CLI for macOS: triage; mark resolved; ignore; reply; tapback; LLM; refresh; no-truncate; resume; aliases; copy replies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Tip: run `python3 reply.py --help-full` or `python3 reply.py help` to see the full guide."
    )
    ap.add_argument("--db", default=IMSG_DB_DEFAULT, help="Path to chat.db (default: %(default)s)")
    ap.add_argument("--days", type=int, default=30, help="Only consider threads active within the last N days (0 = no limit). Default: %(default)s")
    ap.add_argument("--limit", type=int, default=40, help="Max threads to consider. Default: %(default)s")
    ap.add_argument("--include-groups", action="store_true", help="Include group chats (default: only 1:1).")
    ap.add_argument("--context", type=int, default=6, help="Show last N messages for context. Default: %(default)s")
    ap.add_argument("--resolve-names", action="store_true", help="Resolve via Contacts.app")
    ap.add_argument("--names-timeout", type=int, default=12, help="Seconds to wait for Contacts. Default: %(default)s")
    ap.add_argument("--state", default=STATE_DEFAULT, help="Path to JSON state file. Default: %(default)s (legacy fallback to ~/.imreply_state.json)")
    ap.add_argument("--interactive", action="store_true", help="Enter interactive triage mode.")
    ap.add_argument("--no-truncate", action="store_true", help="Do not truncate messages in rendering.")
    ap.add_argument("--resume", action="store_true", help="Start where you left off (uses saved state).")
    ap.add_argument("--help-full", action="store_true", help="Show extended help with examples and interactive keys, then exit.")

    # LLM options (OpenAI only)
    default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    ap.add_argument("--llm-model", default=default_model, help="OpenAI model name for drafting (also via env OPENAI_MODEL). Default: %(default)s")
    ap.add_argument("--applescript-timeout", type=int, default=20, help="Seconds to wait for AppleScript actions.")

    # Non-interactive extras
    ap.add_argument("--export", help="Export list to a file (.json or .csv) and exit.")

    args = ap.parse_args()

    if args.help_full:
        print(FULL_HELP)
        sys.exit(0)

    db_path = os.path.expanduser(args.db)
    if not os.path.exists(db_path):
        print(f"Messages database not found at: {db_path}")
        print("Tip: Ensure you’re on macOS and point --db to your chat.db file.")
        sys.exit(1)

    try:
        conn = open_ro(db_path)
    except sqlite3.OperationalError as e:
        print("Could not open database (read-only).")
        print(f"Error: {e}")
        print("If this is a permissions issue, add your terminal to Full Disk Access and try again.")
        sys.exit(1)

    st = load_state(args.state)
    resolver = NameResolver(enabled=args.resolve_names, timeout=args.names_timeout, overrides=st.get("overrides", {}))

    try:
        threads = build_threads(conn, within_days=args.days, include_groups=args.include_groups)
        if args.limit and args.limit > 0:
            threads = threads[: args.limit]

        if args.export:
            print_threads(conn, threads, limit=len(threads), context_n=args.context, resolver=resolver, no_truncate=args.no_truncate, st=st)
            export_threads(threads, os.path.expanduser(args.export), resolver)
            return

        if args.interactive:
            interactive_loop(
                threads, conn, resolver, st, args.state,
                context_n=args.context,
                llm_model=args.llm_model,
                applescript_timeout=args.applescript_timeout,
                within_days=args.days,
                include_groups=args.include_groups,
                limit=args.limit,
                no_truncate=args.no_truncate,
                resume=args.resume
            )
        else:
            print_threads(conn, threads, limit=len(threads), context_n=args.context, resolver=resolver, no_truncate=args.no_truncate, st=st)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
