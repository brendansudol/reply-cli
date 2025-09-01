# reply — interactive iMessage/SMS triage CLI for macOS

`reply.py` helps you **triage and reply to iMessage/SMS threads** from your Mac, right in the terminal. It ranks threads that likely need a response, shows recent context, lets you **reply**, **copy replies** to the clipboard, **mark resolved until next inbound**, **ignore** threads temporarily/forever, **refresh** without restarting, and generate **OpenAI‑assisted drafts** with optional notes. It also supports **persistent aliases** for phone numbers/emails so you can display the names you want.

> **macOS only.** Reading Messages requires Full Disk Access; sending uses AppleScript to control the Messages app.

---

## Table of contents

- Features
- How it works
- Installation
- Permissions (macOS)
- Quick start
- Command‑line options
- Interactive mode
- OpenAI draft replies (optional)
- State file
- Built‑in help from the CLI
- FAQ / Troubleshooting
- Security & privacy notes
- License

---

## Features

- **Prioritized triage** — finds threads where the latest message is incoming and you haven’t replied; weights recency, consecutive incoming streak, and “question/ask” cues to compute a score.
- **Interactive, one‑by‑one workflow** — step through threads with navigation, ignore/skip, reply, and LLM draft.
- **Context view** — show the last _N_ messages (oldest → newest). Use `--no-truncate` to see full text.
- **Reply from the CLI** — send via Messages (AppleScript) or copy to the clipboard, or do both.
- **Emoji tapbacks** — choose common reactions (❤️ 👍 👎 😂 !! ?) on specific messages; counts as a response.
- **Mark resolved (until next inbound)** — press `z` to hide a thread that doesn’t need a reply; it will **re-appear automatically** when a new incoming message arrives. Use `u` to clear the marker.
- **OpenAI draft (optional)** — send context + your notes to OpenAI to produce a short reply you can accept, edit, copy, or discard.
- **Refresh without restarting** — press `R` to rebuild the list from the DB with current filters.
- **Contact name resolution** — optionally pull names from the Contacts app.
- **Custom aliases** — press `a` to associate your own name with a handle (phone/email). Aliases override Contacts and persist in the state file.
- **Ignore state** — ignore a thread for N days or forever; clear later if needed.
- **Resume where you left off** — pass `--resume` to begin at your last viewed thread.
- **Export** — dump the current list to JSON or CSV for analysis.
- **Extended help** — `python3 reply.py help` or `--help-full` shows a comprehensive guide (this README distilled).

---

## How it works

- **Read‑only DB scan** — opens `~/Library/Messages/chat.db` **read‑only** (SQLite URI `?mode=ro`) and computes per‑thread metadata:
  - last incoming/outgoing time, last message time,
  - last incoming text preview,
  - participants,
  - consecutive incoming streak since your last outgoing.
- **Scoring heuristic** — combines:
  - urgency (time since last incoming),
  - whether you have replied after the last incoming,
  - consecutive incoming streak (capped at 5),
  - ask/question cues in the last incoming (e.g., “can you”, “what time”, “?”, “free”, etc.).
- **Mark resolved (until next inbound)** — if a thread’s latest message doesn’t need a reply, press `z`. We save the latest incoming timestamp as a **resolved marker**. While no newer incoming exists, the thread stays hidden. If a **new incoming** arrives later, the thread returns to the triage list. Clear with `u` anytime.
- **Sending** — replies are sent by telling the **Messages** app to `send "<text>" to chat id "<GUID>"` via AppleScript (`osascript`). **We do not write to the DB** to send.
- **Contacts** — name resolution uses the Contacts app via AppleScript (optional).
- **Aliases** — your overrides (e.g., `tel:15551234567 → “Mom”`) are stored in the state file and always take precedence.

---

## Installation

1. Ensure you’re on macOS with Python 3.9+ available as `python3`.
2. Save `reply.py` somewhere in your PATH (or run it from the repo directory).
3. Make it executable (optional):
   ```bash
   chmod +x reply.py
   ```

No extra pip packages are required (uses only the standard library).

---

## Permissions (macOS)

You’ll likely get prompts the first time you run certain features:

- **Full Disk Access** — required to read `~/Library/Messages/chat.db`.
  - System Settings → **Privacy & Security** → **Full Disk Access** → add and enable your Terminal/iTerm.
- **Automation** (AppleScript) — for controlling **Messages** (sending) and **Contacts** (name resolution).
  - macOS will prompt when first used.
- **SMS relay** — to send SMS (green bubbles) from the Mac, your iPhone must be reachable with **Text Message Forwarding** enabled.

---

## Quick start

### List likely‑to‑reply threads with context and names

```bash
python3 reply.py --context 6 --resolve-names
```

### Enter interactive triage

```bash
python3 reply.py --interactive --context 6 --resolve-names --days 14 --limit 40
```

### Resume where you left off

```bash
python3 reply.py --interactive --resume
```

### Show full message text (no truncation)

```bash
python3 reply.py --interactive --no-truncate
```

### Export list to JSON/CSV

```bash
python3 reply.py --export ~/Desktop/needs_reply.json
python3 reply.py --export ~/Desktop/needs_reply.csv
```

---

## Command‑line options

```
--db PATH                Path to Messages chat.db (default: ~/Library/Messages/chat.db)
--days N                 Only consider threads active within the last N days (0 = no limit). Default: 30
--limit N                Max threads to consider (after scoring/sorting). Default: 40
--include-groups         Include group chats (default: only 1:1)
--context N              Show last N messages of context. Default: 6
--resolve-names          Resolve names using the Contacts app
--names-timeout SECS     Seconds to wait when reading Contacts. Default: 12
--state PATH             Path to JSON state file. Default: ~/.reply_state.json (legacy fallback to ~/.imreply_state.json)
--interactive            Enter interactive triage mode
--no-truncate            Do not truncate message text in rendering
--resume                 Start where you left off (uses saved position/GUID in state)
--llm-model NAME         OpenAI model for drafting (env OPENAI_MODEL also supported). Default: gpt-4o-mini
--applescript-timeout S  Seconds to wait for AppleScript actions. Default: 20
--export FILE            Export the (non-interactive) list to .json or .csv and exit
--help-full              Show extended help (same as `reply.py help`) and exit
```

Environment variables (optional):

- `OPENAI_API_KEY` — required for OpenAI drafting.
- `OPENAI_MODEL` — default model if you don’t pass `--llm-model` (e.g., `gpt-4o-mini`).

---

## Interactive mode

Start interactive mode:

```bash
python3 reply.py --interactive --context 6
```

You’ll see one thread at a time:

- Header with name/participants, timing, needs‑reply flag, and score.
- Optional **Context** list (oldest → newest). Non‑text messages show as `[attachment]` or `[reaction]`.
- If you pass `--no-truncate`, all message text is shown in full.

### Keys / commands

```
n  next           p  previous         j <#>  jump to item #
s  skip (session) i  ignore Ndays     f      ignore forever
z  mark resolved (until next inbound) u      unresolve (clear marker)
r  reply (send/copy/both)            t      tapback reaction (pick #)
g  LLM draft (accept/edit/copy/both) a      alias name (persist)
o  open in Messages                  R      refresh threads from DB
c  clear ignore on this thread       h      help
q  quit (saves position & GUID; use --resume next time)
```

Notes:

- **Resolve `z`** — saves a _resolved marker_ at the latest incoming. The thread is hidden until someone sends another message; then it reappears automatically. Use `u` to clear.
- **Reply `r`** — type your message. If you submit an empty line, your `$EDITOR` (defaults to `nano`) opens for multi‑line. Choose: `(s)end`, `(c)opy`, `(b)oth`, or cancel.
- **Tapback `t`** — react to a specific message with a common tapback (❤️ 👍 👎 😂 !! ?). Counts as a response.
- **Draft `g`** — add optional notes, then choose: `(a)ccept & send`, `(e)dit then send`, `(c)opy`, `(b)oth`, or `d`iscard.
- **Alias `a`** — set a custom name for a participant handle (works in 1:1 and group chats). Aliases are immediate and persist.
- **Refresh `R`** — rebuilds the list from the DB with your current filters; attempts to keep you on the same thread.
- **Ignore `i` / `f`** — hide a thread for N days or forever (persisted). Use `c` to clear ignore later.
- **Skip `s`** — session‑only; thread reappears on next run.

---

## OpenAI draft replies (optional)

```bash
export OPENAI_API_KEY=sk-...
# optional override of the default:
export OPENAI_MODEL=gpt-4o-mini
```

Then in interactive mode press `g` to generate a draft using the last few messages plus your notes. You can accept/send, edit, copy, or discard the draft.

> **Privacy:** Context you send to OpenAI leaves your machine. Consider reducing `--context`, removing sensitive content, or skipping LLM entirely for sensitive threads.

---

## State file

Default path: `~/.reply_state.json` (change with `--state`). Contains:

- `ignored_forever` — list of chat GUIDs
- `ignored_until` — GUID → ISO timestamp
- `resolved_until` — GUID → ISO timestamp of last incoming considered resolved
- `drafts` — last draft per GUID (text + timestamp)
- `history` — append‑only action log
- `position` — last index viewed
- `last_guid` — GUID of last viewed thread (used by `--resume`)
- `overrides` — your **aliases**, keyed by normalized handle (e.g., `tel:15551234567`, `email:user@example.com`)

You can safely delete the state file to reset ignores/aliases/resume position.

---

## Built‑in help from the CLI

- Standard help: `python3 reply.py -h`
- Extended guide (this README distilled):
  - `python3 reply.py --help-full`
  - `python3 reply.py help`

---

## FAQ / Troubleshooting

**I see “Operation not permitted” or empty results.**  
Add your Terminal/iTerm to **Full Disk Access** and restart it.

**Sending fails.**

- Approve the **Automation** prompt allowing Terminal to control **Messages**.
- SMS requires **Text Message Forwarding** from your iPhone to your Mac.
- Ensure Messages is configured for the right account.

**`pbcopy` not found.**  
You’re likely not on macOS. Clipboard copy requires macOS.

**OpenAI errors (401 / timeouts).**  
Set a valid `OPENAI_API_KEY`. Check network connectivity.

**Names don’t resolve.**

- Use `--resolve-names` to read Contacts.
- Or set your own names with `a` (aliases apply even without Contacts).

**Schema changes in macOS?**  
This tool reads common columns from `chat.db`. If Apple changes the schema, queries may need adjusting.

---

## Security & privacy notes

- **Read‑only DB access:** the script never writes to `chat.db`.
- **Sending:** AppleScript asks Messages to send; it doesn’t modify the database directly.
- **State file:** stores ignores, aliases, resolved markers, drafts, and a small history. It stays on your machine.
- **OpenAI drafts:** if you use drafting, the included context and your notes are sent to OpenAI. Consider limiting `--context` or skipping drafting for sensitive threads.

---

## License

MIT
