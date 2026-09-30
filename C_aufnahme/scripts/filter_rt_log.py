"""Retrieval layer over a rt_logs/*.txt training-PC console dump.

/rt-check writes its expectation FIRST, then asks this script for the few
original log lines that bear on it. The caller reads that evidence -- never
the raw log. Output size is capped and does NOT grow with the log: a
50 000-line dump yields the same handful of lines as a 150-line one.

Usage:
    python scripts/filter_rt_log.py <log> --head
    python scripts/filter_rt_log.py <log> --expect "..." --expect "..."
    python scripts/filter_rt_log.py <log>
    python scripts/filter_rt_log.py <log> --full [--max-lines N]
    python scripts/filter_rt_log.py <log> --series "Mean reward" [--series ...] [--points N]
    python scripts/filter_rt_log.py <log> --events "[autodr]"
    python scripts/filter_rt_log.py --self-test

--head    only timestamp, cmd and git of the first run block. That is the
          duplicate key /rt-check needs, with no log text at all.
--series  a TREND: the value of one iteration-block label ("Mean reward",
          "dr/bounds_version", ...) over the whole run, sampled to --points
          rows (default 12) plus min, max and the non-finite count. Blocks are
          counted in LOG ORDER, because the stop-rule block loop repeats an
          iteration index at every block boundary (D-037). Added 2026-09-11:
          --expect returns only the first matches, so a 1500-iteration run
          showed its first blocks and never its end.
--events  lines that START with a prefix: total count, count per key (the
          token before the first ':') and per outcome (the word after '->'),
          plus the first and last line.
--expect  one occurrence per numbered expectation point, verbatim. The
          script pulls search terms out of THAT TEXT ONLY -- never out of
          the log -- and returns the original lines that hit them.
--full    the old whole-digest mode. For a human reading by hand; agents use
          the evidence mode.

Evidence output, every section capped:

    RUN 1: <ts> | git <hash> | exit <code> | cmd: <cmd>
    === BEFUND ===          mechanical facts only, no interpretation
    === STATUS ===          exit-code lines, abort point, unclosed block
    === ANOMALIEN (n) ===   Traceback / FAIL / Error / *** PENDING|MISSING
    === MARKER ===          SCRIPT_MARKER lines
    === E1: <expect text> ===   the lines that hit that expectation
    === ENDE: n Evidenzzeilen von m Logzeilen ===

Numbers are matched with tolerance, not as text: an expectation of 0.09641
has to find 0.09640874557648531, and 0.0 has to find 3.7e-18. Plain
substring search would miss exactly the evidence that decides the run.

Every content line keeps its 1-based source line number as "L<nr>: " and its
byte-exact original text. Nothing is summarised, nothing is shortened --
not even long paths, because an expectation may be about a path.

Stdlib only: runs under a bare laptop Python, no Isaac, no torch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_EVIDENCE_LINES = 60
MAX_PER_EXPECT = 6
MAX_WEAK_PER_EXPECT = 3
MAX_ANOMALY_LINES = 8
MAX_MARKER_LINES = 4
MAX_EXPECTS = 12

# --series / --events
SERIES_POINTS = 12
MAX_SERIES = 16
MAX_EVENT_KEYS = 24
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")

# --full mode only
MAX_JSON_LINES = 40
MAX_BLOCK_LINES = 150
TAIL_LINES = 80

# Own-script tag at line start. Isaac/Kit/omni noise never matches: it writes
# [INFO], [Warning] (capitalized) or a timestamp, never a bare lowercase tag.
OWN_SCRIPT_RE = re.compile(r"^\[[a-z][a-z0-9_]*\]")
# Isaac/Kit/omni boot noise: capitalized tags, ISO timestamps, GPU table rows.
NOISE_RE = re.compile(r"^\[(INFO|Info|Warning|Error)\]|^\d{4}-\d{2}-\d{2}T|^\|")
# Bounds both divider styles: the startup report's (72x '=') and rt_log.ps1's (62x '=').
DIVIDER_RE = re.compile(r"^=+\s*$")
REPORT_MARKER = "INSERTION STARTUP REPORT"
ANOMALY_RE = re.compile(r"FAIL|Traceback|\*\*\*\s+(PENDING|MISSING)|Error|error:")
MARKER_RE = re.compile(r"\bmarker:")

RT_TS_RE = re.compile(r"^\[rt_log\] (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s")
RT_CMD_RE = re.compile(r"^\[rt_log\] cmd: (.*)$")
RT_GIT_RE = re.compile(r"^\[rt_log\] git: (.*)$")
RT_EXIT_RE = re.compile(r"^\[rt_log\] exit code: (.*)$")
# Quoted JSON strings are blanked before counting braces, so a brace inside a
# path or a message cannot unbalance the block.
STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
QUOTED_RE = re.compile(r'"([^"]+)"')
# Words that carry no retrieval value. German first, then English.
STOPWORDS = {
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "und", "oder", "mit", "ohne", "nach", "vor", "auf", "aus", "bei", "von",
    "ist", "sind", "war", "wird", "werden", "soll", "sollen", "muss", "muessen",
    "nicht", "kein", "keine", "sich", "sein", "seine", "dass", "wenn", "dann",
    "auch", "noch", "nur", "alle", "allen", "aller", "jede", "jeder", "jedes",
    "zwei", "drei", "vier", "genau", "danach", "davor", "damit", "dabei",
    "the", "and", "for", "with", "without", "that", "this", "then", "must",
    "should", "will", "shall", "are", "was", "not", "all", "any", "each",
}
NUM_RTOL = 1e-3
NUM_ATOL = 1e-9
STRONG_SIG_DIGITS = 3
STRONG_WORD_LEN = 4


# --------------------------------------------------------------------------
# reading the file
# --------------------------------------------------------------------------


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"error: log file not found: {path}")
    if path.is_dir():
        sys.exit(f"error: not a file: {path}")
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def brace_delta(line: str) -> int:
    s = STR_RE.sub('""', line)
    return s.count("{") + s.count("[") - s.count("}") - s.count("]")


def find_runs(lines: list[str]) -> list[dict]:
    """One entry per [rt_log] header block, in file order."""
    runs: list[dict] = []
    for line in lines:
        m = RT_TS_RE.match(line)
        if m:
            runs.append({"ts": m.group(1), "cmd": None, "git": None, "exit": None})
            continue
        if not runs:
            continue
        for key, rx in (("cmd", RT_CMD_RE), ("git", RT_GIT_RE), ("exit", RT_EXIT_RE)):
            m = rx.match(line)
            if m and runs[-1][key] is None:
                runs[-1][key] = m.group(1).strip()
    return runs


def run_lines(runs: list[dict]) -> list[str]:
    if not runs:
        return ["RUN: kein [rt_log]-Kopf in der Datei -- Herkunft des Textes unbekannt."]
    out = []
    for n, r in enumerate(runs, 1):
        code = r["exit"]
        exit_txt = f"exit {code}" if code is not None else "exit FEHLT (Lauf abgebrochen)"
        git_txt = f"git {r['git']}" if r["git"] else "git ?"
        out.append(f"RUN {n}: {r['ts']} | {git_txt} | {exit_txt} | cmd: {r['cmd'] or '?'}")
    return out


def own_script_blocks(lines: list[str]) -> list[list[int]]:
    """Each own-script line plus the continuation lines that belong to it."""
    blocks: list[list[int]] = []
    n = len(lines)
    i = 0
    while i < n:
        if not OWN_SCRIPT_RE.match(lines[i]):
            i += 1
            continue
        block = [i]
        depth = brace_delta(lines[i])
        j = i + 1
        while j < n:
            line = lines[j]
            if OWN_SCRIPT_RE.match(line):
                break
            if depth > 0:
                block.append(j)
                depth += brace_delta(line)
                j += 1
                continue
            if line.strip() == "":
                break
            # an unindented '{' or '[' right after the tag line opens the block
            if line[:1].isspace() or line.strip() in ("{", "["):
                block.append(j)
                depth += brace_delta(line)
                j += 1
                continue
            break
        blocks.append(block)
        i = j
    return blocks


def unclosed_block_start(lines: list[str]) -> int | None:
    """Index of the first own-script block whose JSON never closes."""
    for block in own_script_blocks(lines):
        if sum(brace_delta(lines[i]) for i in block) > 0:
            return block[0]
    return None


# --------------------------------------------------------------------------
# expectation terms -- derived from the --expect text ONLY, never from the log
# --------------------------------------------------------------------------


def sig_digits(token: str) -> int:
    digits = re.sub(r"[^0-9]", "", token.split("e")[0].split("E")[0])
    return len(digits.lstrip("0"))


def parse_terms(text: str) -> list[tuple[str, object, bool]]:
    """-> [(kind, value, strong)] with kind in {'str', 'num', 'word'}."""
    terms: list[tuple[str, object, bool]] = []
    seen: set[tuple[str, str]] = set()
    rest = text
    for quoted in QUOTED_RE.findall(text):
        key = ("str", quoted)
        if key not in seen:
            seen.add(key)
            terms.append(("str", quoted, True))
        rest = rest.replace(f'"{quoted}"', " ")
    for token in NUM_RE.findall(rest):
        try:
            value = float(token)
        except ValueError:
            continue
        key = ("num", repr(value))
        if key in seen:
            continue
        seen.add(key)
        terms.append(("num", value, sig_digits(token) >= STRONG_SIG_DIGITS))
    for word in WORD_RE.findall(rest):
        low = word.lower()
        if low in STOPWORDS:
            continue
        key = ("word", low)
        if key in seen:
            continue
        seen.add(key)
        terms.append(("word", low, len(low) >= STRONG_WORD_LEN))
    return terms


def line_numbers(line: str) -> list[float]:
    out = []
    for token in NUM_RE.findall(line):
        try:
            out.append(float(token))
        except ValueError:
            pass
    return out


def num_hit(values: list[float], wanted: float) -> bool:
    for value in values:
        if wanted == 0.0:
            if abs(value) <= NUM_ATOL:
                return True
        elif abs(value - wanted) <= NUM_RTOL * abs(wanted):
            return True
    return False


def score_line(line: str, terms, values: list[float]) -> tuple[int, int]:
    """-> (strong hits, weak hits) for one log line against one expectation."""
    low = line.lower()
    strong = weak = 0
    for kind, value, is_strong in terms:
        if kind == "str":
            ok = value in line
        elif kind == "num":
            ok = num_hit(values, value)
        else:
            ok = value in low
        if not ok:
            continue
        if is_strong:
            strong += 1
        else:
            weak += 1
    return strong, weak


def rank_hits(lines, searchable, terms, per_expect):
    """-> (indices, weak_only). Strong hits win; weak hits are the fallback."""
    strong_hits, weak_hits = [], []
    for i in searchable:
        strong, weak = score_line(lines[i], terms, line_numbers(lines[i]))
        if strong:
            strong_hits.append((strong, weak, i))
        elif weak:
            weak_hits.append((0, weak, i))
    if strong_hits:
        strong_hits.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return sorted(i for _, _, i in strong_hits[:per_expect]), False
    weak_hits.sort(key=lambda t: (-t[1], t[2]))
    return sorted(i for _, _, i in weak_hits[:MAX_WEAK_PER_EXPECT]), True


# --------------------------------------------------------------------------
# evidence mode
# --------------------------------------------------------------------------


def build_evidence(lines: list[str], expects: list[str], per_expect: int):
    total = len(lines)
    runs = find_runs(lines)
    if total == 0:
        return ["RUN: (leere Datei)", "=== BEFUND ===", "- Datei ist leer"], 0

    searchable = [
        i for i, line in enumerate(lines) if line.strip() and not NOISE_RE.match(line)
    ]
    own_idx = [i for i in range(total) if OWN_SCRIPT_RE.match(lines[i])]
    exit_idx = [i for i in range(total) if RT_EXIT_RE.match(lines[i])]
    unclosed = unclosed_block_start(lines)
    anomalies = [i for i in searchable if ANOMALY_RE.search(lines[i])]
    markers = [i for i in searchable if MARKER_RE.search(lines[i])]

    hits = [rank_hits(lines, searchable, parse_terms(e), per_expect) for e in expects]

    # ---- BEFUND: only facts the script can prove, no interpretation ----
    befund: list[str] = []
    if not runs:
        befund.append("- kein [rt_log]-Kopf, Herkunft des Textes unbekannt")
    for n, r in enumerate(runs, 1):
        tag = f"Lauf {n}: " if len(runs) > 1 else ""
        if r["exit"] is None:
            befund.append(f"- {tag}exit FEHLT, Lauf abgebrochen")
        else:
            befund.append(f"- {tag}exit {r['exit']}")
    if unclosed is not None:
        befund.append(f"- Lauf endet mitten in einem Block (ab L{unclosed + 1})")
    befund.append(f"- {len(anomalies)} Anomalie(n)" if anomalies else "- keine Anomalie")
    for n, (idxs, _) in enumerate(hits, 1):
        if not idxs:
            befund.append(f"- Erwartung {n} ohne Beleg")
    if not expects:
        befund.append("- keine Erwartung uebergeben (--expect fehlt)")

    out = list(run_lines(runs))
    out.append("=== BEFUND ===")
    out.extend(befund)

    printed: set[int] = set()
    content = 0

    def put(section_lines: list[int], mark: str = "") -> None:
        nonlocal content
        for i in section_lines:
            out.append(f"L{i + 1}: {lines[i]}{mark}")
            printed.add(i)
            content += 1

    status = list(exit_idx)
    if not exit_idx and own_idx:
        status.append(own_idx[-1])
    if unclosed is not None and unclosed not in status:
        status.append(unclosed)
    if status:
        out.append("=== STATUS ===")
        put(sorted(set(status)))

    if anomalies:
        extra = len(anomalies) - MAX_ANOMALY_LINES
        note = f", {extra} weitere unterdrueckt" if extra > 0 else ""
        out.append(f"=== ANOMALIEN ({len(anomalies)} Zeilen{note}) ===")
        put([i for i in anomalies[:MAX_ANOMALY_LINES] if i not in printed])

    fresh_markers = [i for i in markers[:MAX_MARKER_LINES] if i not in printed]
    if fresh_markers:
        out.append("=== MARKER ===")
        put(fresh_markers)

    for n, (expect, (idxs, weak_only)) in enumerate(zip(expects, hits), 1):
        out.append(f"=== E{n}: {expect} ===")
        if not idxs:
            out.append("(keine Zeile gefunden)")
            continue
        mark = "   (schwacher Treffer)" if weak_only else ""
        for i in idxs:
            if i in printed:
                out.append(f"L{i + 1} (siehe oben)")
            else:
                put([i], mark)

    out.append(f"=== ENDE: {content} Evidenzzeilen von {total} Logzeilen ===")
    return out, content


def evidence(lines: list[str], expects: list[str]) -> list[str]:
    """Build the evidence, shrinking per-expectation depth if the global cap
    would be blown. Every cap that bites is reported, never silent."""
    if len(expects) > MAX_EXPECTS:
        dropped = len(expects) - MAX_EXPECTS
        expects = expects[:MAX_EXPECTS]
        head = [f"HINWEIS: {dropped} Erwartung(en) ueber dem Limit {MAX_EXPECTS} ignoriert."]
    else:
        head = []
    for per_expect in (MAX_PER_EXPECT, 3, 2, 1):
        out, content = build_evidence(lines, expects, per_expect)
        if content <= MAX_EVIDENCE_LINES:
            if per_expect < MAX_PER_EXPECT:
                head = head + [
                    f"HINWEIS: Deckel {MAX_EVIDENCE_LINES} Zeilen -- "
                    f"je Erwartung nur {per_expect} Belegzeilen."
                ]
            return head + out
    return head + out


# --------------------------------------------------------------------------
# --full mode: the whole digest, for a human reading by hand
# --------------------------------------------------------------------------


def find_report_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """0-based, end-exclusive (start, end) for each STARTUP REPORT block."""
    blocks: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        if REPORT_MARKER not in line:
            continue
        start = i - 1 if i > 0 and DIVIDER_RE.match(lines[i - 1]) else i
        limit = min(len(lines), start + MAX_BLOCK_LINES)
        end = limit
        for j in range(i + 1, limit):
            if DIVIDER_RE.match(lines[j]):
                end = j + 1
                break
        blocks.append((start, end))
    return blocks


def emit(out: list[str], lines: list[str], indices, printed: set[int]) -> None:
    for i in indices:
        out.append(f"L{i + 1}: {lines[i]}")
        printed.add(i)


def emit_capped(out: list[str], lines: list[str], idxs: list[int], printed: set[int]) -> None:
    if len(idxs) <= MAX_JSON_LINES + 4:
        emit(out, lines, idxs, printed)
        return
    head = MAX_JSON_LINES // 2
    tail = MAX_JSON_LINES - head
    emit(out, lines, idxs[:head], printed)
    out.append(f"    ... {len(idxs) - head - tail} Zeilen ausgelassen ...")
    emit(out, lines, idxs[-tail:], printed)


def build_digest(lines: list[str]) -> tuple[list[str], list[str]]:
    total = len(lines)
    if total == 0:
        return ["RUN: (leere Datei)"], ["(leere Datei)"]
    runs = run_lines(find_runs(lines))
    body: list[str] = []
    printed: set[int] = set()

    blocks = own_script_blocks(lines)
    if blocks:
        count = sum(len(b) for b in blocks)
        body.append(f"=== OWN-SCRIPT OUTPUT ({len(blocks)} Bloecke, {count} Zeilen) ===")
        for block in blocks:
            emit_capped(body, lines, block, printed)
        body.append("")

    for number, (start, end) in enumerate(find_report_blocks(lines), 1):
        if all(i in printed for i in range(start, end)):
            continue
        body.append(f"=== STARTUP REPORT #{number} (Zeilen {start + 1}-{end}) ===")
        emit(body, lines, range(start, end), printed)
        body.append("")

    anomalies = [
        i
        for i, line in enumerate(lines)
        if i not in printed and not NOISE_RE.match(line) and ANOMALY_RE.search(line)
    ]
    if anomalies:
        body.append(f"=== ANOMALIES ({len(anomalies)} Zeilen) ===")
        emit(body, lines, anomalies[:40], printed)
        body.append("")

    if not printed:
        tail_start = max(0, total - TAIL_LINES)
        body.append(f"=== TAIL FALLBACK (letzte {total - tail_start} von {total} Zeilen) ===")
        emit(body, lines, range(tail_start, total), printed)
        body.append("")

    body.append(f"=== ENDE Filter: {len(printed)} von {total} Zeilen gezeigt ===")
    return runs, body


def render_full(lines: list[str], max_lines: int) -> list[str]:
    runs, body = build_digest(lines)
    total = len(runs) + len(body)
    if max_lines > 0 and total > max_lines:
        return runs + [f"OVERFLOW: Digest hat {total} Zeilen (> {max_lines})."]
    return runs + [""] + body


def iteration_blocks(lines: list[str]) -> list[tuple[int, int, int]]:
    """``(line_index, iteration, total)`` for every rsl_rl iteration block, in log order."""
    out = []
    for i, line in enumerate(lines):
        m = ITER_RE.search(line)
        if m:
            out.append((i, int(m.group(1)), int(m.group(2))))
    return out


def series(lines: list[str], label: str) -> list[tuple[int, int, int, str]]:
    """``(block_no, iteration, line_index, value_text)`` for each block carrying ``label``.

    The label must match the text before the colon EXACTLY (after stripping), so
    ``Mean episode success_rate`` does not also collect
    ``Mean episode success_rate_cumulative``. The first hit inside a block wins.
    """
    pat = re.compile(r"^\s*" + re.escape(label.strip()) + r":\s+(\S+)\s*$")
    blocks = iteration_blocks(lines)
    out = []
    for b, (start, it, _total) in enumerate(blocks):
        end = blocks[b + 1][0] if b + 1 < len(blocks) else len(lines)
        for j in range(start + 1, end):
            m = pat.match(lines[j])
            if m:
                out.append((b + 1, it, j, m.group(1)))
                break
    return out


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def sample_positions(n: int, points: int) -> list[int]:
    """Evenly spaced positions in ``range(n)``, always including the first and the last."""
    if n <= 0:
        return []
    if points <= 1 or n == 1:
        return [n - 1]
    return sorted({round(k * (n - 1) / (points - 1)) for k in range(points)})


def render_series(lines: list[str], labels: list[str], points: int) -> list[str]:
    out: list[str] = []
    blocks = iteration_blocks(lines)
    out.append(f"ITERATIONSBLOECKE: {len(blocks)}"
               + (f", Index {blocks[0][1]}..{blocks[-1][1]}/{blocks[-1][2]}" if blocks else ""))
    for label in labels[:MAX_SERIES]:
        rows = series(lines, label)
        if not rows:
            out.append(f'=== SERIE "{label}": (kein Block traegt dieses Label) ===')
            continue
        vals = [(r, _as_float(r[3])) for r in rows]
        finite = [(r, v) for r, v in vals if v is not None and v == v and v not in (float("inf"), float("-inf"))]
        bad = len(rows) - len(finite)
        out.append(f'=== SERIE "{label}": {len(rows)} Bloecke mit Wert, {bad} nicht endlich ===')
        if finite:
            lo = min(finite, key=lambda t: t[1])[0]
            hi = max(finite, key=lambda t: t[1])[0]
            out.append(f"  min {lo[3]} (Block {lo[0]}, it {lo[1]}, L{lo[2] + 1})"
                       f" | max {hi[3]} (Block {hi[0]}, it {hi[1]}, L{hi[2] + 1})")
        for p in sample_positions(len(rows), points):
            b, it, j, v = rows[p]
            out.append(f"  Block {b:>5d} it {it:>5d} L{j + 1}: {v}")
    if len(labels) > MAX_SERIES:
        out.append(f"({len(labels) - MAX_SERIES} weitere --series ignoriert, Deckel {MAX_SERIES})")
    return out


def render_events(lines: list[str], prefixes: list[str]) -> list[str]:
    out: list[str] = []
    blocks = iteration_blocks(lines)
    for prefix in prefixes:
        hits = [(i, l) for i, l in enumerate(lines) if l.startswith(prefix)]
        out.append(f'=== EREIGNIS "{prefix}": {len(hits)} Zeilen ===')
        if not hits:
            continue
        keys: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for _, l in hits:
            rest = l[len(prefix):].strip()
            key = rest.split(":", 1)[0].strip() if ":" in rest else "?"
            keys[key] = keys.get(key, 0) + 1
            if "->" in rest:
                tail = rest.split("->", 1)[1].split()
                kind = tail[0] if tail else "?"
                kinds[kind] = kinds.get(kind, 0) + 1
        shown = sorted(keys.items())[:MAX_EVENT_KEYS]
        out.append("  je Schluessel: " + ", ".join(f"{k} {v}" for k, v in shown)
                   + (f" (+{len(keys) - MAX_EVENT_KEYS} weitere)" if len(keys) > MAX_EVENT_KEYS else ""))
        if kinds:
            out.append("  je Ausgang: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))

        def after_it(idx: int) -> str:
            prev = [it for (li, it, _t) in blocks if li < idx]
            return f"nach it {prev[-1]}" if prev else "vor dem ersten Block"

        for tag, (i, l) in (("erste", hits[0]), ("letzte", hits[-1])):
            out.append(f"  {tag}: L{i + 1} ({after_it(i)}): {l}")
    return out


def print_head(lines: list[str]) -> None:
    runs = find_runs(lines)
    if not runs:
        print("HEAD: kein [rt_log]-Kopf gefunden -- der Text kam nicht aus dem Wrapper.")
        return
    r = runs[0]
    print(r["ts"])
    print(f"cmd: {r['cmd'] or '?'}")
    print(f"git: {r['git'] or '?'}")
    if len(runs) > 1:
        print(f"({len(runs)} Laeufe in der Datei; das ist der erste)")


# --------------------------------------------------------------------------
# self-test: the regression cases /rt-check depends on. Stdlib, no file system.
# --------------------------------------------------------------------------

HEADER = [
    "==============================================================",
    "[rt_log] 2026-08-24 14:17:55  cwd: C:\\x",
    "[rt_log] cmd: python -u scripts/fix_stage_units.py --usd probe.usd",
    "[rt_log] git: 61a8b89",
    "==============================================================",
]
BOOT = [
    "[INFO][AppLauncher]: Using device: cuda:0",
    "2026-08-24T12:17:56Z [345ms] [Warning] [carb] Error in plugin",
    "| GPU | NVIDIA GeForce RTX 3080 | Error |",
]


def _self_test() -> int:
    failures: list[str] = []
    ran: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        ran.append(name)
        if ok:
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name}{(' -- ' + detail) if detail else ''}")
            failures.append(name)

    def ev(log, expects):
        return "\n".join(evidence(log, expects))

    good = HEADER + BOOT + [
        "[fix_stage_units] marker: fix_stage_units-2026-08-24a",
        "[fix_stage_units] AFTER:",
        "{",
        '  "meters_per_unit": 1.0,',
        '  "extents_m": [0.09640874557648531, 0.14350000681588987, 0.15200399261616893],',
        '  "z_min": 3.7e-18,',
        '  "verdict": {"pass": true}',
        "}",
        "[rt_log] exit code: 0",
    ]

    # --- A: PASS, every expectation backed by at least one line ---
    out = ev(good, [
        "meters_per_unit ist 1.0",
        "extents 0.09641 0.14350 0.15200",
        "marker fix_stage_units-2026-08-24a",
    ])
    check("A: E1 belegt", "meters_per_unit" in out.split("=== E1")[1].split("=== E2")[0])
    check("A: E2 belegt", "extents_m" in out.split("=== E2")[1].split("=== E3")[0])
    check("A: E3 belegt", "marker" in out.split("=== E3")[1])
    check("A: kein 'ohne Beleg'", "ohne Beleg" not in out)
    check("A: exit 0 im BEFUND", "- exit 0" in out)

    # --- B: FAIL, the refuting line is delivered ---
    bad = [l.replace("1.0,", "0.001,") if '"meters_per_unit"' in l else l for l in good]
    out = ev(bad, ["meters_per_unit ist 1.0"])
    check("B: widerlegende Zeile da", '"meters_per_unit": 0.001,' in out)
    check("B: mit L<nr>", re.search(r"L\d+: +\"meters_per_unit\": 0\.001,", out) is not None)

    # --- C: UNKLAR, the value is simply absent and nothing is substituted ---
    out = ev(good, ['die Zeile "contract PASS" erscheint'])
    e1 = out.split("=== E1")[1]
    check("C: keine Zeile gefunden", "(keine Zeile gefunden)" in e1)
    check("C: kein Ersatzwert", "meters_per_unit" not in e1)
    check("C: BEFUND meldet es", "- Erwartung 1 ohne Beleg" in out)

    # --- Zahlentoleranz: text search would miss all three of these ---
    out = ev(good, ["extents 0.09641"])
    check("Toleranz 0.09641 -> 0.096408745", "0.09640874557648531" in out)
    out = ev(good, ["z_min ist 0.0"])
    check("Toleranz 0.0 -> 3.7e-18", "3.7e-18" in out)

    # --- exit != 0 ---
    fail_run = good[:-1] + ["[fix_stage_units] REPLACE FAILED: OSError", "[rt_log] exit code: 2"]
    out = ev(fail_run, [])
    check("exit 2 in RUN", "exit 2 |" in out.splitlines()[0])
    check("exit 2 im BEFUND", "- exit 2" in out)
    check("exit 2 in STATUS", re.search(r"L\d+: \[rt_log\] exit code: 2", out) is not None)

    # --- exit-Zeile fehlt ---
    aborted = HEADER + BOOT + ["[fix_stage_units] resolved input: probe.usd"]
    out = ev(aborted, [])
    check("exit FEHLT gemeldet", "exit FEHLT" in out)
    check("Abbruchstelle geliefert", "resolved input" in out)

    # --- Traceback, und kein Isaac-Rauschen ---
    crash = HEADER + BOOT + ["Traceback (most recent call last):", "  File x, line 1",
                             "RuntimeError: boom", "[rt_log] exit code: 1"]
    out = ev(crash, [])
    check("Traceback in ANOMALIEN", "Traceback (most recent call last):" in out)
    check("Exception-Zeile dabei", "RuntimeError: boom" in out)
    check("Isaac [Error] draussen", "Error in plugin" not in out)
    check("GPU-Tabelle draussen", "| GPU |" not in out)

    # --- offener Block ---
    open_block = HEADER + ["[fix_stage_units] AFTER:", "{", '  "meters_per_unit": 1.0,']
    out = ev(open_block, [])
    check("offener Block gemeldet", "mitten in einem Block" in out)

    # --- mehrere Befehle in einer Datei ---
    two = good + HEADER + ["[fix_stage_units] second run", "[rt_log] exit code: 3"]
    out = ev(two, [])
    check("zwei RUN-Zeilen", out.count("\nRUN 2:") == 1 or out.startswith("RUN 1:") and "RUN 2:" in out)
    check("beide exits im BEFUND", "- Lauf 1: exit 0" in out and "- Lauf 2: exit 3" in out)

    # --- grosser Log: Groesse haengt nicht an der Log-Laenge ---
    big = HEADER + [f"2026-01-01T00:00:0{i%10}Z [Warning] [carb] chatter {i}" for i in range(3000)]
    big += [f"[zero_agent] step {i} reward 0.5" for i in range(300)]
    big += ["Traceback (most recent call last):", "RuntimeError: boom", "[rt_log] exit code: 1"]
    out_lines = evidence(big, ["reward soll ueber 0.9 liegen", "300 steps gelaufen"])
    body = [l for l in out_lines if l.startswith("L")]
    check("grosser Log: <= 60 Evidenzzeilen", len(body) <= MAX_EVIDENCE_LINES, str(len(body)))
    check("grosser Log: <= 40 Ausgabezeilen", len(out_lines) <= 40, str(len(out_lines)))
    check("grosser Log: Traceback trotzdem da", any("Traceback" in l for l in out_lines))

    # --- head liefert den Doppelt-Schluessel ---
    check("head: Zeitstempel", find_runs(good)[0]["ts"] == "2026-08-24 14:17:55")
    check("head: git", find_runs(good)[0]["git"] == "61a8b89")

    # --- Begriffe kommen nur aus dem --expect-Text ---
    terms = parse_terms('meters_per_unit ist 1.0 und "contract PASS"')
    kinds = {(k, v) for k, v, _ in terms}
    check("Terme: Zitat erkannt", ("str", "contract PASS") in kinds)
    check("Terme: Zahl erkannt", ("num", 1.0) in kinds)
    check("Terme: Stoppwort raus", ("word", "und") not in kinds)
    check("Terme: 1.0 ist schwach", all(not s for k, v, s in terms if k == "num" and v == 1.0))

    # --- --series: trend over the whole run, in LOG ORDER (2026-09-11) ---
    def block(it, total, reward, extra=None):
        rows = [f"Learning iteration {it}/{total}", f"   Mean reward: {reward}",
                f"   Mean episode success_rate: 0.0000", f"   Mean episode success_rate_cumulative: 0.5000"]
        return rows + (extra or [])

    trend = HEADER + block(0, 3, "1.0") + block(1, 3, "2.0") + block(1, 3, "3.0") + block(2, 3, "nan")
    s = series(trend, "Mean reward")
    check("series: jeder Block, auch der wiederholte Index", [r[1] for r in s] == [0, 1, 1, 2], str(s))
    check("series: Log-Reihenfolge, Werte im Originaltext", [r[3] for r in s] == ["1.0", "2.0", "3.0", "nan"])
    check("series: Label exakt, kein _cumulative",
          [r[3] for r in series(trend, "Mean episode success_rate")] == ["0.0000"] * 4)
    txt = "\n".join(render_series(trend, ["Mean reward"], 12))
    check("series: nicht-endlich gezaehlt", "1 nicht endlich" in txt, txt)
    check("series: max ist der letzte endliche Wert", "max 3.0 (Block 3, it 1," in txt, txt)
    check("series: fehlendes Label gemeldet", "kein Block traegt" in "\n".join(render_series(trend, ["gibt es nicht"], 5)))

    long_run = HEADER + [l for k in range(1500) for l in block(k, 1500, str(float(k)))]
    txt_lines = render_series(long_run, ["Mean reward"], 12)
    rows_out = [l for l in txt_lines if l.strip().startswith("Block")]
    check("series: Ausgabe gedeckelt (<= points)", len(rows_out) <= 12, str(len(rows_out)))
    check("series: erster und LETZTER Block dabei",
          rows_out[0].strip().startswith("Block     1 it     0") and "it  1499" in rows_out[-1], rows_out[-1])
    check("sample_positions: Rand immer drin", sample_positions(7, 3) == [0, 3, 6])

    # --- --events: count per key and outcome, first and last line ---
    evlog = HEADER + block(0, 2, "1.0") + [
        "[autodr] tilt_hi: rate 0.0000 over 240 -> clamped_zero 0 -> 0",
        "[autodr] lat_x_lo: rate 0.0000 over 240 -> clamped_zero 0 -> 0",
    ] + block(1, 2, "2.0") + [
        "[autodr] tilt_hi: rate 0.9000 over 240 -> advance 0 -> 0.01",
        "  [autodr] indented is not an event",
    ]
    etxt = "\n".join(render_events(evlog, ["[autodr]"]))
    check("events: Gesamtzahl nur am Zeilenanfang", '"[autodr]": 3 Zeilen' in etxt, etxt)
    check("events: je Schluessel", "lat_x_lo 1, tilt_hi 2" in etxt, etxt)
    check("events: je Ausgang", "advance 1, clamped_zero 2" in etxt, etxt)
    check("events: letzte Zeile mit Iteration", "(nach it 1): [autodr] tilt_hi: rate 0.9000" in etxt, etxt)

    print(f"\nself-test: {len(failures)} FAIL von {len(ran)} Faellen")
    return 1 if failures else 0


# --------------------------------------------------------------------------


def main() -> None:
    argv = sys.argv[1:]
    if "--self-test" in argv:
        sys.exit(_self_test())

    head = full = False
    max_lines = 0
    points = SERIES_POINTS
    path_arg = None
    expects: list[str] = []
    series_labels: list[str] = []
    event_prefixes: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--head":
            head = True
        elif a == "--full":
            full = True
        elif a == "--expect":
            i += 1
            if i >= len(argv):
                sys.exit("error: --expect braucht einen Text")
            expects.append(argv[i])
        elif a == "--series":
            i += 1
            if i >= len(argv):
                sys.exit("error: --series braucht ein Label")
            series_labels.append(argv[i])
        elif a == "--events":
            i += 1
            if i >= len(argv):
                sys.exit("error: --events braucht ein Praefix")
            event_prefixes.append(argv[i])
        elif a == "--points":
            i += 1
            if i >= len(argv) or not argv[i].isdigit():
                sys.exit("error: --points braucht eine Zahl")
            points = int(argv[i])
        elif a == "--max-lines":
            i += 1
            if i >= len(argv) or not argv[i].lstrip("-").isdigit():
                sys.exit("error: --max-lines braucht eine Zahl")
            max_lines = int(argv[i])
        elif a.startswith("--"):
            sys.exit(f"error: unbekannte Option {a}")
        elif path_arg is None:
            path_arg = a
        else:
            sys.exit("error: nur ein Pfad erlaubt")
        i += 1

    if path_arg is None:
        sys.exit(
            "usage: python scripts/filter_rt_log.py <log> [--head]"
            ' [--expect "..." ...] [--full [--max-lines N]]'
            ' [--series "Label" ... [--points N]] [--events "[prefix]" ...]\n'
            "       python scripts/filter_rt_log.py --self-test"
        )

    lines = load_lines(Path(path_arg))
    if series_labels or event_prefixes:
        out = render_series(lines, series_labels, points) if series_labels else []
        out += render_events(lines, event_prefixes)
        print("\n".join(out))
    elif head:
        print_head(lines)
    elif full:
        print("\n".join(render_full(lines, max_lines)))
    else:
        print("\n".join(evidence(lines, expects)))


if __name__ == "__main__":
    main()
