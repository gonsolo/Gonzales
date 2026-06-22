#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025-2026 Andreas Wendleder
# SPDX-License-Identifier: Apache-2.0

"""Build the Gonzales renderer book from Markdown chapters + live source snippets.

Snippet tagging in source files
--------------------------------
Any source file (Mojo, Python, C, …) can export named snippets:

    # <<listing: Point3f>>
    struct Point3f(TrivialRegisterPassable):
        var x: Float32
        var y: Float32
        var z: Float32
    # <</listing>>

Old-style markers are also supported for backwards compatibility:
    // @doc:snippet-name  ...  // @doc:end

Embedding in Markdown
----------------------
Place a reference anywhere in a .md chapter:

    <!-- <<listing: Point3f>> -->

The builder replaces the marker with a fenced Mojo code block containing
the exact lines from the source.  If the listing is not found, a warning
is printed and a visible placeholder is inserted instead.

Building
--------
    python3 docs/build_book.py [--watch] [--html]

    --watch   Rebuild automatically on source changes (requires watchdog).
    --html    Also produce a self-contained HTML book via Pandoc (if available).

Output: docs/book/*.md   (Markdown with snippets inlined)
        docs/book/*.html (optional, requires Pandoc)
"""

from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DOCS     = ROOT / "docs"
SRC      = ROOT / "src/gonzales"
BOOK_OUT = DOCS / "book"

# ── Snippet extraction ────────────────────────────────────────────────────────

# New-style markers:  # <<listing: Name>>  …  # <</listing>>
NEW_OPEN  = re.compile(r'#\s*<<listing:\s*(.+?)>>')
NEW_CLOSE = re.compile(r'#\s*<</listing>>')

# Old-style markers:  // @doc:name  …  // @doc:end  (any comment prefix)
OLD_OPEN  = re.compile(r'(?://|#)\s*@doc:(\S+)')
OLD_CLOSE = re.compile(r'(?://|#)\s*@doc:end')

# Markdown reference:  <!-- <<listing: Name>> -->
MD_REF    = re.compile(r'<!--\s*<<listing:\s*(.+?)>>\s*-->')

# Old-style markdown reference:  {{snippet:path/to/file:name}}
OLD_REF   = re.compile(r'\{\{snippet:([^:]+):([^}]+)\}\}')

LANG_MAP = {
    '.mojo': 'mojo', '.swift': 'swift', '.scala': 'scala',
    '.c': 'c', '.h': 'c', '.cpp': 'cpp',
    '.py': 'python', '.v': 'verilog', '.sv': 'systemverilog',
    '.sh': 'bash', '.mk': 'makefile',
}


def extract_snippets(path: Path) -> dict[str, str]:
    """Return {name: code_text} for all tagged snippets in *path*."""
    snippets: dict[str, list[str]] = {}
    current: str | None = None
    lines_buf: list[str] = []

    for raw in path.read_text(encoding='utf-8').splitlines():
        # New-style open
        m = NEW_OPEN.search(raw)
        if m:
            current = m.group(1).strip()
            lines_buf = []
            continue
        # New-style close
        if NEW_CLOSE.search(raw):
            if current is not None:
                snippets[current] = _dedent(lines_buf)
            current = None
            lines_buf = []
            continue
        # Old-style open
        m = OLD_OPEN.search(raw)
        if m and m.group(1) != 'end':
            current = m.group(1).strip()
            lines_buf = []
            continue
        # Old-style close
        if OLD_CLOSE.search(raw):
            if current is not None:
                snippets[current] = _dedent(lines_buf)
            current = None
            lines_buf = []
            continue
        # Body
        if current is not None:
            lines_buf.append(raw)

    if current is not None:
        # Unclosed snippet — still save it
        print(f"  WARNING: unclosed snippet '{current}' in {path}", file=sys.stderr)
        snippets[current] = _dedent(lines_buf)

    return {k: v for k, v in snippets.items()}


def _dedent(lines: list[str]) -> str:
    """Strip common leading whitespace (like textwrap.dedent) and trailing blank lines."""
    text = '\n'.join(lines)
    text = textwrap.dedent(text)
    return text.rstrip()


# ── Snippet cache (keyed by resolved path) ────────────────────────────────────

_cache: dict[Path, dict[str, str]] = {}
_all_snippets: dict[str, tuple[Path, str]] = {}   # name -> (source_path, code)


def _load_all_snippets() -> None:
    """Walk src/gonzales and docs/build_book.py for all tagged snippets."""
    _cache.clear()
    _all_snippets.clear()
    for ext in ('*.mojo', '*.py', '*.c', '*.h', '*.cpp', '*.sh'):
        for p in SRC.rglob(ext):
            snips = extract_snippets(p)
            _cache[p] = snips
            for name, code in snips.items():
                if name in _all_snippets:
                    prev, _ = _all_snippets[name]
                    print(f"  WARNING: duplicate snippet '{name}' in {p} "
                          f"(first seen in {prev})", file=sys.stderr)
                _all_snippets[name] = (p, code)


def _lookup(name: str, ref_doc: Path) -> tuple[str | None, Path | None]:
    """Return (code, source_path) for *name*, or (None, None) if missing."""
    if name in _all_snippets:
        p, code = _all_snippets[name]
        return code, p
    return None, None


# ── Markdown processing ───────────────────────────────────────────────────────

def process_markdown(src: Path) -> str:
    text = src.read_text(encoding='utf-8')

    def replace_new(m: re.Match) -> str:
        name = m.group(1).strip()
        code, origin = _lookup(name, src)
        lang = LANG_MAP.get(origin.suffix, '') if origin else 'mojo'
        if code is None:
            print(f"  WARNING [{src.name}]: snippet '{name}' not found", file=sys.stderr)
            return f"<!-- snippet '{name}' not found -->\n```\n(listing not found)\n```"
        rel = origin.relative_to(ROOT) if origin else '?'
        header = f"<!-- <<listing: {name}>> — from {rel} -->"
        return f"{header}\n```{lang}\n{code}\n```"

    def replace_old(m: re.Match) -> str:
        filepath, name = m.group(1).strip(), m.group(2).strip()
        full = ROOT / filepath
        if full not in _cache:
            _cache[full] = extract_snippets(full) if full.exists() else {}
        code = _cache[full].get(name)
        lang = LANG_MAP.get(full.suffix, '')
        if code is None:
            print(f"  WARNING [{src.name}]: snippet '{name}' not found in {filepath}", file=sys.stderr)
            return f"<!-- snippet '{name}' not found in {filepath} -->"
        return f"```{lang}\n{code}\n```"

    text = MD_REF.sub(replace_new, text)
    text = OLD_REF.sub(replace_old, text)
    return text


# ── HTML output via Pandoc ────────────────────────────────────────────────────

CSS = """\
body { font-family: Georgia, serif; max-width: 820px; margin: 2rem auto;
       padding: 0 1rem; color: #1a1a1a; line-height: 1.65; }
h1 { border-bottom: 2px solid #444; padding-bottom: .3em; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: .2em; margin-top: 2em; }
code { background: #f4f4f4; padding: .1em .3em; border-radius: 3px; font-size: .88em; }
pre { background: #f8f8f8; border-left: 4px solid #888; padding: 1em;
      overflow-x: auto; font-size: .86em; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #ccc; margin-left: 0; padding-left: 1em; color: #555; }
a { color: #2a6496; }
"""


def render_html(md_file: Path, html_file: Path, css_file: Path) -> bool:
    if not shutil.which('pandoc'):
        return False
    cmd = [
        'pandoc', str(md_file),
        '--standalone',
        '--css', str(css_file.name),
        '--highlight-style', 'tango',
        '--metadata', f'title={md_file.stem.replace("_", " ").title()}',
        '-o', str(html_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(html_file.parent))
    if result.returncode != 0:
        print(f"  pandoc error: {result.stderr}", file=sys.stderr)
        return False
    return True


# ── Index generation ──────────────────────────────────────────────────────────

def build_index(chapters: list[Path], html: bool) -> str:
    ext = '.html' if html else '.md'
    lines = ['# Gonzales Renderer — Book\n',
             '*A path tracer written in Mojo.*\n',
             '## Chapters\n']
    for ch in chapters:
        title = ch.stem.replace('_', ' ').title()
        lines.append(f'- [{title}]({ch.stem}{ext})')
    return '\n'.join(lines) + '\n'


# ── Main ──────────────────────────────────────────────────────────────────────

def build(html: bool = False) -> None:
    BOOK_OUT.mkdir(parents=True, exist_ok=True)

    print("Scanning source snippets …")
    _load_all_snippets()
    print(f"  {len(_all_snippets)} snippet(s) found across {len(_cache)} file(s).")

    chapters = sorted(DOCS.glob("*.md"))
    if not chapters:
        print("No .md files found in docs/ — nothing to build.")
        return

    css_file = BOOK_OUT / "book.css"
    css_file.write_text(CSS)

    for src in chapters:
        out_md = BOOK_OUT / src.name
        rendered = process_markdown(src)
        out_md.write_text(rendered, encoding='utf-8')
        suffix = ''
        if html:
            out_html = BOOK_OUT / src.with_suffix('.html').name
            ok = render_html(out_md, out_html, css_file)
            suffix = f' + {out_html.name}' if ok else ' (HTML skipped — pandoc not found)'
        print(f"  {src.name} → book/{out_md.name}{suffix}")

    # Write index
    idx_md = BOOK_OUT / "index.md"
    idx_md.write_text(build_index(chapters, html), encoding='utf-8')
    if html:
        idx_html = BOOK_OUT / "index.html"
        render_html(idx_md, idx_html, css_file)
    print(f"\nDone. {len(chapters)} chapter(s) written to {BOOK_OUT.relative_to(ROOT)}/")

    # Snippet coverage report
    used: set[str] = set()
    for src in chapters:
        for m in MD_REF.finditer(src.read_text()):
            used.add(m.group(1).strip())
    unused = set(_all_snippets) - used
    if unused:
        print(f"\n  {len(unused)} tagged snippet(s) not yet referenced in any chapter:")
        for name in sorted(unused):
            path, _ = _all_snippets[name]
            print(f"    «{name}»  ({path.relative_to(ROOT)})")


def watch_mode() -> None:
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("watchdog not installed. Run: pip install watchdog", file=sys.stderr)
        sys.exit(1)

    import time

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory and (
                event.src_path.endswith('.md') or
                event.src_path.endswith('.mojo')
            ):
                print(f"\n[watch] {event.src_path} changed — rebuilding …")
                build(html=args.html)

    observer = Observer()
    observer.schedule(Handler(), str(DOCS), recursive=False)
    observer.schedule(Handler(), str(SRC), recursive=True)
    observer.start()
    print(f"Watching {DOCS} and {SRC} — Ctrl-C to stop.")
    build(html=args.html)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--watch', action='store_true', help='Rebuild on file changes')
    parser.add_argument('--html',  action='store_true', help='Also produce HTML via Pandoc')
    args = parser.parse_args()

    if args.watch:
        watch_mode()
    else:
        build(html=args.html)
