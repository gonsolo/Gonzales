#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025-2026 Andreas Wendleder
# SPDX-License-Identifier: Apache-2.0

"""Build book from markdown files with embedded source code snippets.

Source files use markers:  // @doc:snippet-name  ...code...  // @doc:end
Book markdown uses:        {{snippet:path/to/file:snippet-name}}

Usage: python3 docs/build_book.py
Output: docs/book/ (rendered markdown with code inlined)
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BOOK_OUT = DOCS / "book"

MARKER_RE = re.compile(r'@doc:(\S+)')
SNIPPET_REF = re.compile(r'\{\{snippet:([^:]+):([^}]+)\}\}')

COMMENT_CHARS = {
    '.swift': '//', '.scala': '//', '.v': '//', '.sv': '//', '.c': '//', '.h': '//',
    '.py': '#', '.mk': '#', '.sh': '#',
}


def extract_snippets(filepath: Path) -> dict[str, list[str]]:
    """Extract all @doc:name...@doc:end blocks from a source file."""
    suffix = filepath.suffix
    comment = COMMENT_CHARS.get(suffix, '//')
    snippets: dict[str, list[str]] = {}
    current = None
    lines: list[str] = []

    for line in filepath.read_text().splitlines():
        stripped = line.strip().lstrip(comment).strip()
        m = MARKER_RE.match(stripped)
        if m:
            name = m.group(1)
            if name == 'end':
                if current:
                    snippets[current] = lines
                    current = None
                    lines = []
            else:
                current = name
                lines = []
        elif current is not None:
            lines.append(line)

    return snippets


def lang_for(path: str) -> str:
    """Guess markdown code fence language from file extension."""
    ext = Path(path).suffix
    return {'.swift': 'swift', '.scala': 'scala', '.c': 'c', '.h': 'c',
            '.py': 'python', '.v': 'verilog', '.sv': 'systemverilog'}.get(ext, '')


def process_markdown(src: Path, snippet_cache: dict) -> str:
    """Replace {{snippet:file:name}} with actual code blocks."""
    text = src.read_text()

    def replace(m: re.Match) -> str:
        filepath, name = m.group(1), m.group(2)
        full = ROOT / filepath
        key = str(full)
        if key not in snippet_cache:
            snippet_cache[key] = extract_snippets(full)
        snippets = snippet_cache[key]
        if name not in snippets:
            print(f"  WARNING: snippet '{name}' not found in {filepath}", file=sys.stderr)
            return f"<!-- snippet '{name}' not found in {filepath} -->"
        code = '\n'.join(snippets[name])
        lang = lang_for(filepath)
        return f"```{lang}\n{code}\n```"

    result = SNIPPET_REF.sub(replace, text)
    # Rewrite internal .md links to .html for GitHub Pages
    result = re.sub(r'\]\(([A-Za-z0-9]{2}_\w+)\.md\)', r'](\1.html)', result)
    return result


def main():
    BOOK_OUT.mkdir(parents=True, exist_ok=True)
    snippet_cache: dict = {}
    sources = sorted(DOCS.glob("*.md"))

    if not sources:
        print("No markdown files found in docs/. Add .md files to get started.")
        return

    for src in sources:
        out = BOOK_OUT / src.name
        rendered = process_markdown(src, snippet_cache)
        out.write_text(rendered)
        print(f"  {src.name} -> book/{src.name}")

    print(f"Done. {len(sources)} file(s) written to docs/book/")


if __name__ == "__main__":
    main()
