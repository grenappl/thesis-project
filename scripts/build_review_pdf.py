"""Builds the PDFs sent to the data-science validator.

    uv run --with mistune python scripts/build_review_pdf.py \
        --tag validation-v1 --commit <full hash> \
        --return-date "5 December 2026" \
        --access "GitHub collaborator invite to grenappl/thesis-project" \
        --contact "name@example.com"

Produces, in --out-dir (default samples/):

    data_scientist_review_guide.pdf   the guide and review form: send this first
    reviewer_reference.pdf            the authors' reported figures: the guide tells
                                      the reviewer not to open it until Part 1 is done

Any placeholder you do not supply stays in the document, highlighted in yellow,
and is listed when the script finishes, so an unfinished PDF cannot be sent by
accident. Rendering uses Microsoft Edge or Google Chrome in headless mode (the
guide contains characters such as R^2, 10^-6 and the floor brackets that
reportlab's built-in fonts cannot draw).
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parent.parent

# Placeholder text exactly as it appears in the guide -> the option that fills it.
PLACEHOLDERS = {
    '`[validation tag — fill in before sending]`': 'tag',
    '`[full commit hash — fill in before sending]`': 'commit',
    '`[date — fill in before sending]`': 'return_date',
    '`[access method: collaborator invite / zip — fill in before sending]`': 'access',
    '`[contact details — fill in before sending]`': 'contact',
}

BROWSERS = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
  @bottom-left  { content: "%(footer)s"; font: 8pt 'Segoe UI', Arial, sans-serif; color: #6b7280; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8pt 'Segoe UI', Arial, sans-serif; color: #6b7280; } }
* { box-sizing: border-box; }
body { font: 10pt/1.45 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; color: #111827; }
h1 { font-size: 19pt; margin: 0 0 10pt; padding-bottom: 6pt; border-bottom: 2pt solid #1f3a5f; color: #1f3a5f; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 0.75pt solid #c9ced6; color: #1f3a5f; break-after: avoid; }
h2.pb { break-before: page; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; color: #1f2937; break-after: avoid; }
p { margin: 0 0 6pt; } li { margin-bottom: 3pt; }
hr { border: 0; border-top: 0.75pt solid #d1d5db; margin: 12pt 0; }
code { font: 8.6pt Consolas, 'Courier New', monospace; background: #f1f3f5; padding: 0.5pt 3pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { background: #f6f8fa; border-left: 3pt solid #9aa7b8; padding: 6pt 8pt; margin: 6pt 0 8pt; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.6pt; }
table { width: 100%%; border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.8pt; }
th { background: #eaeef3; text-align: left; font-weight: 600; }
th, td { border: 0.75pt solid #c3c9d2; padding: 3.5pt 5pt; vertical-align: top; overflow-wrap: break-word; }
td code, th code { overflow-wrap: normal; }
tr { break-inside: avoid; }
table.form th:nth-child(1), table.form td:nth-child(1) { width: 5%%; }
table.form th:nth-child(2), table.form td:nth-child(2) { width: 38%%; }
table.form th:nth-child(3), table.form td:nth-child(3) { width: 13%%; }
table.log th:nth-child(1), table.log td:nth-child(1) { width: 6%%; }
table.log th:nth-child(2), table.log td:nth-child(2) { width: 11%%; }
table.log th:nth-child(3), table.log td:nth-child(3) { width: 34%%; }
table.log th:nth-child(4), table.log td:nth-child(4) { width: 27%%; }
tr:has(> td:empty) > td { height: 4em; }
blockquote { margin: 6pt 0 8pt; padding: 0 8pt; min-height: 3.2em; break-inside: avoid; border: 0.75pt solid #c3c9d2; background: #fbfbfc; }
mark.todo { background: #fff176; color: #111827; padding: 0 2pt; border-radius: 2pt; font-weight: 600; }
strong { font-weight: 600; }
"""


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).is_file():
            return candidate
    for name in ('msedge', 'google-chrome', 'chromium', 'chrome'):
        found = shutil.which(name)
        if found:
            return found
    sys.exit('No Edge/Chrome found. Install Microsoft Edge or Google Chrome to render the PDF.')


def to_html(markdown_text: str, title: str, footer: str) -> tuple[str, list[str]]:
    body = mistune.create_markdown(plugins=['table', 'strikethrough'], hard_wrap=True)(markdown_text)
    # Remaining placeholders: highlight them so nothing unfinished goes out unseen.
    unfilled = re.findall(r'<code>(\[[^<]*fill in before sending\])</code>', body)
    body = re.sub(r'<code>(\[[^<]*fill in before sending\])</code>', r'<mark class="todo">\1</mark>', body)
    body = body.replace('<h2>8. Review form</h2>', '<h2 class="pb">8. Review form</h2>')
    # the floor brackets render as thin bars in some fonts; spell the function out in print
    body = body.replace('⌊p/3⌋', 'floor(p/3)')
    body = re.sub(r'<table>(?=(?:(?!</table>).)*<th>Verdict</th>)', '<table class="form">', body, flags=re.S)
    body = re.sub(r'<table>(?=(?:(?!</table>).)*<th>Severity</th>)', '<table class="log">', body, flags=re.S)
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
           f'<style>{CSS % {"footer": footer.replace(chr(34), chr(39))}}</style></head><body>{body}</body></html>')
    return doc, unfilled


def render_pdf(browser: str, html_text: str, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / 'page.html'
        page.write_text(html_text, encoding='utf-8')
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        cmd = [browser, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
               f'--user-data-dir={Path(tmp) / "profile"}', f'--print-to-pdf={out}', page.as_uri()]
        subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    if not out.is_file() or out.stat().st_size < 1000:
        sys.exit(f'The browser did not produce {out}.')


def main() -> int:
    # The placeholder text contains an em dash; a legacy Windows console codepage cannot print it.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', default=str(ROOT / 'samples'))
    parser.add_argument('--tag', help='git tag of the version under review, e.g. validation-v1')
    parser.add_argument('--commit', help='full commit hash of that tag')
    parser.add_argument('--return-date', dest='return_date')
    parser.add_argument('--access', help='how the reviewer gets the repository')
    parser.add_argument('--contact', help='who the reviewer asks questions')
    args = parser.parse_args()

    browser = find_browser()
    out_dir = Path(args.out_dir)
    jobs = [
        (ROOT / 'samples' / 'data_scientist_review_guide.md', 'Data Scientist Validation - Review Guide'),
        (ROOT / 'samples' / 'reviewer_reference.md', 'Reviewer Reference'),
    ]
    all_unfilled: list[str] = []
    for source, title in jobs:
        text = source.read_text(encoding='utf-8')
        for token, option in PLACEHOLDERS.items():
            value = getattr(args, option)
            if value and token in text:
                text = text.replace(token, f'**{value}**')
        doc, unfilled = to_html(text, title, f'{title} | Lyric and Audio Emotional Alignment thesis')
        out = out_dir / (source.stem + '.pdf')
        render_pdf(browser, doc, out)
        all_unfilled += unfilled
        print(f'wrote {out}  ({out.stat().st_size / 1024:.0f} KB)')

    if all_unfilled:
        print('\nSTILL UNFILLED (highlighted yellow in the PDF; do not send yet):')
        for item in dict.fromkeys(all_unfilled):
            print(f'  - {item}')
    else:
        print('\nAll placeholders filled.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
