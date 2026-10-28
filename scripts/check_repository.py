"""Check first-party Python syntax and Markdown local files/fragments.
Run from any directory with: python scripts/check_repository.py
External URLs and model behavior are deliberately outside this check.
"""
import ast
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(
    ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
).decode().split('\0')
files = sorted({ROOT / p for p in paths if p and (ROOT / p).is_file()})
errors = []

def anchors(path):
    result, counts = set(), {}
    in_fence = False
    for line in path.read_text().splitlines():
        if line.startswith(('```', '~~~')):
            in_fence = not in_fence
        if in_fence or not re.match(r'^#{1,6} ', line):
            continue
        title = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', line.lstrip('#').strip())
        slug = ''.join(c for c in title.lower() if c.isalnum() or c in ' _-').replace(' ', '-')
        n = counts.get(slug, 0)
        counts[slug] = n + 1
        result.add(slug if not n else f'{slug}-{n}')
    return result

for path in files:
    if path.suffix == '.py':
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
    if path.suffix != '.md':
        continue
    # Ignore fenced code examples containing template Markdown.
    source = re.sub(r'^```.*?^```[^\n]*', '', path.read_text(), flags=re.M | re.S)
    for dest in re.findall(r'\]\(([^)\s]+)\)', source):
        if re.match(r'^[a-zA-Z][\w+.-]*:', dest) or '<' in dest:
            continue
        target, _, fragment = unquote(dest).partition('#')
        resolved = (path.parent / target).resolve() if target else path
        if not resolved.exists():
            errors.append(f'{path.relative_to(ROOT)}: missing {dest}')
        elif fragment and resolved.suffix == '.md' and fragment not in anchors(resolved):
            errors.append(f'{path.relative_to(ROOT)}: missing heading {dest}')
for error in errors:
    print(error)
print(f'Checked {sum(p.suffix == ".py" for p in files)} Python files and '
      f'{sum(p.suffix == ".md" for p in files)} Markdown files; {len(errors)} errors.')
raise SystemExit(bool(errors))
