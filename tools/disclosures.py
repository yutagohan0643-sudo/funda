"""銘柄ごとの開示一覧（コード_開示.md）を更新する道具。

使い方（リポジトリの一番上で）:
  python tools/disclosures.py fetch 2334   # 会社の IR ページから開示一覧を取り直し、まだ分析のない開示を表示
  python tools/disclosures.py text 2334 GUID  # その開示の PDF を読み、本文の先頭を表示
  python tools/disclosures.py build 2334   # data/2334/ のデータから 2334_開示.md を作り直す

データ（data/コード/）:
  items.json    開示の一覧（日付・種類・表題・URL）
  comments.tsv  1件ずつの分析（guid<TAB>分析）。新しい開示はここに1行足す
  stars.txt     流れを変えた重要な開示（★を付ける guid）
  head.md       ファイル前半の「開示から見えたこと」（年表・業績・資金調達・関係者）
  source.json   開示一覧の取り方（今は eir-parts の IR 配信のみ）
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIND = {'press': '適時開示', 'tanshin': '決算短信', 'pr': 'お知らせ', 'yuho': '法定書類', 'gms': '株主総会',
        'announcement': '株主総会', 'cgr_ds': 'ガバナンス', 'financial_results': '英文決算',
        'ir_material_for_fiscal_ym': 'IR資料'}
KEEP = ('guid', 'date', 'title', 'link', 'news_type', 'type')
UA = {'User-Agent': 'Mozilla/5.0 (funda disclosure tracker)'}


def ddir(code: str) -> Path:
    return ROOT / 'data' / code


def load_items(code: str) -> list[dict]:
    p = ddir(code) / 'items.json'
    return json.loads(p.read_text()) if p.exists() else []


def load_comments(code: str) -> dict[str, str]:
    p = ddir(code) / 'comments.tsv'
    out = {}
    for line in p.read_text().splitlines() if p.exists() else []:
        if line.startswith('#') or '\t' not in line:
            continue
        g, c = line.split('\t', 1)
        out[g] = c.strip()
    return out


def fetch_feed(code: str) -> list[dict]:
    src = json.loads((ddir(code) / 'source.json').read_text())
    url = src['feed']
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        s = r.read().decode('utf-8-sig')
    d = json.loads(s[s.index('(') + 1:s.rindex(')')])
    return [{k: x.get(k, '') for k in KEEP} for x in d['item']]


def cmd_fetch(code: str) -> None:
    old = {x['guid']: x for x in load_items(code)}
    new = fetch_feed(code)
    merged = {x['guid']: x for x in new}
    for g, x in old.items():  # 一覧から消えた古い開示も残す
        merged.setdefault(g, x)
    items = sorted(merged.values(), key=lambda x: (x['date'], x['guid']), reverse=True)
    (ddir(code) / 'items.json').write_text(json.dumps(items, ensure_ascii=False, indent=0) + '\n')
    added = [x for x in items if x['guid'] not in old]
    com = load_comments(code)
    todo = [x for x in items if x['guid'] not in com]
    print(f'一覧 {len(items)}件（新規 {len(added)}件）。分析がまだの開示: {len(todo)}件')
    for x in todo:
        print(f"{x['date']}\t{x['guid']}\t{KIND.get(x['news_type'], x['news_type'])}\t{x['title']}\t{x['link']}")


def cmd_text(code: str, guid: str, pages: int = 6) -> None:
    x = next((x for x in load_items(code) if x['guid'] == guid), None)
    if not x:
        sys.exit(f'{guid} が items.json にありません')
    if x.get('type') != 'pdf' or not x['link'].endswith('.pdf'):
        print(f"PDF ではありません: {x['link']}")
        return
    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / 'a.pdf'
        with urllib.request.urlopen(urllib.request.Request(x['link'], headers=UA), timeout=120) as r:
            pdf.write_bytes(r.read())
        txt = subprocess.run(['pdftotext', '-l', str(pages), '-layout', str(pdf), '-'],
                             capture_output=True, text=True).stdout
    txt = unicodedata.normalize('NFKC', txt)
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n\s*\n+', '\n', txt)
    print(f"# {x['date']} {x['title']}\n# {x['link']}\n")
    print(txt[:8000])


def cmd_build(code: str) -> None:
    d = ddir(code)
    items = load_items(code)
    com = load_comments(code)
    stars = set((d / 'stars.txt').read_text().split()) if (d / 'stars.txt').exists() else set()
    lines = [(d / 'head.md').read_text().rstrip(), '', '## 全開示一覧（新しい順）', '',
             '★ は流れを変えた重要な開示。「分析」は開示本文を読んだうえでのひとこと（定型の書類は種類ごとの説明）。', '']
    by = collections.defaultdict(list)
    for x in items:
        by[x['date'][:4]].append(x)
    missing = 0
    for y in sorted(by, reverse=True):
        xs = sorted(by[y], key=lambda x: x['date'], reverse=True)
        lines += [f'### {y}年（{len(xs)}件）', '', '| 日付 | 種類 | 表題 | 分析 |', '|---|---|---|---|']
        for x in xs:
            c = com.get(x['guid'])
            if c is None:
                missing += 1
                c = '（分析はこれから）'
            t = x['title'].replace('|', '／').replace('\n', ' ')
            st = '★' if x['guid'] in stars else ''
            lines.append(f"| {x['date'][5:10]} | {KIND.get(x['news_type'], x['news_type'])} | {st}[{t}]({x['link']}) | {c.replace('|', '／')} |")
        lines.append('')
    out = ROOT / f'{code}_開示.md'
    out.write_text('\n'.join(lines) + '\n')
    print(f'{out.name}: {len(items)}件（分析なし {missing}件）')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, code = sys.argv[1], sys.argv[2]
    if cmd == 'fetch':
        cmd_fetch(code)
    elif cmd == 'text':
        cmd_text(code, sys.argv[3])
    elif cmd == 'build':
        cmd_build(code)
    else:
        sys.exit(__doc__)
