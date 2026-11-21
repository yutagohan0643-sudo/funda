"""銘柄ごとの開示一覧（コード_開示.md）を更新する道具。

使い方（リポジトリの一番上で）:
  python tools/disclosures.py fetch 2334   # 会社の IR ページから開示一覧を取り直し、まだ分析のない開示を表示
  python tools/disclosures.py text 2334 GUID  # その開示の PDF を読み、本文の先頭を表示
  python tools/disclosures.py build 2334   # data/2334/ のデータから 2334_開示.md とテーマ別ファイルを作り直す

データ（data/コード/）:
  items.json    開示の一覧（日付・種類・表題・URL）
  comments.tsv  1件ずつの分析（guid<TAB>分析）。新しい開示はここに1行足す
  stars.txt     流れを変えた重要な開示（★を付ける guid）
  head.md       ファイル前半の「開示から見えたこと」（年表・業績・資金調達・関係者）
  source.json   開示一覧の取り方。type: eirparts（feed）／irpocket（code）／html（pages: [{url, category, pdf_only}]）
  topics.json   テーマ別ファイルの設定（表題か分析に pattern が当たる開示＋include を集め、exclude を除く）
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unicodedata
import urllib.parse
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


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _kind_from(cat: str, title: str) -> str:
    """配信元の分類名と表題から、このリポジトリの種類（news_type）を決める"""
    t = title
    if re.search(r'決算短信', t) and not re.search(r'訂正|説明|補足|を掲載', t):
        return 'tanshin'
    if re.search(r'を掲載しました', t):
        return 'ir_material_for_fiscal_ym'
    if re.search(r'業績予想|配当|自己株式|自社株|株式報酬|買収防衛|役員の異動|代表取締役|特別(?:利益|損失)|計上に関する|減損|主要株主|株式分割|新株|第三者割当|公開買付|資本業務提携|業務提携|子会社|株式取得|事業譲|訴訟|インシデント|不正|中期経営計画.*修正', t):
        return 'press'
    if False:
        return 'tanshin'
    if re.search(r'有価証券報告書|四半期報告書|半期報告書|内部統制報告書|臨時報告書|確認書', t) or cat in ('有価証券報告書', '法定開示'):
        return 'yuho'
    if re.search(r'招集|株主総会|決議ご?通知', t) or cat == '株主総会':
        return 'gms'
    if re.search(r'コーポレート・?ガバナンス', t):
        return 'cgr_ds'
    if cat in ('IR資料', '決算説明資料') or re.search(r'説明資料|補足|説明会|プレゼンテーション|中期経営計画|成長可能性|統合報告|アニュアル|レポート|書き起こし', t):
        return 'ir_material_for_fiscal_ym'
    if cat in ('PR情報', 'お知らせ', 'ニュース', 'プレスリリース', 'topics'):
        return 'pr'
    return 'press'


def fetch_feed(code: str) -> list[dict]:
    src = json.loads((ddir(code) / 'source.json').read_text())
    typ = src.get('type', 'eirparts')
    if typ == 'eirparts':
        s = _get(src['feed']).decode('utf-8-sig')
        d = json.loads(s[s.index('(') + 1:s.rindex(')')])
        return [{k: x.get(k, '') for k in KEEP} for x in d['item']]
    if typ == 'irpocket':  # IR Pocket（例: サイバートラスト）
        s = _get(f"https://xml.irpocket.com/{src['code']}/JS/release-all-all.js").decode('utf-8')
        d = json.loads(s[s.index('(') + 1:s.rindex(')')])
        out = []
        for x in d['item']:
            link = x.get('link', '')
            link = 'https:' + link if link.startswith('//') else link
            out.append({'guid': 'irp_' + x['id'], 'date': x['published'].replace('-', '/') + ('' if x['published'].count(':') == 2 else ':00'),
                        'title': x['title'].strip(), 'link': link, 'news_type': _kind_from(x.get('category_name', ''), x['title']),
                        'type': 'pdf' if link.lower().endswith('.pdf') else 'url'})
        return out
    if typ == 'html':  # 会社の IR ページを読む（日付と PDF のリンクを拾う）
        out, seen = [], set()
        for page in src['pages']:
            url, cat = page['url'], page.get('category', '')
            h = _get(url).decode('utf-8', 'ignore')
            # <dt>2026.11.13</dt><dd><a href="x.pdf">表題</a> 型と、<a href="x.pdf"><p>2026年11月13日</p><p>表題</p></a> 型
            pat = re.compile(r'(\d{4})[./年](\d{1,2})[./月](\d{1,2})日?\s*(?:</[^>]+>\s*)*(?:<[^>]+>\s*)*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                             r'|<a[^>]+href="([^"]+)"[^>]*>\s*(?:<[^>]+>\s*)*(\d{4})年(\d{1,2})月(\d{1,2})日(.*?)</a>'
                             r'|<a[^>]+href="([^"]+\.pdf)"[^>]*>(.*?)</a>', re.S)
            for m in pat.finditer(h):
                if m.group(4):
                    y, mo, da, link, title = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
                elif m.group(6):
                    link, y, mo, da, title = m.group(6), m.group(7), m.group(8), m.group(9), m.group(10)
                else:
                    link, title = m.group(11), m.group(12)
                    um = re.search(r'/uploads/(\d{4})/(\d{2})/', link)
                    fm = re.search(r'(20\d{2})(\d{2})(\d{2})', link.rsplit('/', 1)[-1])
                    if fm:
                        y, mo, da = fm.groups()
                    elif um:
                        y, mo, da = um.group(1), um.group(2), '00'
                    else:
                        continue
                title = re.sub(r'<[^>]+>', ' ', title)
                title = re.sub(r'\s+', ' ', title).strip()
                if not title or link in seen:
                    continue
                if page.get('pdf_only') and not link.lower().endswith('.pdf'):
                    continue
                link = urllib.parse.urljoin(url, link)
                seen.add(link)
                gid = 'h_' + hashlib.md5(link.encode()).hexdigest()[:12]
                out.append({'guid': gid, 'date': f'{y}/{int(mo):02d}/{int(da):02d} 00:00:00', 'title': title, 'link': link,
                            'news_type': _kind_from(cat, title), 'type': 'pdf' if link.lower().endswith('.pdf') else 'url'})
        return out
    raise SystemExit(f'不明な source.type: {typ}')


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
    write_list(code, items, com, d / 'head.md', ROOT / f'{code}_開示.md', '全開示一覧（新しい順）')
    # テーマ別のファイル（data/コード/topics.json）
    tp = d / 'topics.json'
    for t in json.loads(tp.read_text()) if tp.exists() else []:
        pat = re.compile(t['pattern'])
        inc, exc = set(t.get('include', [])), set(t.get('exclude', []))
        sel = [x for x in items if x['guid'] not in exc and (
            x['guid'] in inc or pat.search(x['title']) or pat.search(com.get(x['guid'], '')))]
        write_list(code, sel, com, d / t['head'], ROOT / t['file'], '開示一覧（新しい順）')


def write_list(code: str, items: list[dict], com: dict[str, str], head: Path, out: Path, heading: str) -> None:
    d = ddir(code)
    stars = set((d / 'stars.txt').read_text().split()) if (d / 'stars.txt').exists() else set()
    lines = [head.read_text().rstrip(), '', f'## {heading}', '',
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
