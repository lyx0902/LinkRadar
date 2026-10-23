"""Offline validation for current Markdown links and SVG XML."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

ROOT=Path(__file__).resolve().parents[1]
LINK=re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')

def check_markdown(files):
    failures=[];count=0
    for path in files:
        text=path.read_text(encoding='utf-8-sig')
        text=re.sub(r'```.*?```','',text,flags=re.S)
        for match in LINK.finditer(text):
            url=match.group(1).strip()
            if url.startswith('<') and '>' in url:url=url[1:url.index('>')]
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:',url) or url.startswith('#'):continue
            url=unquote(url.split('#',1)[0].split('?',1)[0])
            if not url:continue
            count+=1
            target=(path.parent/url).resolve()
            if not target.exists():failures.append(f'{path.relative_to(ROOT)}: missing {url}')
    return count,failures


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    files=[ROOT/'README.md']+sorted((ROOT/'docs').rglob('*.md'))+[ROOT/'web/linkradar/README.md']
    count,failures=check_markdown(files)
    for path in (ROOT/'docs/figures').glob('*.svg'):
        try:ElementTree.parse(path)
        except ElementTree.ParseError as exc:failures.append(f'{path}: invalid SVG XML: {exc}')
    print(json.dumps(dict(markdown_files=len(files),local_links=count,svg_xml='checked',failures=failures),ensure_ascii=False,indent=2))
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
