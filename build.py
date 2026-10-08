"""Bundle promo.html and music.mp3 into one self-contained HTML player.

    python build.py  ->  虚功原理宣传片.html  (fonts from Google Fonts; music embedded)
"""
import base64, json, os
here = os.path.dirname(os.path.abspath(__file__))
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700;900&family=Cinzel:wght@400;700'
         '&family=Cormorant+Garamond:ital,wght@0,500;1,500&family=STIX+Two+Text:ital,wght@0,400;1,400&display=block" rel="stylesheet">')
html = open(os.path.join(here, 'promo.html'), encoding='utf-8').read().replace('<!--FONTS-->', FONTS)
audio = base64.b64encode(open(os.path.join(here, 'music.mp3'), 'rb').read()).decode('ascii')
marker = "<script>\n'use strict';"
assert marker in html
html = html.replace(marker, f'<script>window.EMBED = {json.dumps({"audio": audio})};</script>\n' + marker, 1)
out = os.path.join(here, '虚功原理宣传片.html')
open(out, 'w', encoding='utf-8').write(html)
print(f'wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)')
