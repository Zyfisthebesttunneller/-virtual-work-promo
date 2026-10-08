"""Render promo.html offline, frame by frame, in headless Chromium.

    python export.py --shots 5,20.5,60          single frames -> shots/
    python export.py --cues cues.json           sound-effect cue sheet for music.py
    python export.py --from 0 --to 30 --out seg.mp4 [--fps 30]   video-only segment
"""
import argparse, base64, os, re, subprocess, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = '/home/claude/fonts'


def local_font_css():
    """The fontsource CSS (with unicode-ranges) rewritten to absolute file URLs."""
    css = []
    for pkg, files in [('fontsource-cinzel-5.3.0', ['400.css', '700.css']),
                       ('fontsource-cormorant-garamond-5.3.0', ['500.css', '500-italic.css']),
                       ('fontsource-stix-two-text-5.3.0', ['400.css', '400-italic.css'])]:
        for f in files:
            s = open(os.path.join(FONTS, pkg, f), encoding='utf-8').read()
            css.append(s.replace('./files/', f'file://{FONTS}/{pkg}/files/'))
    return '<style>\n' + '\n'.join(css) + '\n</style>'


def build_export_page():
    html = open(os.path.join(HERE, 'promo.html'), encoding='utf-8').read()
    html = html.replace('<!--FONTS-->', local_font_css())
    out = os.path.join(HERE, 'promo_export.html')
    open(out, 'w', encoding='utf-8').write(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shots'); ap.add_argument('--cues'); ap.add_argument('--out')
    ap.add_argument('--from', dest='t0', type=float, default=0); ap.add_argument('--to', dest='t1', type=float)
    ap.add_argument('--fps', type=int, default=30); ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    page_path = build_export_page()
    log = open(a.log, 'a') if a.log else sys.stdout
    with sync_playwright() as p:
        b = p.chromium.launch(args=['--disable-gpu', '--allow-file-access-from-files', '--font-render-hinting=none', '--force-color-profile=srgb'])
        pg = b.new_page(viewport={'width': 1920, 'height': 1080})
        pg.on('pageerror', lambda e: print('PAGE ERROR:', e, file=log, flush=True))
        pg.on('console', lambda m: print('console:', m.text, file=log, flush=True) if m.type == 'error' and 'ERR_' not in m.text else None)
        pg.goto('file://' + page_path + '?export=1')
        for _ in range(120):
            time.sleep(.5)
            try:
                if pg.evaluate("typeof window.__frame === 'function' && window.__ready()"):
                    break
            except Exception:
                pass
        if a.cues:
            import json
            json.dump(pg.evaluate('window.__cues()'), open(a.cues, 'w'), indent=1)
            print('wrote', a.cues, file=log); return
        if a.shots:
            os.makedirs(os.path.join(HERE, 'shots'), exist_ok=True)
            for t in [float(x) for x in a.shots.split(',')]:
                t_0 = time.time(); url = pg.evaluate(f'__frame({t})')
                fn = os.path.join(HERE, 'shots', f't{t:07.2f}.jpg')
                open(fn, 'wb').write(base64.b64decode(url.split(',', 1)[1]))
                print(f'{fn}  {time.time() - t_0:.2f}s', file=log, flush=True)
            return
        dur = pg.evaluate('window.__duration')
        t1 = min(a.t1 if a.t1 is not None else dur, dur)
        n = round((t1 - a.t0) * a.fps)
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', str(a.fps), '-c:v', 'mjpeg', '-i', '-',
                               '-c:v', 'libx264', '-preset', 'fast', '-crf', str(a.crf), '-pix_fmt', 'yuv420p', '-profile:v', 'high',
                               '-r', str(a.fps), '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
        t_start = time.time()
        for i in range(n):
            url = pg.evaluate(f'__frame({a.t0 + i / a.fps})')
            ff.stdin.write(base64.b64decode(url.split(',', 1)[1]))
            if i % 60 == 0 or i == n - 1:
                el = time.time() - t_start
                print(f'{a.out}: frame {i + 1}/{n}  {(i + 1) / max(el, 1e-6):.2f} fps  eta {(n - i - 1) / max((i + 1) / max(el, 1e-6), 1e-6):.0f}s', file=log, flush=True)
        ff.stdin.close(); ff.wait()
        print(f'done {a.out} in {time.time() - t_start:.0f}s', file=log, flush=True)
        b.close()


if __name__ == '__main__':
    main()
