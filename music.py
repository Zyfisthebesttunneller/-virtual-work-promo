"""Score and sound design for 虚功原理: D minor, 90 BPM, locked to promo.html's bar timeline.

    python export.py --cues cues.json     (the page's sound-effect cue sheet)
    python music.py                       -> music.wav, music.mp3

Organ, formant choir, strings, piano, celesta, plucks and taiko, plus synthesized foley placed from the
cue sheet; separate buses with convolution reverb, kick sidechain, glue compression and a limiter,
then ffmpeg two-pass loudness normalization to -14 LUFS. Every "balance" moment resolves to D major.
"""
import json, subprocess, wave
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, fftconvolve, lfilter
from scipy.ndimage import maximum_filter1d

SR = 44100
BPM = 90
BEAT = 60 / BPM
BAR = 4 * BEAT
EIGHTH = BEAT / 2
NBARS = 61
TOTAL = NBARS * BAR + 4.0
N = int((TOTAL + 1) * SR)
F32 = np.float32
TAU = 2 * np.pi
rng = np.random.default_rng(1717)
BUS = {k: np.zeros((2, N), F32) for k in ('mus', 'pad', 'drm', 'sfx', 'hall', 'plate', 'room', 'sfxwet')}
DUCK = []                                       # (time, depth) for the sidechain on sustained pads


def midi(m): return 440.0 * 2 ** ((m - 69) / 12)
def b(x): return x * BAR
def tv(d): return np.arange(max(1, int(d * SR))) / SR
def noise(n): return rng.standard_normal(n).astype(F32)


def put(sig, t0, bus, gain=1.0, pan=0.0, hall=0.0, plate=0.0, room=0.0, wet=0.0):
    sig = np.asarray(sig, F32)
    if sig.ndim == 1:
        a = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)]) * 1.4142
    i = int(round(t0 * SR))
    if i < 0:
        sig, i = sig[:, -i:], 0
    n = min(sig.shape[1], N - i)
    if n <= 0:
        return
    seg = sig[:, :n] * gain
    BUS[bus][:, i:i + n] += seg
    for k, g in (('hall', hall), ('plate', plate), ('room', room), ('sfxwet', wet)):
        if g:
            BUS[k][:, i:i + n] += seg * g


def env(n, a, d, s, r, hold):
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    return (e * (1 - np.clip((t - hold) / max(r, 1e-4), 0, 1))).astype(F32)


_SOS = {}
def _sos(o, f, kind):
    k = (o, f if np.isscalar(f) else tuple(f), kind)
    if k not in _SOS:
        _SOS[k] = butter(o, f, kind, fs=SR, output='sos')
    return _SOS[k]
def lp(x, f, o=2): return sosfilt(_sos(o, min(f, SR * .45), 'low'), x).astype(F32)
def hp(x, f, o=2): return sosfilt(_sos(o, max(f, 10), 'high'), x).astype(F32)
def bp(x, lo, hi, o=2): return sosfilt(_sos(o, [max(lo, 10), min(hi, SR * .45)], 'band'), x).astype(F32)


def saw(f, t, det=0.0, vib=0.0, vr=5.5, ph=0.0):
    """Band-limited sawtooth (polyBLEP) with optional vibrato."""
    inst = f * (1 + det) * (1 + vib * np.sin(TAU * vr * t + ph)) if vib else np.full(t.shape, f * (1 + det))
    phase = (np.cumsum(inst, dtype=np.float64) / SR + ph / TAU) % 1.0
    dt = inst / SR
    y = 2 * phase - 1
    m = phase < dt; x = phase[m] / dt[m]; y[m] -= 2 * x - x * x - 1
    m = phase > 1 - dt; x = (phase[m] - 1) / dt[m]; y[m] -= x * x + 2 * x + 1
    return y.astype(F32)


def chirp(f0, f1, d):
    t = tv(d); f = f0 * (f1 / f0) ** (t / d)
    return np.sin(TAU * np.cumsum(f) / SR).astype(F32), t


# ================================================================== instruments
REG = {'full': [(1, 1), (2, .62), (3, .3), (4, .3), (6, .13), (8, .1)], 'flute': [(1, 1), (2, .22), (3, .05)], 'pedal': [(.5, .6), (1, 1), (2, .45), (3, .2)]}


def organ(t0, notes, dur, gain=.06, att=.4, rel=1.4, reg='full', hall=1.0, spread=.6, bus='pad'):
    n = int((dur + rel) * SR); t = np.arange(n) / SR
    e = env(n, att, .5, .92, rel, dur) * (1 + .025 * np.sin(TAU * 5.4 * t)).astype(F32)
    H = REG[reg]
    for i, m in enumerate(notes):
        f = midi(m); s = np.zeros(n, F32)
        for h, a in H:
            if f * h > 8000: continue
            for dv in (-.0012, .0011):
                s += (a * np.sin(TAU * f * h * (1 + dv) * t + rng.random() * TAU)).astype(F32)
        c = int(.06 * SR); s[:c] += bp(noise(c), f * 1.5, f * 6) * np.exp(-np.arange(c) / SR / .015).astype(F32) * .35
        put(s * e / (2 * len(H)), t0, bus, gain, ((i / max(1, len(notes) - 1)) - .5) * spread, hall=hall)


VOW = {'a': [(800, 1.0), (1150, .5), (2900, .22), (3900, .1)], 'o': [(450, 1.0), (800, .45), (2830, .12)], 'u': [(325, 1.0), (700, .3), (2530, .08)]}


def choir(t0, notes, dur, gain=.05, att=1.0, rel=1.8, vowel='a', hall=1.3):
    n = int((dur + rel) * SR); t = np.arange(n) / SR; e = env(n, att, 1.0, .9, rel, dur)
    for i, m in enumerate(notes):
        f = midi(m)
        src = sum(saw(f, t, det=(v - 1) * .005, vib=.0045, vr=4.7 + v * .6, ph=rng.random() * 6) for v in range(3)) + noise(n) * .2
        out = sum(a * bp(src, fc * .87, fc * 1.13) for fc, a in VOW[vowel])
        put(out * e, t0, 'pad', gain, ((i / max(1, len(notes) - 1)) - .5) * .8, hall=hall)


def strings(t0, notes, dur, gain=.04, att=.5, rel=1.0, cut=3500, hall=.8, bus='pad'):
    n = int((dur + rel) * SR); t = np.arange(n) / SR; e = env(n, att, .8, .9, rel, dur)
    for i, m in enumerate(notes):
        s = sum(saw(midi(m), t, det=(v - 1.5) * .0035, vib=.0035, vr=5.1 + v * .45, ph=rng.random() * 6) for v in range(4))
        put(lp(s, cut) * e / 4, t0, bus, gain, ((i / max(1, len(notes) - 1)) - .5) * .9, hall=hall)


def spic(t0, m, gain=.05, dur=.16, pan=0.0):
    t = tv(dur + .25); s = sum(saw(midi(m), t, det=d) for d in (-.004, 0, .004)) / 3
    put(lp(s, 2800) * env(len(t), .006, .08, .35, .12, dur), t0, 'mus', gain, pan, hall=.35)


def piano(t0, m, gain=.1, vel=.8, pan=0.0):
    f = midi(m); t = tv(5.0); B = .00015 * 2 ** ((m - 48) / 12); s = np.zeros(len(t), F32)
    for h in range(1, 11):
        fh = f * h * np.sqrt(1 + B * h * h)
        if fh > 12000: break
        a = vel ** (1 + .25 * (h - 1)) / h ** .85
        tau = max(.25, 3.2 - (m - 60) * .04) / (1 + .45 * (h - 1))
        dec = np.exp(-t / tau).astype(F32)
        for dc in (-.0006, 0, .0007):
            s += (a / 3 * np.sin(TAU * fh * (1 + dc) * t + rng.random() * 6)).astype(F32) * dec
    k = int(.012 * SR); s[:k] += lp(noise(k), 1500 + 3000 * vel) * np.exp(-np.arange(k) / SR / .003).astype(F32) * .3 * vel
    put(s * np.minimum(1, t / .002), t0, 'mus', gain, pan, hall=.6)


def bell(t0, m, gain=.05, dec=1.4, pan=0.0, ratio=3.5, index=1.8, bus='mus', hall=.8, plate=0.0, wet=0.0):
    f = midi(m); t = tv(dec * 5)
    s = np.sin(TAU * f * t + index * np.exp(-t / .3) * np.sin(TAU * f * ratio * t)) * np.exp(-t / dec)
    s += .3 * np.sin(TAU * f * 2.01 * t) * np.exp(-t / (dec * .4))
    put(s * np.minimum(1, t / .0015), t0, bus, gain, pan, hall=hall, plate=plate, wet=wet)


def celesta(t0, m, gain=.04, dec=1.0, pan=0.0, bus='mus', wet=0.0):
    bell(t0, m, gain, dec, pan, ratio=4.0, index=.9, bus=bus, hall=.7, plate=.25, wet=wet)


def pluck(t0, m, gain=.04, dur=.2, pan=0.0):
    t = tv(dur + .3); s = saw(midi(m), t, det=.003) + saw(midi(m), t, det=-.003); k = np.exp(-t / .07).astype(F32)
    put((lp(s, 6000) * k + lp(s, 900) * (1 - k)) * env(len(t), .003, .15, .3, .12, dur), t0, 'mus', gain, pan, hall=.3, plate=.3)


def harp(t0, m, gain=.05, pan=0.0):
    f = midi(m); t = tv(3.0)
    s = sum(np.sin(TAU * f * h * t) * np.exp(-t * (1.2 + h * .8)) / h for h in range(1, 7))
    put(s * np.minimum(1, t / .002), t0, 'mus', gain, pan, hall=.8, plate=.2)


def sub(t0, m, dur, gain=.16):
    f = midi(m); t = tv(dur + .3)
    put(np.tanh(1.5 * (np.sin(TAU * f * t) + .2 * np.sin(TAU * 2 * f * t))) * env(len(t), .02, .3, .9, .25, dur), t0, 'pad', gain)


def bass8(t0, m, gain=.1, dur=EIGHTH * .9):
    f = midi(m); t = tv(dur + .15)
    put((lp(saw(f, t), 700) + .6 * np.sin(TAU * f * t).astype(F32)) * env(len(t), .004, .12, .6, .08, dur), t0, 'mus', gain)


def braam(t0, root, dur, gain):
    t = tv(dur + .5)
    for m in (root, root + 12, root + 19, root + 24):
        s = saw(midi(m), t, det=.004) + saw(midi(m), t, det=-.005); k = np.exp(-t / 1.0).astype(F32)
        put(np.tanh(2.2 * (lp(s, 2600) * k + lp(s, 380) * (1 - k))) * env(len(t), .04, 1.5, .5, .5, dur), t0, 'mus', gain / 4, hall=.7)


def kick(t0, gain=.45, duck=.6):
    t = tv(.55); f = 44 + 130 * np.exp(-t / .03)
    s = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / .26) + .35 * hp(noise(len(t)), 2000) * np.exp(-t / .004)
    put(np.tanh(1.6 * s), t0, 'drm', gain, room=.15); DUCK.append((t0, duck))


def taiko(t0, gain=.35, pitch=1.0, pan=0.0):
    t = tv(1.6); f = (58 + 70 * np.exp(-t / .045)) * pitch
    body = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / .55) + .35 * np.sin(TAU * np.cumsum(f * 1.58) / SR) * np.exp(-t / .25)
    s = body + bp(noise(len(t)), 150, 900) * np.exp(-t / .07) * .6 + hp(noise(len(t)), 2500) * np.exp(-t / .004) * .4
    put(np.tanh(1.3 * s), t0, 'drm', gain, pan, room=.3, hall=.35); DUCK.append((t0, .4))


def snare(t0, gain=.18, pan=0.0, big=False):
    t = tv(.7); s = bp(noise(len(t)), 900, 8000) * np.exp(-t / (.22 if big else .14))
    s += .6 * np.sin(TAU * 185 * t) * np.exp(-t / .06) + .3 * np.sin(TAU * 330 * t) * np.exp(-t / .04)
    put(s, t0, 'drm', gain, pan, room=.35, hall=.4 if big else .15)


def clap(t0, gain=.13):
    n = int(.5 * SR); s = np.zeros(n, F32)
    for k, d in enumerate((0, .011, .023)):
        i = int(d * SR); s[i:] += bp(noise(n - i), 1000, 5000) * np.exp(-np.arange(n - i) / SR / (.012 if k < 2 else .12)).astype(F32)
    put(s, t0, 'drm', gain, room=.4, plate=.3)


def hat(t0, gain=.035, open_=False, pan=.25):
    t = tv(.3 if open_ else .07); put(hp(noise(len(t)), 7000) * np.exp(-t / (.1 if open_ else .018)), t0, 'drm', gain, pan, room=.1)


def crash(t0, gain=.1, dur=2.6, rev=False):
    t = tv(dur); s = hp(noise(len(t)), 4500) * np.exp(-t / (dur * .35))
    for r in (3.1, 4.7, 6.9, 9.4):
        s += .08 * np.sin(TAU * 410 * r * t) * np.exp(-t / (dur * .25))
    if rev: put(s[::-1].copy(), t0 - dur, 'drm', gain, hall=.5)
    else: put(s, t0, 'drm', gain, hall=.5)


def tom(t0, m, gain=.22, pan=0.0):
    t = tv(.8); f = midi(m) * (1 + .5 * np.exp(-t / .05))
    put(np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / .25), t0, 'drm', gain, pan, room=.3, hall=.3)


def impact(t0, lvl=1.0, root=26):
    t = tv(6); f = 28 + 60 * np.exp(-t / .3)
    put(np.tanh(2 * np.sin(TAU * np.cumsum(f) / SR)) * np.exp(-t / 1.6), t0, 'drm', .5 * lvl)
    taiko(t0, .45 * lvl, .8); taiko(t0 + .012, .3 * lvl, 1.3, -.3)
    put(lp(noise(len(t)), 2500) * np.exp(-t / .45), t0, 'drm', .1 * lvl, hall=1.2)
    crash(t0, .09 * lvl); braam(t0, root, 4.5, .11 * lvl); DUCK.append((t0, 1.0))


def riser(t_end, dur, gain=.14):
    t = tv(dur); x = t / dur; s = hp(noise(len(t)), 1200) * x ** 2.2 * .7
    for o in range(3):
        f = 180 * 2 ** o * 2 ** (x * 2)
        s += (np.sin(TAU * np.cumsum(f) / SR) * x ** 2 * .12 * np.sin(np.pi * np.clip(x * 1.2 - o * .1, 0, 1))).astype(F32)
    put(s.astype(F32), t_end - dur, 'mus', gain, hall=.6)


# ================================================================== harmony + arrangement helpers
CH = {'Dm': [50, 53, 57, 62], 'Bb': [46, 53, 58, 62], 'F': [48, 53, 57, 60], 'C': [48, 52, 55, 60], 'Gm': [50, 55, 58, 62], 'A': [49, 52, 57, 61],
      'D': [50, 54, 57, 62], 'Dm9': [50, 53, 57, 64], 'Asus': [50, 52, 57, 62], 'Eb': [51, 55, 58, 63]}
ROOT = {'Dm': 38, 'Bb': 34, 'F': 41, 'C': 36, 'Gm': 43, 'A': 33, 'D': 38, 'Dm9': 38, 'Asus': 33, 'Eb': 39}
ARP = {k: sorted(set(v + [x + 12 for x in v])) for k, v in CH.items()}


def pads(start, chords, kind='organ', **kw):
    for i, c in enumerate(chords):
        if kind == 'organ': organ(b(start + i), CH[c], BAR, **kw)
        elif kind == 'strings': strings(b(start + i), CH[c], BAR, **kw)
        else: choir(b(start + i), [x + 12 for x in CH[c][1:]], BAR, **kw)


def bass_hold(start, chords, gain=.12):
    for i, c in enumerate(chords): sub(b(start + i), ROOT[c], BAR, gain)


def bass_8(start, chords, gain=.1):
    for i, c in enumerate(chords):
        for k in range(8): bass8(b(start + i) + k * EIGHTH, ROOT[c] + (12 if k % 4 == 3 else 0), gain)
        sub(b(start + i), ROOT[c], BAR, .08)


def arps(start, chords, inst='celesta', div=2, oct_=12, gain=.03, pat=(0, 2, 1, 3, 2, 4, 3, 5)):
    step = BEAT / div
    for i, c in enumerate(chords):
        notes = ARP[c]
        for k in range(4 * div):
            m = notes[pat[k % len(pat)] % len(notes)] + oct_
            t, g, pan = b(start + i) + k * step, gain * (1 if k % div == 0 else .75), np.sin(k * 1.3) * .5
            if inst == 'celesta': celesta(t, m, g, dec=.7, pan=pan)
            elif inst == 'harp': harp(t, m, g, pan)
            else: pluck(t, m, g, dur=step * .9, pan=pan)


def drums(start, bars, kick_=(0, 2), snare_=(), clap_=(), hats=0, taikos=(), kg=.45, sg=.18, fill=False):
    for i in range(bars):
        t = b(start + i)
        for k in kick_: kick(t + k * BEAT, kg)
        for s in snare_: snare(t + s * BEAT, sg, big=True)
        for c in clap_: clap(t + c * BEAT)
        for k in taikos: taiko(t + k * BEAT, .28, 1.0, pan=.2 * np.sin(k))
        for h in range(hats): hat(t + h * BAR / hats, .035 if h % 2 == 0 else .024)
        if fill and i == bars - 1:
            for k, m in enumerate([50, 47, 45, 43, 41, 38, 36, 33]): tom(t + 2 * BEAT + k * BEAT / 4, m, .2, pan=.6 - k * .17)


def sparkle_notes(t0, dur, count, lo=86, gain=.025, seed=0):
    r = np.random.default_rng(seed); pent = [62, 64, 65, 69, 72]
    for _ in range(count):
        m = pent[r.integers(5)] + 12 * r.integers(1, 3)
        if m < lo: m += 12
        celesta(t0 + r.random() * dur, int(m), gain * (.5 + .5 * r.random()), dec=1.2, pan=r.random() * 1.6 - .8)


def resolve_D(t0, dur, lvl=1.0):
    """The recurring 'balance' gesture: a bright D major chord with choir and a rising celesta."""
    organ(t0, CH['D'] + [38, 45], dur, gain=.06 * lvl, att=.06, rel=1.6)
    choir(t0, [66, 69, 74, 78], dur, gain=.045 * lvl, att=.25, rel=1.6)
    sub(t0, 38, dur, .12 * lvl)
    for k, m in enumerate([74, 78, 81, 86, 90]): celesta(t0 + .1 + k * EIGHTH * .75, m, .03 * lvl, dec=1.6, pan=(k - 2) * .25)


# ================================================================== the score
print('score…')
# ---- 0-6  cold open: drone under the flash-cuts, a hard cut to silence, the question, the lever assembles
organ(0.0, [38, 45], 6.55, gain=.07, att=1.5, rel=.06, reg='pedal')
strings(1.0, [50, 57, 62, 69], 5.55, gain=.03, att=4.5, rel=.06, cut=1600)
riser(6.62, 4.5, .07)
organ(7.35, [62, 69, 74], 3.0, gain=.02, att=1.6, rel=1.6, reg='flute')
organ(b(4), CH['Dm'], b(2) - .25, gain=.05, att=2.6, rel=.5)
choir(b(4) + 2.4, [62, 65, 69], b(2) - 2.7, gain=.03, att=2.2, rel=.5)
sparkle_notes(b(4), b(2) - .5, 22, seed=1)
riser(b(6), b(1), .16)
crash(b(6), .08, 2.2, rev=True)
# ---- 6-9  title
impact(b(6), 1.0)
organ(b(6), CH['Dm'] + [38, 45], b(3) - .3, gain=.075, att=.08, rel=1.6)
choir(b(6), [62, 65, 69, 74], b(3) - .3, gain=.055, att=.5)
sub(b(6), 38, b(3) - .5, .16)
for k, m in enumerate([74, 77, 81, 86, 81, 77, 74, 69, 74, 77, 81, 86]):
    celesta(b(6.75) + k * EIGHTH, m, .035, dec=1.4, pan=(k % 4 - 1.5) * .3)
# ---- 9-17  I balance: calm, then the busy "old way", then "what if?", resolving to D major on "= 0"
BA = ['Dm', 'Bb', 'F', 'C', 'Dm', 'Bb', 'A']
pads(9, BA, 'organ', gain=.036, reg='flute', att=.8)
pads(9, BA, 'strings', gain=.025, att=.9)
bass_hold(9, BA, .11)
arps(11, ['F', 'C'], 'pluck', div=4, gain=.022, pat=(0, 2, 4, 2, 1, 3, 5, 3))
for k in range(16): hat(b(11) + k * BEAT / 2, .02)
arps(13, ['Dm', 'Bb'], 'celesta', div=2, gain=.022)
for i in (13, 14, 15): kick(b(i), .2, duck=.25)
choir(b(15), [61, 64, 69], BAR, gain=.035, att=.8)
riser(b(16) - .05, b(1), .09)
resolve_D(b(16), BAR - .2)
# ---- 17-23  II virtual: hushed, glassy; a piano motif over low strings
VI = ['Dm9', 'Bb', 'Gm', 'A', 'Dm9', 'Asus']
pads(17, VI, 'strings', gain=.024, att=1.2, cut=1800)
organ(b(17), [38, 45], b(6) - .3, gain=.028, att=1.5, reg='pedal')
MOT = [(0, 74), (1, 69), (1.5, 72), (2, 70), (4, 69), (5, 65), (5.5, 67), (6, 69)]
for rep in (17, 19, 21):
    for bt, m in MOT:
        piano(b(rep) + bt * BEAT, m, .07, .6, .15); piano(b(rep) + bt * BEAT, m - 12, .035, .5, -.15)
for i in range(19, 23):
    for k in range(8): bass8(b(i) + k * EIGHTH, ROOT[VI[i - 17]], .045, dur=EIGHTH * .5)
sparkle_notes(b(19), b(3), 18, gain=.018, seed=4)
# ---- 23-32  III the principle: the ledger builds, the helpers vanish, impact on δW = 0
PR = ['Dm', 'Bb', 'F', 'C', 'Gm', 'A', 'A']
pads(23, PR, 'strings', gain=.03)
arps(23, PR, 'pluck', div=4, gain=.021)
bass_8(25, PR[2:], .085)
drums(25, 5, kick_=(0, 2), kg=.3)
drums(27, 3, kick_=(), hats=8)
choir(b(28), [61, 64, 69], b(2), gain=.035, att=1.2)
for k in range(16): snare(b(29) + k * BEAT / 4, .02 + .011 * k)
riser(b(30), b(1.4), .18)
crash(b(30), .08, 2.0, rev=True)
impact(b(30), .9)
organ(b(30), CH['Dm'] + [38, 45], b(2) - .3, gain=.07, att=.08, rel=1.4)
choir(b(30), [62, 65, 69, 74], b(2) - .3, gain=.05, att=.4)
sub(b(30), 38, b(2) - .3, .14)
# ---- 32-39  IV machines: spiccato ostinato and drums; calmer for the force × distance curve
MA = ['Dm', 'C', 'Bb', 'A', 'Dm', 'C', 'A']
impact(b(32), .45)
pads(32, MA, 'strings', gain=.026, cut=2500)
for i, c in enumerate(MA[:5]):
    r = ROOT[c] + 24
    for k in range(8): spic(b(32 + i) + k * EIGHTH, [r, r + 7, r + 12, r + 7][k % 4], .045, pan=(k % 2 - .5) * .4)
bass_8(32, MA[:5], .1)
drums(32, 2, kick_=(0, 2), hats=16, kg=.32)
drums(34, 3, kick_=(0, 1.75, 2, 3.5), snare_=(1, 3), hats=16, taikos=(0,), kg=.4, sg=.16)
bass_hold(37, MA[5:], .1)
arps(37, MA[5:], 'celesta', div=2, gain=.024)
drums(37, 2, kick_=(0, 2), hats=8, kg=.28, fill=True)
# ---- 39-47  V energy landscape: wide and airy; the hanging chain; D major on the cathedral
EN = ['Bb', 'F', 'Gm', 'Dm', 'Gm', 'A', 'Bb']
impact(b(39), .35)
pads(39, EN, 'organ', gain=.034, reg='flute', att=1.0)
pads(39, EN, 'choir', gain=.022, att=1.2, vowel='o')
bass_hold(39, EN, .1)
arps(39, EN[:3], 'celesta', div=4, oct_=24, gain=.017, pat=(0, 3, 1, 4, 2, 5, 3, 6))
for i in range(41, 44): taiko(b(i), .14, .9)
for k, m in enumerate([86, 81, 77, 74, 69, 65, 62, 57, 62, 65, 69, 74, 77, 81]): harp(b(43.8) + k * BEAT / 3, m, .045, pan=np.sin(k) * .5)
riser(b(46), b(1.2), .14)
impact(b(46), .6, root=38)
resolve_D(b(46), BAR - .2, 1.15)
# ---- 47-54  VI d'Alembert and Lagrange (piano over strings), then the finite elements build into the finale
BY = ['Dm', 'Gm', 'A', 'Dm']
impact(b(47), .3)
pads(47, BY, 'strings', gain=.026, att=1.2, cut=2200)
bass_hold(47, BY, .09)
MEL = [(0, 74), (1.5, 72), (2, 69), (4, 70), (5.5, 69), (6, 67), (8, 69), (9.5, 70), (10, 72), (12, 73), (13, 74), (14, 76)]
for bt, m in MEL:
    piano(b(47) + bt * BEAT, m, .085, .7, .1); piano(b(47) + bt * BEAT, m - 12, .04, .6, -.1)
FE = ['Bb', 'C', 'A']
pads(51, FE, 'strings', gain=.03)
arps(50.5, ['Dm'] + FE, 'pluck', div=4, gain=.024)
bass_8(51, FE, .09)
drums(51, 3, kick_=(0, 2), hats=16, kg=.34, fill=True)
riser(b(54), b(1.5), .18)
crash(b(54), .08, 2.0, rev=True)
# ---- 54-61+  finale: montage at full tilt, then the title resolving to D major
FI = ['Dm', 'Bb', 'F', 'C']
impact(b(54), 1.0)
pads(54, FI, 'organ', gain=.05)
pads(54, FI, 'choir', gain=.034)
arps(54, FI, 'celesta', div=4, oct_=12, gain=.02)
bass_8(54, FI, .1)
drums(54, 4, kick_=(0, 1, 2, 3), snare_=(1, 3), hats=16, taikos=(0, 2), kg=.42, sg=.17, fill=True)
crash(b(58), .08, 2.0, rev=True)
impact(b(58), .75)
organ(b(58), CH['Dm9'] + [38], BAR - .1, gain=.06, att=.1, rel=.8)
choir(b(58), [62, 65, 69, 76], BAR - .1, gain=.045, att=.4, rel=.8)
organ(b(59), CH['Bb'] + [34], BAR - .1, gain=.06, att=.3, rel=.8)
choir(b(59), [62, 65, 70, 74], BAR - .1, gain=.045, att=.4, rel=.8)
sub(b(58), 38, BAR, .14); sub(b(59), 34, BAR, .14)
END = TOTAL - b(60)
organ(b(60), CH['D'] + [38, 45, 66, 74], END, gain=.07, att=.3, rel=2.5)
choir(b(60), [66, 69, 74, 78], END - .5, gain=.05, att=.6, rel=2.5)
strings(b(60), [50, 57, 62, 66, 69], END - .5, gain=.03, att=1.0, rel=2.5)
sub(b(60), 38, END - .8, .14)
for k, m in enumerate([74, 78, 81, 86, 81, 78, 81, 86, 90]): celesta(b(60) + .3 + k * EIGHTH, m, .03, dec=1.8, pan=(k % 3 - 1) * .4)
bell(b(60) + 6.0, 86, .035, dec=3.0, index=1.0, hall=1.2)


# ================================================================== sound design from the cue sheet
def whoosh_sig(dur, f0=200, f1=2500, rev=False):
    t = tv(dur); x = t / dur; s = noise(len(t)); out = np.zeros(len(t), F32); M = 24
    for j in range(M):
        a, z = j * len(t) // M, (j + 1) * len(t) // M
        fc = f0 * (f1 / f0) ** np.sin(np.pi * x[(a + z) // 2] * .5 if not rev else np.pi * .5 * (1 - x[(a + z) // 2]))
        out[a:z] = bp(s[max(0, a - 2000):z], fc * .7, fc * 1.4)[-(z - a):]
    return out * np.sin(np.pi * x) ** 1.5


def fx_thump(t, **o): taiko(t, .4, .7)
def fx_hit(t, k=.5, **o):
    kick(t, .3 + .2 * k, duck=.3); s, tt = chirp(3200, 300, .12); put(s * np.exp(-tt / .04), t, 'sfx', .05 * k, wet=.3)
    put(hp(noise(len(tt)), 3000) * np.exp(-tt / .02), t, 'sfx', .1 * k)
def fx_suck(t, dur=.45, **o): put(whoosh_sig(dur, 300, 3000, rev=True)[::-1].copy(), t, 'sfx', .14, wet=.3)
def fx_ring(t, dur=2.5, **o):
    tt = tv(dur); put(sum(np.sin(TAU * f * tt) * np.exp(-tt / (dur * .4)) for f in (1180, 1771, 2493)) / 3, t, 'sfx', .05, wet=.8)
def fx_heart(t, **o): taiko(t, .25, .55); taiko(t + .22, .18, .55)
def fx_sparkle(t, dur=3.0, **o):
    r = np.random.default_rng(int(t * 10))
    for _ in range(int(dur * 9)): celesta(t + r.random() * dur, int(r.choice([86, 88, 90, 93, 95, 98])), .012 + .01 * r.random(), dec=.5, pan=r.random() * 1.6 - .8, bus='sfx', wet=.6)
def sweep_lp(x, f0, f1, M=20):
    out = np.zeros(len(x), F32)
    for j in range(M):
        a, z = j * len(x) // M, (j + 1) * len(x) // M
        out[a:z] = lp(x[max(0, a - 2000):z], f0 * (f1 / f0) ** ((j + .5) / M))[-(z - a):]
    return out
def fx_swell(t, dur=1.4, **o):
    tt = tv(dur); x = tt / dur; put(sweep_lp(noise(len(tt)), 600, 4600) * x ** 2, t, 'sfx', .08, wet=.5)
def fx_whoosh(t, dur=1.2, rev=0, pitch=1.0, **o): put(whoosh_sig(dur, 180 * pitch, 2600 * pitch, bool(rev)), t, 'sfx', .1, (-.4 if rev else .4), wet=.3)
def fx_shimmer(t, **o):
    for k, m in enumerate([81, 86, 88, 93]): bell(t + k * .05, m, .02, dec=1.6, index=.8, bus='sfx', hall=0, wet=.8, pan=(k - 1.5) * .4)
def fx_swish(t, pitch=1.0, **o): put(whoosh_sig(.45, 400 * pitch, 5000 * pitch), t, 'sfx', .08, wet=.2)
def fx_chime(t, midi_=81, **o): bell(t, midi_, .05, dec=2.2, index=1.2, bus='sfx', hall=0, wet=.8)
def fx_tick(t, gain=.35, pitch=1.0, hi=None, **o):
    tt = tv(.05); f = 2400 * pitch * (1.25 if hi else 1.0)
    put(np.sin(TAU * f * tt) * np.exp(-tt / .006) + .4 * hp(noise(len(tt)), 4000) * np.exp(-tt / .003), t, 'sfx', .09 * gain / .35, wet=.15)
def fx_boom(t, **o): impact(t, .55, root=38)
def fx_ding(t, midi_=81, **o): bell(t, midi_, .05, dec=1.6, index=1.4, bus='sfx', hall=0, wet=.7)
def fx_pop(t, midi_=76, **o):
    f = midi(midi_); tt = tv(.18); fr = f * (1 + np.exp(-tt / .015))
    put(np.sin(TAU * np.cumsum(fr) / SR) * np.exp(-tt / .06), t, 'sfx', .04, np.sin(midi_ * 1.7) * .5, wet=.4)
def fx_blip(t, midi_=84, **o):
    f = midi(midi_); tt = tv(.25); fr = f * (1 + .5 * np.exp(-tt / .02))
    put(np.tanh(2.5 * np.sin(TAU * np.cumsum(fr) / SR)) * np.exp(-tt / .1), t, 'sfx', .04, np.sin(midi_) * .4, wet=.3)
def fx_glitch(t, **o):
    tt = tv(.26); s = np.round(noise(len(tt)) * 3) / 3 * (np.sin(TAU * 31 * tt) > 0).astype(F32)
    put(bp(s, 500, 6000), t, 'sfx', .07, wet=.2); fx_blip(t + .05, 96)
def fx_scan(t, dur=2.0, **o):
    tt = tv(dur); s = np.sin(TAU * 1150 * tt + 2 * np.sin(TAU * 7 * tt)) * .4 + bp(noise(len(tt)), 2000, 4000)
    put(s * np.minimum(1, tt / .1) * np.minimum(1, (dur - tt) / .15), t, 'sfx', .02, wet=.3)
def fx_zoom(t, dur=3.5, **o):
    s, tt = chirp(120, 1800, dur); put(s * np.sin(np.pi * tt / dur) ** 2 * .5 + lp(noise(len(tt)), 3000) * (tt / dur) ** 2 * .3, t, 'sfx', .05, wet=.5)
def fx_shatter(t, **o):
    r = np.random.default_rng(3)
    for _ in range(40):
        tt = tv(.12); f = 2500 + 6000 * r.random()
        put(np.sin(TAU * f * tt) * np.exp(-tt / (.01 + .03 * r.random())), t + r.random() ** 2 * .7, 'sfx', .03, r.random() * 1.6 - .8, wet=.4)
    tt = tv(.4); put(hp(noise(len(tt)), 2500) * np.exp(-tt / .08), t, 'sfx', .12, wet=.3)
def fx_dissolve(t, dur=2.6, **o):
    fx_sparkle(t, dur); put(whoosh_sig(dur, 2000, 300), t, 'sfx', .06, wet=.6)
def fx_rope(t, dur=3.0, **o):
    tt = tv(dur); r = np.random.default_rng(int(t)); s = bp(noise(len(tt)), 300, 1600) * (.4 + .6 * (r.random(len(tt)) > .995).astype(F32) * 3)
    put(s * np.sin(np.pi * tt / dur) * .6, t, 'sfx', .08, .3, wet=.2)
    for k in range(int(dur * 6)): fx_tick(t + k / 6, .2, .5)
def fx_gears(t, dur=4.0, **o):
    for k in range(int(dur * 9)):
        tt = tv(.04); put(bp(noise(len(tt)), 1500, 5000) * np.exp(-tt / .006), t + k / 9, 'sfx', .05 if k % 3 else .08, -.3, wet=.2)
    put(lp(saw(55, tv(dur)), 300) * np.sin(np.pi * tv(dur) / dur), t, 'sfx', .03)
def fx_hydro(t, dur=3.0, **o):
    tt = tv(dur); s = lp(noise(len(tt)), 500) * (1 + .5 * np.sin(TAU * .6 * tt))
    for k in range(10):
        bt = tv(.15); f = 300 + 500 * np.random.default_rng(k).random()
        put(np.sin(TAU * np.cumsum(f * (1 + 2 * bt)) / SR) * np.exp(-bt / .04), t + k * dur / 10, 'sfx', .025, .4, wet=.4)
    put(s * np.sin(np.pi * tt / dur), t, 'sfx', .06, .3, wet=.3)
def fx_roll(t, dur=3.0, **o):
    tt = tv(dur); put(lp(noise(len(tt)), 400) * (1 + .6 * np.sin(TAU * 9 * tt)) * np.exp(-tt / (dur * .5)), t, 'sfx', .12, wet=.3)
def fx_success(t, **o):
    bell(t, 86, .045, dec=1.4, index=1.0, bus='sfx', hall=0, wet=.7); bell(t + .12, 93, .045, dec=1.8, index=1.0, bus='sfx', hall=0, wet=.7)
def fx_fall(t, dur=1.6, **o):
    s, tt = chirp(900, 120, dur); put(s * np.exp(-tt / dur) * .5 + lp(noise(len(tt)), 600) * .3, t, 'sfx', .07, .4, wet=.4)
def fx_chain(t, dur=2.2, **o):
    r = np.random.default_rng(12)
    for _ in range(70):
        tt = tv(.05); f = 3000 + 4000 * r.random(); u = r.random() ** 1.6
        put(np.sin(TAU * f * tt) * np.exp(-tt / .008) + .3 * hp(noise(len(tt)), 5000) * np.exp(-tt / .004), t + u * dur, 'sfx', .03, r.random() * 1.4 - .7, wet=.3)
def fx_scribe(t, dur=3.0, **o):
    tt = tv(dur); s = bp(noise(len(tt)), 2500, 7000) * (.5 + .5 * np.abs(np.sin(TAU * 3.3 * tt))) ** 3
    put(s * np.minimum(1, tt / .1) * np.minimum(1, (dur - tt) / .2), t, 'sfx', .05, .3, wet=.2)
def fx_page(t, **o):
    tt = tv(.5); put(bp(noise(len(tt)), 800, 6000) * np.sin(np.pi * tt / .5) ** 3, t, 'sfx', .1, .2, wet=.3)
def fx_data(t, dur=4.0, **o):
    r = np.random.default_rng(int(t)); k = 0
    while k * EIGHTH / 2 < dur:
        if r.random() < .55:
            tt = tv(.05); f = 800 + 2200 * r.random()
            put(np.sign(np.sin(TAU * f * tt)) * np.exp(-tt / .012), t + k * EIGHTH / 2, 'sfx', .013, r.random() - .5, wet=.2)
        k += 1


FX = {k[3:]: v for k, v in list(globals().items()) if k.startswith('fx_')}
cues = json.load(open('cues.json', encoding='utf-8'))
missing = sorted({c['name'] for c in cues} - FX.keys())
if missing:
    raise SystemExit(f'no sound for cues: {missing}')
for c in cues:
    FX[c['name']](c['t'], **{('midi_' if k == 'midi' else k): v for k, v in c.items() if k not in ('t', 'name')})
print(f'placed {len(cues)} sound effects')


# ================================================================== mix + master
def ir(bands, dur, pre=.02, seed=0, er=8):
    r = np.random.default_rng(seed); n = int(dur * SR); t = np.arange(n) / SR; out = []
    for _ in range(2):
        x = r.standard_normal(n).astype(F32); y = np.zeros(n, F32)
        for (lo, hi), tau in bands: y += (bp(x, lo, hi) if lo else lp(x, hi)) * np.exp(-t / tau)
        y *= np.minimum(1, t / .008)
        for k in range(er): y[int((pre + r.random() * .06) * SR)] += (.6 - k * .05) * (1 if r.random() < .5 else -1)
        y = np.concatenate([np.zeros(int(pre * SR), F32), y]); out.append(y / np.sqrt(np.sum(y ** 2)))
    return out


print('reverbs…')
HALL = ir([((0, 300), 1.3), ((300, 1500), 1.05), ((1500, 5000), .7), ((5000, 16000), .35)], 3.8, .025, 1)
PLATE = ir([((0, 500), .55), ((500, 4000), .6), ((4000, 16000), .45)], 1.8, .008, 2)
ROOM = ir([((0, 400), .18), ((400, 6000), .16), ((6000, 16000), .08)], .7, .004, 3)
conv = lambda bus, h: np.stack([fftconvolve(BUS[bus][c], h[c])[:N].astype(F32) for c in range(2)])
wet_music = conv('hall', HALL) * .55 + conv('plate', PLATE) * .5 + conv('room', ROOM) * .6
wet_sfx = conv('sfxwet', PLATE) * .6

duck = np.zeros(N, F32)
for tk, depth in DUCK:
    i = int(tk * SR); L = min(int(.45 * SR), N - i)
    if L > 0: duck[i:i + L] = np.maximum(duck[i:i + L], depth * np.exp(-np.arange(L) / SR / .16))
music = BUS['mus'] + BUS['pad'] * (1 - .5 * duck) + BUS['drm'] * .9 + wet_music
sfx = BUS['sfx'] + wet_sfx

tt = np.arange(N, dtype=F32) / SR
gate = np.clip((6.66 - tt) / .05, 0, 1) + np.clip((tt - 7.2) / .6, 0, 1)        # the hard cut after the teaser
music *= np.minimum(gate, 1)

mix = music * .92 + sfx
mix = np.stack([hp(mix[c], 30) for c in range(2)])
mix -= .6 * sosfiltfilt(butter(2, 120, 'low', fs=SR, output='sos'), mix, axis=1).astype(F32)
mix -= .3 * sosfiltfilt(butter(2, 7000, 'high', fs=SR, output='sos'), mix, axis=1).astype(F32)

mono = np.sqrt((mix[0] ** 2 + mix[1] ** 2) / 2)
a = np.exp(-1 / (.12 * SR)); envl = np.sqrt(lfilter([1 - a], [1, -a], mono ** 2) + 1e-12)
lvl = 20 * np.log10(envl / (np.max(envl) + 1e-9) + 1e-9)
mix *= (10 ** (-np.maximum(0, lvl + 12) * (1 - 1 / 2.2) / 20)).astype(F32)
mix /= np.max(np.abs(mix)) + 1e-9
L = int(.004 * SR); pk = maximum_filter1d(np.max(np.abs(mix), axis=0), size=2 * L + 1)
g = np.minimum(1, .5 / np.maximum(pk, 1e-9)); a = np.exp(-1 / (.09 * SR)); gs = lfilter([1 - a], [1, -a], g)
mix *= np.minimum(g, gs).astype(F32)
end = int(TOTAL * SR); mix = mix[:, :end]
mix *= np.minimum(1, (end - np.arange(end)) / (3.0 * SR)).astype(F32)
mix /= np.max(np.abs(mix)) + 1e-9
with wave.open('music.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix.T * 32000).astype(np.int16).tobytes())
LN = 'loudnorm=I=-14:TP=-1.5:LRA=11'
m = subprocess.run(['ffmpeg', '-hide_banner', '-i', 'music.wav', '-af', LN + ':print_format=json', '-f', 'null', '-'], capture_output=True, text=True).stderr
m = json.loads(m[m.rindex('{'):m.rindex('}') + 1])
LN += f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'music.wav', '-af', LN + ',alimiter=limit=-1.8dB:level=false', '-ar', '44100', '-b:a', '192k', 'music.mp3'], check=True)
print(f'music: {TOTAL:.2f}s')
