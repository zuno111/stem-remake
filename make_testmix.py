# -*- coding: utf-8 -*-
"""파이프라인 검증용 합성 믹스 생성(베이스+코드+드럼+멜로디)."""
import numpy as np, soundfile as sf

sr = 44100
bpm = 100
beat = 60 / bpm
bars = 4
dur = bars * 4 * beat
n = int(dur * sr)
t = np.arange(n) / sr
mix = np.zeros(n)


def note(midi, start, length, amp, wave="sine"):
    f = 440 * 2 ** ((midi - 69) / 12)
    s = int(start * sr); ln = int(length * sr)
    tt = np.arange(ln) / sr
    if wave == "saw":
        w = 2 * (tt * f % 1) - 1
    else:
        w = np.sin(2 * np.pi * f * tt)
    env = np.exp(-tt * 1.5)
    seg = w * env * amp
    e = min(s + ln, n)
    mix[s:e] += seg[: e - s]


# 베이스(한 박마다)
bass_line = [40, 40, 45, 43]
for bar in range(bars):
    for b in range(4):
        note(bass_line[b], (bar * 4 + b) * beat, beat * 0.9, 0.35, "saw")
# 코드 패드(마디마다 3음)
chords = [[52, 55, 59], [52, 57, 60], [53, 57, 60], [50, 55, 59]]
for bar in range(bars):
    for m in chords[bar]:
        note(m, bar * 4 * beat, beat * 3.8, 0.12, "saw")
# 멜로디(반 박)
mel = [64, 67, 69, 67, 64, 62, 60, 62]
for bar in range(bars):
    for i, mm in enumerate(mel):
        note(mm, (bar * 4 + i * 0.5) * beat, beat * 0.4, 0.2, "sine")
# 드럼: 킥(1,3박) 스네어(2,4박) 하이햇(매 반박)
for bar in range(bars):
    base = bar * 4 * beat
    for b in [0, 2]:
        s = int((base + b * beat) * sr); k = np.sin(2 * np.pi * np.linspace(110, 45, int(0.2 * sr)) * np.arange(int(0.2 * sr)) / sr)
        k *= np.exp(-np.arange(int(0.2 * sr)) / sr * 18); e = min(s + len(k), n); mix[s:e] += k[: e - s] * 0.5
    for b in [1, 3]:
        s = int((base + b * beat) * sr); ln = int(0.15 * sr)
        sn = np.random.randn(ln) * np.exp(-np.arange(ln) / sr * 22); e = min(s + ln, n); mix[s:e] += sn[: e - s] * 0.3
    for b in np.arange(0, 4, 0.5):
        s = int((base + b * beat) * sr); ln = int(0.04 * sr)
        h = np.random.randn(ln) * np.exp(-np.arange(ln) / sr * 90); e = min(s + ln, n); mix[s:e] += h[: e - s] * 0.12

mix = mix / (np.max(np.abs(mix)) or 1) * 0.9
sf.write("io/test_mix.wav", mix.astype(np.float32), sr)
print(f"test_mix.wav 생성: {dur:.1f}s, {sr}Hz, BPM {bpm}")
