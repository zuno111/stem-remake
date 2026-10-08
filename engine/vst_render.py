# -*- coding: utf-8 -*-
"""
VST/AU 악기 호스팅 렌더(모드 3). Pedalboard로 사용자가 보유한 플러그인을 직접 불러
전사한 MIDI를 그 악기 음색으로 렌더한다.

- Windows: .vst3
- macOS: .vst3 또는 .component(AU)
로직 내장 악기는 외부 호스트로 불러올 수 없으므로 대상 아님(서드파티 AU/VST는 가능).

이 모듈은 실제 플러그인이 설치된 사용자 PC에서 동작한다. 플러그인이 없으면 호출부에서
numpy 합성(synth.py)으로 폴백한다.
"""
import numpy as np


def load_instrument(plugin_path, preset_path=None):
    """플러그인 로드. 실패 시 예외."""
    from pedalboard import load_plugin
    inst = load_plugin(plugin_path)
    if preset_path:
        try:
            inst.load_preset(preset_path)  # .vstpreset / .fxp 등
        except Exception:
            pass
    return inst


def _notes_to_messages(notes, is_drum=False, program=None):
    """노트 dict 리스트 → mido 메시지(절대 시간 초) 리스트. program 지정 시 패치 변경."""
    import mido
    msgs = []
    ch0 = 9 if is_drum else 0
    if program is not None and not is_drum:
        msgs.append(mido.Message("program_change", program=int(program), channel=ch0, time=0.0))
    for n in notes:
        ch = 9 if is_drum else 0
        on = mido.Message("note_on", note=int(n["midi"]), velocity=int(n["vel"]),
                          channel=ch, time=float(n["start"]))
        off = mido.Message("note_off", note=int(n["midi"]), velocity=0,
                           channel=ch, time=float(max(n["end"], n["start"] + 0.05)))
        msgs.append(on)
        msgs.append(off)
    msgs.sort(key=lambda m: m.time)
    return msgs


def render_notes_vst(notes, sr, duration, plugin_path, preset_path=None, is_drum=False, program=None):
    """
    전사 노트를 VST/AU 악기로 렌더. (samples,) 모노 float32 반환.
    program: MIDI 프로그램(패치) 번호. 멀티팀브럴 악기(HALion 등) 음색 선택용.
    실패 시 예외를 올리므로 호출부에서 synth 폴백.
    """
    inst = load_instrument(plugin_path, preset_path)
    msgs = _notes_to_messages(notes, is_drum=is_drum, program=program)
    audio = inst(msgs, duration=float(duration) + 0.5, sample_rate=float(sr))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # (channels, samples) → mono
        audio = audio.mean(axis=0)
    # 길이 맞추기
    target = int(duration * sr)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return audio


def available():
    """이 환경에서 VST 호스팅이 가능한지(pedalboard load_plugin + mido)."""
    try:
        from pedalboard import load_plugin  # noqa: F401
        import mido  # noqa: F401
        return True
    except Exception:
        return False
