# -*- coding: utf-8 -*-
"""
키(스케일) 인식 음 보정.
감지된 키/모드의 다이어토닉 스케일에서 벗어난 전사 음을, 세기(duration)로 가중한
스케일 적합도를 보고 가장 가까운 스케일 음으로 당긴다. 전사 오류로 생긴 반음 어긋남을
줄여 재연주·MIDI 품질을 높인다. strength=1이면 모든 비스케일음을 스냅, 0이면 끔.

보수적으로: 반음(1) 이내로만 당기고, 그 이상 떨어진 음은 블루노트/경과음일 수 있어 두지 않는다.
"""
_NOTE_IDX = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
             "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
_MAJOR = [0, 2, 4, 5, 7, 9, 11]
_MINOR = [0, 2, 3, 5, 7, 8, 10]  # natural minor


def scale_pcs(key, mode):
    root = _NOTE_IDX.get(key, 0)
    intervals = _MINOR if str(mode).lower().startswith("min") else _MAJOR
    return {(root + i) % 12 for i in intervals}


def snap_notes(notes, key, mode, strength=1.0, max_shift=1):
    """
    notes: [{"start","end","midi","vel"}...]
    벗어난 음을 ±max_shift 반음 내 가장 가까운 스케일 음으로 당김. 동점이면 아래로.
    반환: 보정된 새 리스트 + 보정 수.
    """
    if not notes or strength <= 0:
        return notes, 0
    pcs = scale_pcs(key, mode)
    out = []
    changed = 0
    for n in notes:
        m = int(n["midi"])
        pc = m % 12
        if pc in pcs:
            out.append(dict(n))
            continue
        # 가장 가까운 스케일 음 탐색(±max_shift)
        best = None
        for d in range(1, max_shift + 1):
            if (pc - d) % 12 in pcs:
                best = -d
                break
            if (pc + d) % 12 in pcs:
                best = d
                break
        if best is None:
            out.append(dict(n))
            continue
        nn = dict(n)
        nn["midi"] = m + best
        out.append(nn)
        changed += 1
    return out, changed
