# 스템 리메이크 (Stem Remake)

업로드한 음원을 스템(보컬·드럼·베이스·기타외)으로 분리하고, 파트를 다시 연주해 새 버전을 만드는 도구. STUDIO ig.

## 동작
업로드 → 스템 분리 → 분석(BPM·키) → 파트별 처리(재연주 / 정제 유지) → 재믹스·마스터링 → 결과

- 연주곡 모드: 베이스·드럼 재연주, 기타외 정제 유지, 보컬 제외
- 보컬곡 모드: 보컬 정제 유지, 베이스·드럼 재연주, 기타외 정제 유지

## 백엔드 2단계
- demucs: torch + Demucs(htdemucs). 고품질. GitHub Actions 빌드에 포함.
- lite: librosa DSP 분리. torch 없이 동작(개발/폴백).

## 실행(소스)
    pip install -r requirements-lite.txt   # 개발(lite)
    python app/main.py                     # 브라우저 자동 오픈

## 단독 실행파일
GitHub Actions(build-windows)가 Windows용 단독 실행 폴더를 빌드해 zip 아티팩트로 제공.
압축을 풀고 StemRemake.exe 더블클릭 → 브라우저 자동 오픈.

## 모드
- 리메이크(기본): 분리→재연주→마스터로 새 버전 생성
- DAW 내보내기(모드 2): 스템 + 파트별 MIDI 패키지 → 큐베이스·로직에서 import해 마무리 (DAW 무관)
- VST 악기 모드(모드 3): 보유한 .vst3(Win) / AU·VST3(Mac) 플러그인으로 파트 재연주. 로직 내장 악기는 외부 호출 불가

## 빌드 대상
- Windows: .github/workflows/build-windows.yml → StemRemake-windows.zip
- macOS:   .github/workflows/build-macos.yml → StemRemake-macos.zip
