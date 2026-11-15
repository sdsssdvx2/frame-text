# Hard Subtitle Extractor (HSE) — Claude 최종본 2026-11-15

영상에 박힌 자막(하드자막)을 무료 로컬 OCR(PP-OCRv5, RapidOCR)로 읽어 **SRT/ASS 자막 파일**로 만드는 Windows 프로그램입니다.
별도 자막(내장 자막 트랙·사이드카 SRT·공개 웹 자막)이 있으면 그것을 먼저 쓰고, 없을 때만 OCR을 합니다.
유료 API·구독이 필요 없고, 인터넷은 설치할 때(모델·런타임 다운로드)와 웹 소스를 쓸 때만 사용합니다.

## 할 수 있는 것

| 모드 (GUI 맨 위) | 설명 |
| --- | --- |
| 사이트 회차 주소 | 회차 URL → 별도 자막 우선 → 없으면 한국어 하드자막 OCR. 현재 사이트 플레이어(FirePlayer)는 **PC에 설치된 Chrome/Edge**로 화면을 재생해 OCR합니다(브라우저 프로필·쿠키는 사용하지 않음). 24분 회차 기준 약 50분 소요(실측 1회). |
| 내 영상 1개 | 로컬 영상 → 내장 자막/옆 SRT 우선 → 없으면 하드자막 OCR(한·영·일·중 선택). |
| 내 영상 최대 50개 일괄 | 여러 파일을 순서대로 처리, 기존 결과는 덮어쓰지 않고 건너뜀/재개. |
| 웹 자막 우선 (YouTube·WeTV 등) | yt-dlp로 공개 자막 트랙을 받고, 없으면(선택) 영상 OCR. |

선택 기능: 별도 고화질 영상과 자동 싱크, 1프레임 OCR 미리보기, Subtitle Edit/메모장 교정 연결 등은 **"고급·실험 도구 표시"** 를 체크하면 보입니다.
"실험:" 이 붙은 옵션은 기본 꺼짐이며, 켜면 일부 영상에서 자막이 빠질 수 있습니다.

## 설치 (1회, 관리자 권한 불필요)

1. ZIP을 아무 폴더(예: 바탕 화면의 `HSE` 폴더)에 **전체 압축 해제**합니다.
2. `Install-HSE.cmd` 를 더블클릭합니다. (5~15분, 약 400MB 다운로드: 개인용 Python 3.12, 패키지, OCR 모델)
3. 끝나면 바탕화면/시작 메뉴에 **Hard Subtitle Extractor** 바로가기가 생깁니다.

설치 내용
- 앱: `%LOCALAPPDATA%\Programs\HardSubtitleExtractor`
- 런타임(uv, Python): `%LOCALAPPDATA%\Programs\HardSubtitleExtractor-runtime`
- 모든 다운로드(uv, FFmpeg, yt-dlp, SeConv, OCR 모델)는 **SHA-256 고정값으로 검증**합니다. 불일치 시 설치 중단.
- 새 버전을 옆 폴더에 먼저 만들고 **자체검사가 통과해야만** 교체합니다. 실패하면 기존 설치는 그대로입니다.
- 변경 없이 계획만 보기: `powershell -ExecutionPolicy Bypass -File install.ps1 -DryRun`

## 사용

1. 바로가기 실행 → 모드 선택 → URL 또는 영상 선택 → 결과 파일 기준 경로 확인.
2. `자막 추출 시작`. 진행 로그와 진행 막대가 표시되고, `안전하게 취소` 로 중단할 수 있습니다.
3. 결과: `<기준경로>.srt`, `<기준경로>.ass` (같은 이름이 이미 있으면 덮어쓰지 않고 오류로 알려줌).
4. 처음 한 번은 `고급·실험 도구 표시` → `자체검사` 로 FFmpeg·모델 상태를 확인해도 좋습니다.

## 업데이트

새 ZIP을 받아 압축 해제 후 `Install-HSE.cmd` 를 다시 실행합니다. 기존 모델은 재검증 후 재사용하고, 이전 버전은 `HardSubtitleExtractor.backup-날짜` 로 보관됩니다.

## 되돌리기 / 제거

- 이전 버전으로 복원: `Restore-Previous-HSE.cmd`
- 제거: `Uninstall-HSE.cmd` (앱 폴더와 바로가기만 제거). 런타임까지 지우려면 `uninstall.ps1 -RemoveRuntime`.
- 절대 지우지 않는 것: 영상·자막 결과물, `%LOCALAPPDATA%\HardSubtitleExtractor`(yt-dlp 갱신본 등 사용자 데이터), 백업 폴더(필요 없으면 직접 삭제).

## 사용자가 직접 해야 할 1회성 작업

- `Install-HSE.cmd` 실행 1회.
- EpSite 모드를 쓰려면 Chrome 또는 Edge가 설치되어 있어야 합니다(대부분 이미 설치됨). 로그인은 필요 없습니다.

## 알려진 한계

- OCR 문구 정확도는 영상마다 다릅니다. 장편 1회 완주 실측은 있으나 정답 자막 대비 정확도는 아직 측정되지 않았습니다. 중요한 작품은 결과 SRT를 한 번 훑어보세요(`무료 교정` 버튼).
- 처리 속도: 3분 영상 OCR에 수 분~십수 분(CPU 성능에 따라). 기본으로 무료 OpenVINO 가속을 쓰고, 문제가 생기면 자동으로 기본 엔진(ONNX Runtime)으로 전환합니다.
- DRM·로그인·CAPTCHA 우회는 하지 않습니다.
