# block-detect

하루치 촬영 이미지를 Dropbox에서 가져와 `blocked` / `normal`로 분류하는 배치 프로젝트 스켈레톤입니다.

## 목표

- Dropbox에서 하루치 이미지 수집
- 로컬 작업 디렉터리 정리
- 차단 판정 로직 연결
- `blocked` / `normal` 배치 분류

## 구조

```text
block-detect/
├── src/block_detect/
│   ├── cli.py
│   ├── classifier.py
│   ├── config.py
│   ├── dropbox_client.py
│   └── pipeline.py
├── tests/
│   └── test_cli.py
├── data/
│   ├── inbox/
│   ├── samples/
│   └── working/
├── reports/
└── secrets/
```

## 빠른 시작

```bash
cd /home/jh/bmotion/block-detect
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python -m block_detect.cli --help
```

## Dropbox 인증

다음 환경 변수 중 하나의 조합을 사용할 수 있게 뼈대를 잡아두었습니다.

- `DROPBOX_APP_KEY`
- `DROPBOX_APP_SECRET`
- `DROPBOX_REFRESH_TOKEN`
- `DROPBOX_ACCESS_TOKEN`
- `DROPBOX_ACCOUNT_INFO_FILE`

## 1차 범위

- Dropbox 폴더에서 하루치 이미지 목록 조회
- 대상 이미지 로컬 다운로드
- 이미지별 `blocked` / `normal` 판정
- 일자 기준 결과 요약 출력

## 현재 상태

- 실제 이미지 판정 알고리즘은 아직 구현되지 않았습니다.
- Dropbox 목록 조회/다운로드도 메서드 시그니처만 정의된 상태입니다.
