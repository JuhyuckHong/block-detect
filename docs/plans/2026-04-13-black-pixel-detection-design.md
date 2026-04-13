# Black Pixel Detection Design

## Goal

Dropbox에서 하루치 이미지를 가져와 로컬에서 분류하고, 검은색 비율 기반 규칙으로 `abnormal` / `normal` 판정을 수행하는 1차 배치 흐름을 만든다.

## Current State

- CLI는 작업 디렉터리만 생성한다.
- 파이프라인은 로컬 이미지 분류/요약만 가능하다.
- Dropbox 클라이언트는 자격 증명 로딩만 구현되어 있고 실제 목록 조회/다운로드는 비어 있다.
- 분류기는 항상 `unknown`을 반환하는 플레이스홀더다.

## Detection Rule

1차 판정은 이미지 전체를 grayscale로 변환한 뒤 두 개의 통계를 계산한다.

- `dark_ratio`: 밝기가 `dark_threshold` 이하인 픽셀 비율
- `mean_brightness`: 전체 평균 밝기

판정 규칙:

- `dark_ratio >= dark_ratio_threshold`
- `mean_brightness <= mean_brightness_threshold`

위 두 조건을 모두 만족하면 `abnormal`, 아니면 `normal`

임계값 튜닝 기준:

- 최소 기준 샘플은 `tests/ab-1.jpg`
- `ab-1`이 반드시 `abnormal`로 판정되도록 임계값을 잡는다
- 그 상태에서 `tests/normal.jpg`가 `normal`을 유지하는 범위로 조정한다

출력에는 다음 정보를 남긴다.

- 라벨
- 점수: `dark_ratio`
- 이유 문자열: `dark_ratio`, `mean_brightness`, 임계값 요약

## Pipeline Shape

배치는 다음 순서로 동작한다.

1. 설정 로드 및 작업 디렉터리 준비
2. 대상 일자용 Dropbox 경로 계산
3. Dropbox에서 이미지 목록 조회
4. 로컬 inbox 또는 working 디렉터리로 다운로드
5. 각 이미지에 대해 검은색 비율 기반 판정 수행
6. 일자별 요약 JSON 또는 텍스트 리포트 생성
7. CLI에 총 처리 수와 판정 요약 출력

## Dropbox Scope

1차 구현 범위는 다음으로 제한한다.

- 폴더 내 파일 목록 조회
- 이미지 확장자만 필터링
- 대상 디렉터리 다운로드

업로드는 필수 범위에서 제외하고, 나중에 확장 가능한 형태로 남긴다.

## Error Handling

- 자격 증명이 없으면 명확한 예외 메시지를 반환한다.
- Dropbox 대상 폴더가 없거나 비어 있으면 0건 처리로 종료한다.
- 개별 다운로드 실패는 예외를 올려 배치를 실패 처리한다.
- 이미지 판독 실패는 해당 파일을 `unknown` 대신 실패로 볼지 추후 확장 가능하게 하되, 1차 구현에서는 예외로 처리한다.

## Testing

- `tests/ab-1.jpg` ~ `tests/ab-5.jpg`는 `abnormal`
- `tests/normal.jpg`는 `normal`
- CLI는 대상 날짜, Dropbox 루트, 임계값 인자를 받을 수 있어야 한다.
- Dropbox 네트워크 호출은 테스트에서 fake client로 대체한다.

## Notes

- 현재 작업 폴더는 Git 저장소가 아니므로 문서 커밋 단계는 수행할 수 없다.
