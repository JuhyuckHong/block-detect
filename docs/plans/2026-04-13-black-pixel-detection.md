# Black Pixel Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Dropbox에서 하루치 이미지를 내려받아 검은 픽셀 비율 기반으로 `abnormal` / `normal` 분류와 요약 리포트를 수행하는 첫 배치 버전을 만든다.

**Architecture:** CLI가 설정과 실행 인자를 받고, 파이프라인이 Dropbox 조회/다운로드와 로컬 분류를 오케스트레이션한다. 분류기는 grayscale 통계로 `dark_ratio`와 `mean_brightness`를 계산해 규칙 기반 판정을 수행하고, 결과는 일자별 요약으로 저장한다.

**Tech Stack:** Python 3.11, setuptools, Dropbox SDK, unittest, Pillow

---

### Task 1: Add classifier test coverage from sample images

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

테스트를 추가한다.

```python
def test_classifier_marks_ab_samples_abnormal(self):
    ...

def test_classifier_marks_normal_sample_normal(self):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL because the placeholder classifier returns `unknown`

**Step 3: Write minimal implementation**

아직 구현하지 않는다. 이 태스크는 실패하는 테스트를 먼저 고정하는 단계다.

**Step 4: Run test to verify it fails correctly**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL with mismatched label assertions

**Step 5: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: define classifier behavior from sample images"
```

### Task 2: Implement black-pixel classifier

**Files:**
- Modify: `src/block_detect/classifier.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

Task 1의 테스트를 기준으로 유지한다. 필요하면 이유 문자열과 점수 범위 검증을 한 줄 더 추가한다.

```python
self.assertEqual(result.label, "abnormal")
self.assertGreater(result.score, 0.0)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL on sample classification

**Step 3: Write minimal implementation**

- `Pillow` 의존성을 추가한다
- grayscale 변환
- `dark_ratio`, `mean_brightness` 계산
- 기본 임계값을 `ab-1`이 `abnormal`로 통과하도록 맞춘다
- 결과 이유 문자열에 통계를 남긴다

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli -v`
Expected: PASS for classifier sample tests

**Step 5: Commit**

```bash
git add pyproject.toml src/block_detect/classifier.py tests/test_cli.py
git commit -m "feat: add black pixel classifier"
```

### Task 3: Expand settings for pipeline execution inputs

**Files:**
- Modify: `src/block_detect/config.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

설정 객체에 다음 값이 들어가는 테스트를 추가한다.

```python
self.assertEqual(settings.dark_threshold, 32)
self.assertEqual(settings.dropbox_day_template, "{date}")
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL because new settings fields do not exist

**Step 3: Write minimal implementation**

- 분류 임계값 설정 필드 추가
- 대상 날짜용 Dropbox 경로 템플릿 추가
- 기본값 정의

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/block_detect/config.py tests/test_cli.py
git commit -m "feat: add execution settings"
```

### Task 4: Add pipeline execution test with fake Dropbox client

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

fake Dropbox client를 만들어 하루치 파일 목록과 다운로드 결과를 제공하고, 파이프라인 실행 결과가 기대 요약을 반환하는 테스트를 추가한다.

```python
def test_pipeline_runs_day_batch(self):
    ...
    self.assertEqual(summary.abnormal_count, 5)
    self.assertEqual(summary.normal_count, 1)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL because the pipeline cannot run a day batch yet

**Step 3: Write minimal implementation**

아직 구현하지 않는다. 테스트 실패를 고정한다.

**Step 4: Run test to verify it fails correctly**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL with missing method or attribute errors around pipeline execution

**Step 5: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: define daily pipeline behavior"
```

### Task 5: Implement Dropbox-backed day pipeline

**Files:**
- Modify: `src/block_detect/pipeline.py`
- Modify: `src/block_detect/dropbox_client.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

Task 4 테스트를 유지한다. 필요하면 빈 목록 처리도 한 건 추가한다.

```python
self.assertEqual(summary.processed_count, 0)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL on missing pipeline behavior

**Step 3: Write minimal implementation**

- `PipelineSummary`를 `abnormal` 기준으로 정리
- 대상 날짜 문자열을 받아 Dropbox 경로 생성
- 목록 조회, 이미지 다운로드, 분류, 요약 실행 메서드 추가
- Dropbox 클라이언트에 이미지 확장자 필터와 자격 증명 검증 구조 추가

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli -v`
Expected: PASS for pipeline tests

**Step 5: Commit**

```bash
git add src/block_detect/pipeline.py src/block_detect/dropbox_client.py tests/test_cli.py
git commit -m "feat: add Dropbox day pipeline"
```

### Task 6: Add CLI execution flow and report output

**Files:**
- Modify: `src/block_detect/cli.py`
- Modify: `src/block_detect/pipeline.py`
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

CLI가 날짜 인자를 받고 파이프라인 실행 결과를 반환하는 테스트를 추가한다.

```python
def test_main_runs_batch_for_date(self):
    exit_code = cli.main(["--date", "2026-04-13"])
    self.assertEqual(exit_code, 0)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_cli -v`
Expected: FAIL because CLI does not accept batch execution arguments

**Step 3: Write minimal implementation**

- `--date`
- `--dropbox-path`
- `--prepare-only`
- 임계값 오버라이드 인자
- 실행 후 요약 출력
- 리포트 파일 저장

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_cli -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/block_detect/cli.py src/block_detect/pipeline.py tests/test_cli.py
git commit -m "feat: wire CLI batch execution"
```

### Task 7: Verify end-to-end behavior

**Files:**
- Test: `tests/test_cli.py`

**Step 1: Run the full test suite**

Run: `python3 -m unittest -v`
Expected: PASS

**Step 2: Run CLI help**

Run: `PYTHONPATH=src python3 -m block_detect.cli --help`
Expected: PASS with date and threshold options visible

**Step 3: Note environment dependency**

- 실제 Dropbox 호출 검증에는 유효한 자격 증명이 필요하다
- 네트워크 환경에서는 수동 검증 케이스를 별도로 수행한다

**Step 4: Commit**

```bash
git add docs/plans/2026-04-13-black-pixel-detection*.md
git commit -m "docs: add black pixel detection design and plan"
```
