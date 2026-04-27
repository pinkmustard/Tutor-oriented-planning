# Tutor-oriented-planning / tutor_aop

## Project Overview
KCC 2026 실험 코드베이스. MATH-500 데이터셋에서 두 가지 튜터링 시스템을 비교:
- **Baseline**: 단일 튜터 LLM의 Socratic 멀티턴 (no planning layer)
- **AOP**: Meta-Tutor + Detector + Workers(diagnosis/tutor_move/retrieval) + Auditor 의 plan-then-execute 구조

각 episode 흐름:
`student.initial_solve → (정답이면 skip) → 멀티턴 튜터링 → student.independent_resolve → 채점`

## Reference Codebase
`../PedagogicalRL/src/classroom.py` — 튜터-학생 멀티턴의 형식 기준점.
이번 리팩토링에서 judge/reward 단계는 제외하고 **대화 형식과 stage-wise 배치 패턴만** 차용.

## Architectural Decisions

### 1. 대화 타입은 ATTEMPTED-only
PedagogicalRL의 두 타입(GUIDED / ATTEMPTED) 중 ATTEMPTED만 사용. 학생이 먼저 풀이를 내고 이후 튜터가 합류하는 흐름. 이 결정에 따라 dialogue는 항상 `[student(initial), tutor, student, tutor, ...]`로 시작.

### 2. Perspective rotation으로 진짜 multi-turn chat 사용
- **튜터**: 학생 발화 → `user`, 튜터 발화 → `assistant`
- **학생**: 학생 발화 → `assistant`, 튜터 발화 → `user`
- Flat transcript를 단일 user 메시지에 박아넣는 방식은 사용하지 않음.
- `independent_resolve`도 같은 chat 세션을 그대로 이어가고 마지막에 `STUDENT_FINAL_USER` user turn 한 번만 추가 (PedagogicalRL의 `GENERATE_SOLUTION` state 미러링).

### 3. 학생 측 system prompt는 multi-turn에서 통일
- `STUDENT_INITIAL_SYSTEM`: 초기 단일 풀이 전용
- `STUDENT_DIALOGUE_SYSTEM`: respond + independent_resolve **모두 동일하게** 사용 (`{problem}` 슬롯 포함)
- `STUDENT_FINAL_USER`: independent_resolve의 마지막 user turn — "대화 끝났으니 완전한 풀이 써라"

### 4. `initial_solution` 중복 제거 (튜터 측)
학생의 초기 풀이는 dialogue 첫 user 메시지로 이미 들어가므로 시스템 프롬프트에 또 박지 않음. 영향 받은 곳:
- `BASELINE_TUTOR_SYSTEM`, `META_TUTOR_AGENDA_USER`, `META_TUTOR_FINAL_USER`, `DIAGNOSIS_USER`에서 `{initial_solution}` 슬롯 제거
- `baseline_tutor.respond`, `meta_tutor.plan_agenda`/`generate_final`, `DiagnosisWorker.run`에서 인자 제거
- 단, JSONL 로그의 `initial_solution` 필드는 보존(분석 도구 호환).

### 5. Baseline은 stage-wise 배치 병렬화 (AOP는 아직 순차)
PedagogicalRL의 `Classroom.sample_conversations` 패턴 이식. `tutor_aop/classroom.py`에 구현.
- `ConvState` 상태머신: `START → STUDENT_INITIAL → TUTOR_TURN ↔ STUDENT_TURN → STUDENT_RESOLVE → END`
- 매 stage마다 같은 상태에 있는 conv만 모아 ThreadPoolExecutor로 동시 HTTP 요청 → vLLM이 서버 사이드에서 continuous batching
- 문항마다 종료 턴이 달라도 상태머신이 자연스럽게 active set에서 제외 → batch 크기가 턴마다 줄어듦
- AOP runner는 아직 문항-순차. 추후 동일 패턴으로 확장 가능.

### 6. vLLM 인프라는 그대로 유지
- 기존 OpenAI 호환 vLLM 서버 두 개(8000=tutor, 8001=student) 그대로
- `VLLMManager.ensure_active`의 sleep/wake는 stage 전환 시에만 1회 실제 swap (lock 내부 idempotent)
- `experiment.concurrency` (default 32) 로 thread-pool 너비 조절. vLLM `max_num_seqs` 와 정렬해서 설정.

## Key Files

| 파일 | 역할 |
|---|---|
| `classroom.py` | (신규) `BaselineConv` 상태머신 + `_run_parallel` + `run_baseline_batch` + `conv_to_log_row` |
| `baseline_runner.py` | 한 번 호출로 전체 데이터셋 stage-wise 처리. CLI 플래그 유지 |
| `runner.py` | AOP 순차 runner (현재 미병렬화) |
| `student.py` | `initial_solve` / `respond(problem, dialogue)` / `independent_resolve(problem, dialogue)` |
| `baseline_tutor.py` | `respond(problem, dialogue)` — perspective rotation으로 messages 구성 |
| `meta_tutor.py` | AOP의 plan_agenda / replan / generate_final / revise_final |
| `prompts/*.py` | 시스템/사용자 프롬프트 모음 |
| `vllm_manager.py` | tutor/student 두 서버 sleep/wake 관리 |

## Common Commands

```bash
# Baseline (단일 모델)
python -m tutor_aop.baseline_runner --tutor-model Qwen/Qwen2.5-7B-Instruct

# Baseline (병렬도 조정)
python -m tutor_aop.baseline_runner --concurrency 64

# Mock 스모크 테스트
python -m tutor_aop.baseline_runner --mock --num 2 --out tutor_aop/logs/smoke.jsonl
python -m tutor_aop.runner --mock --num 2 --out tutor_aop/logs/aop_smoke.jsonl

# AOP 실험
python -m tutor_aop.runner
```

## Logging Schema (JSONL, episode-level)
```
index, problem, gold_answer, level, subject,
initial_solution, initial_grade,
tutoring_needed, dialogue (if tutored),
turns: [{turn_idx, tutor_response, student_response, errors, ...}],
ended_by, post_tutoring_solution, post_tutoring_grade,
elapsed_sec, tutor_model, student_model, system
```
- `ended_by` 가능값: `skip_correct_initial` / `end_token` / `max_turns` / `error_in_turn_<n>` / `fatal_error_initial`
- AOP turn은 추가 키: `initial_agenda`, `detector_output`, `replan_agenda`, `executed_agenda`, `worker_outputs`, `draft_response`, `auditor_output`, `revised_response`, `final_tutor_response`

## Pending / Future Work
- AOP runner를 `classroom.py` 패턴으로 병렬화 (turn 안에서 agenda → detector → workers → final → auditor 각 단계 batched). 사용자가 baseline 실험 끝나는 대로 진행 예정.
