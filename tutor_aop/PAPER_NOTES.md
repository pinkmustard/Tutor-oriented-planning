# AOP 논문 정리 — 제안 방법 + 실험 설정

KCC 2026 투고용 논문의 §3 제안 방법, §4 실험 작성을 위한 참고 문서.
**이 문서의 사용 의도**: 다른 AI 도우미가 논문 본문을 함께 작성·검토할 때 본 연구
시스템의 구조와 설계 의도를 cold-start로 파악할 수 있도록 한 self-contained 요약.

> 구현·인프라 (배치 실행 패턴, GPU/서버 운영, 라이브러리 의존성 등)는 의도적으로 제외했다.

---

## 1. 문제 정의

수학 문항에 대해 학생이 오답을 산출한 상황에서, 튜터 LLM이 멀티턴 Socratic 대화로
학생의 추론을 교정해 학생 스스로 정답에 도달하도록 안내한다.

평가는 두 단계 정답률로 이루어진다:

- **Initial accuracy** — 튜터링 없이 학생이 푼 결과의 정답률
- **Post-tutoring accuracy** — 멀티턴 튜터링이 끝난 뒤 학생이 처음부터 다시
  완전한 풀이를 작성한 결과의 정답률

따라서 시스템의 핵심 효용 지표는 **rescue rate** = post_correct / tutored —
"튜터링이 오답을 정답으로 얼마나 끌어올리는가" 이다. 튜터가 정답을 직접 알려주는
것은 금지되어 있어, rescue rate는 곧 튜터의 *교육학적 scaffolding 품질*을 직접 측정한다.

---

## 2. 제안 방법: AOP (Agenda-Oriented Planning)

### 2.1 개요

AOP는 plan-then-execute 구조의 multi-agent 튜터 시스템이다. 단일 LLM이 매 턴
"다음 발화"를 즉흥적으로 생성하는 통상적 방식과 달리, AOP는 다음 단계로 작동한다.

1. **교육학적 agenda를 명시적으로 계획**한다. (어떤 진단·교수 결정·외부 지식이 필요한지)
2. agenda를 **여러 worker agent에게 위임**해 구조화된 진단·결정·검색 결과를 모은다.
3. **메타 튜터가 worker 출력을 합성**해 학생-facing utterance의 draft를 만든다.
4. **교육학적 auditor가 draft를 검수**하고, 필요하면 메타 튜터가 한 번 **revision**한다.
5. 통과된 utterance가 학생에게 전달되고, 학생이 응답하면 다음 턴이 시작된다.

플랜 계층을 두는 목적은 **튜터링 결정의 명시화**다 — 단일 LLM 호출이 암묵적으로
수행하던 "오류 찾기 + 교수 전략 선택 + 발화 작성"을 분리·검증·재계획 가능한
모듈로 풀어, 모델이 잘못된 결정을 했을 때 중간 단계에서 catch·recover할 수 있게 한다.

### 2.2 컴포넌트

#### Meta-Tutor
대화의 매 턴에서 세 가지 역할을 수행하는 조정자.
- **Agenda generation**: 어떤 worker를 어떤 sub-task로 호출할지 JSON 형식의 계획을 생성
- **Final synthesis**: worker 출력을 받아 학생-facing utterance draft를 작성
- **Revision**: auditor가 draft를 거부할 경우 feedback을 반영해 한 번 재작성

학생과 직접 대화하지 않으며, 시스템 내부 결정자(decision maker) 역할만 담당한다.

#### Plan Detector
Meta-Tutor의 agenda를 plan-level 두 속성에 대해 검사:
- **Completeness** — 이 턴에 의미 있는 튜터 발화를 구성하기에 sub-task가 충분한가
- **Non-redundancy** — 둘 이상의 sub-task가 같은 정보 요구를 중복하지 않는가

문제 발견 시 Meta-Tutor가 **최대 1회**의 replan을 통해 agenda를 수정한다.

#### Workers
agenda 항목에 따라 호출되는 세 종류의 전문 agent.

- **Diagnosis Worker** — 학생의 첫 오류 단계, 오류 유형(`arithmetic` /
  `algebraic` / `conceptual` / `procedural` / `notation`), 오개념(misconception),
  선수 지식 결손(prerequisite gap)을 짧은 영문 라벨로 출력. LaTeX·수식 verbatim
  복사는 금지(메타 튜터가 dialogue를 별도로 보기 때문에 라벨링만 수행).

- **Tutor Move Selector** — 다음 튜터 발화의 *유형*을 4지선다로 결정.
  - *Focus*: 다음 수학적 단계로 학생의 주의를 좁힘
  - *Probing*: 학생이 자기 추론을 설명·재검토하도록 질문
  - *Telling*: 막힌 한 가지 사실만 짧게 직접 설명 (드물게)
  - *Generic*: 사회·정서적 지원·격려·확인. 학생이 이미 corrected step을 산출한
    경우 자연스러운 confirmation 발화로 사용

- **Retrieval Worker** — 외부 개념 풀에서 정의·정리·선수 개념을 짧게 가져옴.
  특정 concept이 막힌 경우만 호출. 풀이 전체 retrieval은 금지.

#### Pedagogical Auditor
Meta-Tutor의 draft utterance를 학생에게 전달하기 전에 4가지 기준으로 검수.
**우선순위 순서**:

1. **premature_termination** (lead check) — 종료 토큰이 합당한 시점에 emit
   되었는가. 학생이 corrected step 또는 정답을 산출한 적이 없는데 종료 토큰이
   draft에 박혀 있으면 자동 fail.
2. **answer_leaked** — 최종 답·완전한 풀이 단계를 노출했는가
3. **socratic_style** — 강의식이 아니라 scaffolding 형태인가
4. **length_ok** — 분량이 적정한가 (~80 words 이내)

위 4가지를 **모두** 통과해야 `pedagogically_compliant=true`. 하나라도 실패하면
revision을 트리거.

### 2.3 한 턴의 실행 흐름

매 튜터 턴마다 다음 8단계로 진행:

1. **PLAN** — Meta-Tutor가 turn_idx, max_turns, dialogue를 보고 agenda 생성
2. **DETECT** — Plan Detector가 completeness / non-redundancy 검사
3. **REPLAN** *(필요 시 1회)* — 검출 결과를 반영해 agenda 수정
4. **EXECUTE** — agenda에 명시된 worker만 실행 (diagnosis / tutor_move / retrieval)
5. **GENERATE_FINAL** — Meta-Tutor가 worker 출력을 종합해 utterance draft 작성
6. **AUDIT** — Pedagogical Auditor가 dialogue + draft를 함께 보고 검수
7. **REVISE** *(auditor 거부 시 1회)* — Meta-Tutor가 draft 재작성
8. 통과된 utterance를 dialogue에 추가 → 종료 토큰 포함이면 대화 종료, 아니면 학생 턴

### 2.4 대화 흐름 (Conversation Flow)

PedagogicalRL의 ATTEMPTED 패턴을 차용 — 학생이 먼저 자기 풀이를 끝까지 시도하고,
그 풀이가 오답일 때만 튜터가 합류한다.

```
1. Student.initial_solve(problem)
2. 채점 → 정답이면 종료 (skip_correct_initial)
3. 오답이면 멀티턴 진입:
     for turn = 0..max_turns-1:
         Tutor.respond(problem, dialogue)   ← AOP 파이프라인
         if 종료 토큰: break
         Student.respond(problem, dialogue)
4. Student.independent_resolve(problem, dialogue) → 사후 풀이
5. 사후 풀이 채점
```

**Perspective rotation**: 튜터 측에서는 학생 발화를 user, 자기 발화를 assistant로
보고; 학생 측에서는 정반대로 본다. 이로써 양 모델 모두 자기가 assistant인
자연스러운 multi-turn chat을 진행한다 (전체 대화를 단일 user 메시지에 박는 flat
transcript 방식은 사용하지 않는다).

### 2.5 핵심 설계 결정

#### (a) Multi-turn Socratic 흐름의 명시화
초기 prompt 설계에서 튜터는 첫 턴에 한 번 probing 질문을 던지고 곧바로
종료 토큰을 emit하는 *turn-1 collapse* 패턴을 보였다 — 학생이 응답할 기회조차 없이
대화가 끝나버린다. 이를 다음 흐름으로 풀어내도록 했다.

```
Q1 (오류 표면화: probing) → 학생 응답
→ Q2 (개념 redirect: focus / telling) → 학생 응답
→ Q3 (확인: generic confirmation) → 정당한 종료
```

**Absolute Termination Rule**: nudge prompt 최상단에 turn_idx / max_turns를
노출하고, `turn_idx == 0`인 경우 종료 토큰 emit을 *절대 금지*한다 — turn 0의
대화는 학생의 초기 오답 한 메시지뿐이므로 "이해 도달" 조건이 정의상 충족 불가능하다.
`turn_idx > 0`에서는 학생의 가장 최근 메시지가 **explicitly corrected step 또는
정답**을 포함할 때만 종료를 허용한다.

이 룰은 (1) Meta-Tutor의 final/revise nudge에 명시, (2) Auditor의
premature_termination 검출 fast-path로 이중화, (3) 위 두 안전장치가 모두 실패한
edge case에서도 turn 0 발화의 종료 토큰을 deterministic하게 제거하는 최종
보호층까지, 세 단으로 보장된다.

#### (b) Turn-Progression-Aware Planning
Meta-Tutor의 agenda 생성 prompt가 turn_idx / max_turns / dialogue 진행도를 보고
턴 단계별로 다른 agenda를 짜도록 가이드:

| 단계 | 조건 | 전형 agenda | 전형 move |
|---|---|---|---|
| Early | turn 0, 튜터 발화 없음 | `[diagnosis, tutor_move]` | Probing |
| Middle | 학생이 이전 튜터 턴에 응답했지만 corrected step 미산출 | `[tutor_move]` 또는 새 오류 발생 시 `[diagnosis, tutor_move]` | Focus / Telling |
| Late | 학생이 corrected step 또는 정답 산출 | `[tutor_move]` (diagnosis 호출 금지) | Generic (confirmation) |

후반 턴에서 불필요한 diagnosis 호출이 사라지고, confirmation 발화가 자연스럽게
도출되어 정당한 종료가 가능해진다.

#### (c) Tutor Move Selector의 Generic 확장
Generic move 선택 조건을 *학생의 좌절·안심 요청*에서 *학생이 explicitly corrected
step 또는 정답을 산출한 경우*까지 확장. 후반 턴 confirmation의 trigger를 명시하여
Q3 단계와 정합되게 한다.

#### (d) Auditor의 Dialogue-aware 검수
Auditor의 입력에 dialogue 전체를 포함시켜 "학생이 corrected step을 냈는지"를
직접 평가 가능하게 했다. 이는 premature_termination 판정의 필수 정보다.
또한 premature_termination을 4가지 기준 중 *1번*(맨 앞)으로 배치하고
fast-path 룰("학생 메시지 1개뿐인 대화에서 종료 토큰 들어간 어떤 draft든 자동
fail")을 prompt 앞쪽에 명시해 모델 attention을 우선적으로 끌어온다.

---

## 3. 베이스라인

### 3.1 Single-Tutor Baseline
플랜 계층 없는 vanilla 튜터. 매 튜터 턴 단일 LLM 호출로 다음을 입력한다:

- **system prompt**: Socratic guideline + 문항
- **dialogue**: perspective-rotated chat history
- **final user nudge**: brevity / Socratic 제약 / 종료 토큰 조건 재상기

`<end_of_conversation>` 종료 메커니즘과 동일한 Socratic 가이드라인을 사용한다.

### 3.2 Bridge Baseline
사전 연구 (Bridging Tutoring 계열)의 task-decomposition 구조를 차용 — 매 튜터
턴마다 세 가지 sub-task를 수행:

- **Task 1 (Error Inference)** — 학생 오류를 사전 정의된 7개 유형 + 자유 기술 중
  하나로 분류 (e.g., careless mistake, misinterpretation, conceptual gap, …)
- **Task 2 (Strategy + Intention)** — 11개 교정 전략 (질문 / 힌트 / 워크드 예제 / …)
  과 11개 의도 (동기 부여 / 진단 / 오해 해소 / …) 중 각각 하나를 선택
- **Task 3 (Response Generation)** — Task 1·2의 출력을 받아 실제 학생-facing
  utterance를 생성

Task 1·2의 prompt는 원논문 verbatim. Task 3의 system prompt만 비교 공정성을 위해
Single-Tutor Baseline과 동일한 Socratic 가이드라인 + `<end_of_conversation>`
종료 메커니즘으로 통일했다. 이로써 세 시스템 (Single, Bridge, AOP)이 동일한
**pedagogical guardrail**과 **종료 affordance**를 공유하며, 차이는 오직 발화
생성에 사용된 **인지 구조**(plain → 3-task decomposition → planning + worker
delegation + audit)에서만 발생한다.

### 3.3 세 시스템 비교

| 시스템 | 인지 구조 | 한 튜터 턴의 LLM 호출 수 |
|---|---|---|
| Single-Tutor | 단일 호출 | 1 |
| Bridge | Task1 → Task2 → Task3 (Task1·2 독립) | 3 |
| AOP (제안) | Plan → Detect → (Replan) → Workers → Final → Audit → (Revise) | 가변 (보통 5–7) |

---

## 4. 실험 설정

### 4.1 모델

- **Tutor** (실험에 따라 교체):
  - Qwen3-4B-Instruct-2507
  - Qwen2.5-7B-Instruct
- **Student simulator**: Llama-3.1-8B-Instruct (모든 실험에서 고정)

세 시스템 모두 같은 두 튜터 모델, 같은 학생 모델을 공유하므로 시스템 간 차이는
architecture에서만 발생한다.

### 4.2 데이터셋

- **MATH-500** — `HuggingFaceH4/MATH-500`, test split (500문항)
- **Big-Math-RL-Verified-Filtered** — `rd211/Big-Math-RL-Verified-Filtered`, test split
  (광범위한 수학 도메인의 verified 풀이 데이터셋, 500문항 사용)

### 4.3 통제 변인 — Fixed Initial Solutions

서로 다른 튜터 모델·시스템을 비교할 때 **학생의 초기 풀이가 매 실행마다 달라지면**
사후 정확도 비교가 학생의 stochasticity에 오염된다. 이를 막기 위해 학생의
`initial_solve` 결과를 데이터셋 단위로 **사전 한 번 생성·고정**하여 모든 실험이
동일한 초기 풀이 분포를 사용하도록 했다.

이 통제 덕분에:
- 사후 정확도 차이는 오직 **튜터링 품질**의 차이로 해석된다
- 튜터링이 필요했던 문항 (`tutoring_needed=true`)의 집합이 모든 실험에서 동일하다
- rescue rate가 시스템 간 직접 비교 가능한 정량 지표가 된다

### 4.4 메트릭

- **Initial accuracy** — 학생의 초기 풀이 정답률 *(모든 실험에서 같음, 통제 변인)*
- **Post-tutoring accuracy** — 멀티턴 튜터링 후 사후 풀이 정답률
- **Rescue rate** = post_correct / tutored — 가장 직접적 튜터링 효과 지표
- **Total accuracy** = (initial_correct + post_correct) / total
- **Multi-turn engagement** — 평균 turn 수, turn 분포, `ended_by` 분포
  (`end_token` / `max_turns` / `error`)

### 4.5 하이퍼파라미터

| 항목 | 값 |
|---|---|
| `max_turns` (튜터 턴 상한) | 5 |
| `max_replan` | 1 |
| `max_revision` | 1 |
| 튜터 temperature | 0.7 |
| 학생 temperature | 0.6 |
| 튜터 발화 분량 상한 | 60 단어 / 2–3 문장 |
| 학생 응답 분량 상한 | 동일 (응답 collapse 방지) |

### 4.6 채점

`<answer>\\boxed{...}</answer>` 형태의 사후 풀이에서 boxed 답을 추출하고,
gold answer와 LaTeX 정규화 + sympy `simplify`를 통한 동등성 비교로 정답 여부를
판정한다.

---

## 5. (Optional Ablation) AOP Prompt 강화 과정

본 연구의 prompt 설계는 다음과 같이 진화했다. 논문 본문에 ablation으로 활용 가능.

### v1 — 초기 설계
Meta-Tutor의 종료 조건을 *"학생이 이해한 것 같으면 종료"*라는 느슨한 자연어로
표현. **결과**: 튜터가 첫 턴에 한 번 probing 질문을 던지고 곧바로 종료 토큰을
emit하는 *turn-1 collapse*가 100% 발생. 멀티턴 Socratic 흐름 자체가 사라지고
학생은 응답 기회조차 얻지 못한 채 사후 풀이로 진입.

### v2 — Strict criterion 도입
Meta-Tutor의 system prompt에 multi-turn 단계 가이드와 strict termination 조건
("학생이 corrected step을 산출했을 때만 종료") 추가. **결과**: 296/318 (93%)
가 *여전히* turn 1 종료. 원인 분석:
- system prompt 후행 항목이라 모델 attention 부족
- Auditor의 premature_termination 검출이 false-negative 75% — 검출 자체 실패
- 검출에 성공해도 revise 출력이 다시 종료 토큰 박음 (100%)

### v3 — 세 단 보강 동시 적용 *(현재 시스템)*
1. **Source-level**: Meta-Tutor의 final / revise nudge prompt 최상단에 turn_idx
   노출 + Absolute Termination Rule (turn 0 절대 금지)을 *맨 앞*에 명시. 모델이
   prompt를 읽기 시작하는 시점에 가장 강한 제약을 본다.
2. **Detection 강화**: Auditor의 premature_termination 검사를 4번째 → *1번째*
   기준으로 이동. fast-path 룰 ("대화에 학생 메시지 1개뿐이면 종료 토큰 박힌
   draft는 자동 fail, do NOT second-guess")과 false-negative 패턴 example을
   prompt 앞쪽에 추가.
3. **Safety net (deterministic)**: 위 두 prompt-level 안전장치가 모두 실패해도
   turn 0 발화의 종료 토큰을 algorithmically 제거. 학생이 최소 한 번은 응답
   하도록 물리적으로 보장.

**v3 효과** (smoke 표본 기준): turn-1 collapse 비율 100% → 0%, multi-turn
Socratic 진행 정상화. 본실험에서 rescue rate가 baseline 대비 어떻게 변하는지가
이 연구의 핵심 정량적 결론.

---

## 6. 시스템·코드 명명 규약 (참고용)

논문 작성 시 시스템 표기 일관성을 위해 사용된 약속.

- **AOP** — Agenda-Oriented Planning. 본 연구의 제안.
- **Bridge** — task1/task2/task3 baseline.
- **Single-Tutor Baseline** — vanilla 단일-호출 baseline.
- **Tutor model** — 학생을 가르치는 LLM. 본 연구에서는 Qwen 계열 두 모델 비교.
- **Student model** — 풀이·응답을 시뮬레이션하는 LLM. Llama-3.1-8B-Instruct 고정.
- **Episode** — 한 문항에 대한 한 번의 (initial_solve → 튜터링 → independent_resolve)
  완전 시퀀스.
- **Turn** — 튜터-학생 대화 한 라운드 (튜터 발화 + 학생 응답).
- **Rescue** — `tutoring_needed`였던 episode가 사후 풀이에서 정답으로 전환되는 사건.
