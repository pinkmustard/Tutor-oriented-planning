# tutor_aop — Detector-guided, AOP-inspired Math Tutoring

Light-weight reimplementation of the AOP (Agent-Oriented Planning) skeleton adapted
to math tutoring. The Meta-Tutor first produces a pedagogical **agenda**, which is
checked by a **Plan Detector** (replan x1) and executed by workers, after which the
drafted tutor utterance is checked by a **Pedagogical Utterance Auditor**
(revision x1). Student is simulated by a separate LLM.

No reward model, no "representative works".

## Pipeline per tutoring turn

1. Meta-Tutor: Fast Pedagogical Decomposition & Allocation (agenda)
2. Plan Detector (completeness / non-redundancy)
3. Replan (max 1)
4. Worker Execution (Diagnosis / Tutor Move Selector / Retrieval)
5. Meta-Tutor Final Response Generation
6. Pedagogical Utterance Auditor
7. Response Revision (max 1)

Tutor moves: `Focus` / `Probing` / `Telling` / `Generic`.

## Install

```bash
pip install -r tutor_aop/requirements.txt
```

## Launch vLLM servers (two, V100-friendly)

```bash
# Tutor server (all agents use this model)
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 \
    --port 8000 \
    --max-model-len 8192

# Student server
CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --port 8001 \
    --max-model-len 8192
```

`--dtype float16` is mandatory on V100 (no bf16 support).

## Run the experiment

```bash
cd /raid/nlpdrkim/song-workspace/tutor/kcc2026/Tutor-oriented-planning

# Live run (requires both vLLM servers up)
python -m tutor_aop.runner --config tutor_aop/config.yaml

# Smaller slice
python -m tutor_aop.runner --num 20 --start 0

# Mock run (no vLLM needed) — end-to-end dry run with deterministic canned outputs
python -m tutor_aop.runner --mock --num 2
```

Logs are written as JSONL to `tutor_aop/logs/episodes.jsonl` (one row per problem).

## Per-turn log fields

Each episode row includes, for every tutoring turn:

- `initial_agenda`
- `detector_output`
- `replan_agenda` (null if no replan)
- `executed_agenda`
- `worker_outputs` (diagnosis / tutor_move / retrieval outputs)
- `draft_response`
- `auditor_output`
- `revised_response` (null if no revision)
- `final_tutor_response`
- `student_response`

Plus episode-level: `initial_solution`, `initial_grade`, `post_tutoring_solution`,
`post_tutoring_grade`, `ended_by` (`end_token` / `max_turns` / `skip_correct_initial` /
`error_in_turn_N`), `dialogue`.

## RAG pool

`tutor_aop/rag_pool/proofwiki_examples.jsonl` holds ~27 proofwiki-style entries
covering core MATH-500 topics. The Retrieval Worker uses simple keyword-overlap
ranking (no embedding dependency) so the pipeline runs end-to-end without
additional models. Swap to dense retrieval later by replacing
`RetrievalWorker._load_pool` / the scoring routine.

Entry schema:
```json
{"id": "pw_001", "title": "...", "category": "...", "statement": "...", "keywords": [...]}
```

## Config knobs (`config.yaml`)

- `experiment.max_turns` — tutor/student pair count cap (default 5).
- `experiment.max_replan` / `max_revision` — set to 0 to disable.
- `retrieval.enabled` — `false` to turn the retrieval worker off globally.
- `retrieval.top_k` — retrieval depth.

## File layout

```
tutor_aop/
  config.yaml
  llm_client.py          # OpenAI-compatible wrapper; --mock supported
  mock_backend.py        # deterministic canned outputs for --mock
  utils.py               # JSON, boxed extraction, sympy equivalence
  meta_tutor.py          # agenda / replan / final / revise
  detector.py
  auditor.py
  student.py
  evaluator.py
  runner.py              # episode loop + JSONL logging (entry point)
  workers/
    diagnosis.py
    tutor_move.py
    retrieval.py
  prompts/
    meta_tutor_prompts.py
    detector_prompts.py
    auditor_prompts.py
    worker_prompts.py
    student_prompts.py
  rag_pool/
    proofwiki_examples.jsonl
  logs/
```
