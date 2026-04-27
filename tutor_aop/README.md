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

## Single-GPU vLLM management (default)

With `vllm_manager.enabled: true` in `config.yaml`, the runner auto-launches
both tutor and student vLLM servers on the same GPU with `--enable-sleep-mode`
and swaps them via `/sleep` + `/wake_up` so only one model holds GPU weights
at a time. No manual vLLM startup is needed — just run:

```bash
python -m tutor_aop.runner --config tutor_aop/config.yaml
```

Startup sequence per run: tutor server boots → sleeps → student server boots
→ sleeps → episodes begin, waking the active role on demand. Wake/sleep takes
~1–2s per swap (CPU↔GPU weight copy). Requires vLLM ≥ 0.6.4 and enough CPU
RAM to hold both model weights simultaneously (~28 GB for two 7–8B fp16 models).

Tune in `config.yaml`:

- `vllm_manager.cuda_visible_devices` — which GPU to use (default `"0"`).
- `vllm_manager.gpu_memory_utilization` — fraction of total VRAM each server
  may use when awake. `0.45` is safe on a 48 GB A6000 for two 7–8B models.
  Bump higher if you only ever wake one at a time *and* startup allocation
  profiling still succeeds; lower if you see OOM during boot.
- `vllm_manager.dtype` — `float16` or `bfloat16`. A6000/Ampere supports both.
- `vllm_manager.extra_args` — passed through to each vLLM process
  (e.g. `["--trust-remote-code"]`).

### Multi-GPU setup (fallback)

Set `vllm_manager.enabled: false` and launch the two servers manually on
separate GPUs as before:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dtype float16 --port 8000 --max-model-len 8192

CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 --port 8001 --max-model-len 8192
```

## Run the experiment

```bash
cd /raid/nlpdrkim/song-workspace/tutor/kcc2026/Tutor-oriented-planning

# Live run (single-GPU default — the runner spawns vLLM itself)
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
