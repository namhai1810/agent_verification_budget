# CBV-Bench Generator V2

CBV-Bench is an append-only, causal counterfactual data generator. It is separate from C2/C3
training and never uses correctness labels as model inputs.

## Execution tree

For each task and rollout, the planner runs once. Each action edge either preserves (`S`) or asks
the frozen TeacherVerifier to correct (`V`) the current artifact. The corrected artifact is passed
to the next Student agent, producing a new checkpoint state whose parent is that exact action edge.
Consequently a `V` at CP1 creates its own Coder and Reviewer states; baseline CP2/CP3 artifacts are
never spliced into that branch.

With a cold cache, Level 1 needs 3 unique Teacher calls and 6 Student calls per task/rollout. Level 2
needs 7 unique Teacher calls and 7 Student calls, not 12/24, because exact prefixes are shared.

## Safety and reproducibility

- Selection manifests are frozen from metadata before generation and contain no tests or solutions.
- Evaluator-only cases are typed separately and checked against every model message.
- Cache compatibility covers public task content, model/generation settings, prompts, workflow,
  dataset versions, software and rollout identity.
- A budget-blocked or failed branch remains explicit; validation refuses to call the dataset full.
- Split assignment hashes dataset/task ID and a split seed. Development IDs can be forced to
  `exploratory`; outcome-based test selection is unsupported.
- Historical Level-1 import is allowed only for temperature-zero, one-rollout data with exact model
  and prompt configuration. Its original seed provenance is retained. Stochastic studies generate a
  native tree.

## BigCodeBench status

The adapter pins the Hub revision, keeps `instruct_prompt` public and `test` private, and never
copies `canonical_solution`. Its evaluator is a separate isolated unittest bridge and reports that
mode explicitly. It does not alter HumanEval or LiveCodeBench evaluators and does not claim
BigCodeBench leaderboard equivalence or the official sanitized/calibrated submission protocol.

## Typical lifecycle

1. Prepare an immutable selection manifest.
2. Estimate cold-cache calls/tokens.
3. Collect or resume.
4. Validate, optionally with deep private-data scanning.
5. Export JSONL/Parquet and generate CSV/JSON/Markdown/PNG reports.

All outputs live below the configured `collection.results_dir`; legacy datasets and reports are never
rewritten.

## Student--Teacher model conditions

The generator does not infer model identity from hardcoded Qwen names. Every config declares a
`model_pair` descriptor and the runtime resolves it into a versioned `cbv-model-pair-v1` snapshot.
The snapshot stores:

- `model_pair_id` and an exact `model_pair_fingerprint`;
- `student_model` and `teacher_model` identity, family, revision, quantization and metadata;
- role-specific `prompt_version`;
- role-specific `inference_config` (provider, endpoint, sampling, token limit, seed and prices);
- provider runtime settings without serializing API-key values.

The full snapshot is embedded in every trajectory and run manifest. Transitions/effects carry both
pair identifiers, and exports include a normalized `model_pairs.jsonl` / `model_pairs.parquet` table.
Validation rejects branches from a different exact pair condition. Task/selection records remain
model-independent so the same frozen task split can be reused for cross-model generalization.

The current condition remains Qwen2.5-Coder-7B-Instruct-AWQ as Student and
Qwen2.5-Coder-32B-Instruct-AWQ as Teacher. To add another condition, create a config overlay with a
new result directory and override both `model_pair` and all four runtime role configs:

~~~yaml
_base_: cbv_bench_bigcodebench_v2.yaml

collection:
  name: cbv_bench_v2_my_student_my_teacher
  results_dir: results/cbv_bench_v2_my_student_my_teacher

model_pair:
  pair_id: my-student__my-teacher
  student_model:
    model_id: org/my-student
    family: my-family
    revision: pinned-revision
    quantization: ""
    parameter_count_b: 7
    metadata: {role: student}
  teacher_model:
    model_id: org/my-teacher
    family: my-family
    revision: pinned-revision
    quantization: ""
    parameter_count_b: 32
    metadata: {role: teacher}
  tags: [cross-model]
  metadata:
    research_axis: cross_model_generalization

models:
  planner: {model_id: org/my-student}
  coder: {model_id: org/my-student}
  reviewer: {model_id: org/my-student}
  teacher: {model_id: org/my-teacher}
~~~

The config validator requires all Student roles to match the declared Student identity and the
Teacher role to match the declared Teacher identity. Changing model, prompt, sampling, endpoint or
runtime settings creates a new exact condition fingerprint and therefore cannot reuse incompatible
generation cache entries.

## Commands

Review the config and start the already-configured Student/Teacher endpoints first. Local vLLM still
uses the repository explicit call guard:

~~~bash
conda activate agent_verify
cd /home/llm/hainn/hainn/agent_verification_budget
export VLLM_API_KEY=local
export AGENT_VERIFY_ALLOW_SPEND=YES
~~~

Static tests and config/data validation:

~~~bash
pytest -q tests/test_cbv_bench_v2.py
python scripts/cbv_bench.py --config configs/cbv_bench_v2.yaml validate-config
python scripts/cbv_bench.py --config configs/cbv_bench_v2.yaml validate-data --deep-privacy
~~~

Level-2 pilot over 15 deterministic existing HumanEval tasks (imports compatible Level 1):

~~~bash
python scripts/cbv_bench.py --config configs/cbv_bench_v2.yaml prepare-selection \
  --dataset humaneval --mode existing --task-count 15 \
  --selection-manifest results/cbv_bench_v2/selections/humaneval_existing_l2_15.json

python scripts/cbv_bench.py --config configs/cbv_bench_v2.yaml estimate \
  --dataset humaneval --level 2 \
  --selection-manifest results/cbv_bench_v2/selections/humaneval_existing_l2_15.json

python scripts/cbv_bench.py --config configs/cbv_bench_v2.yaml collect \
  --dataset humaneval --level 2 \
  --selection-manifest results/cbv_bench_v2/selections/humaneval_existing_l2_15.json
~~~

New-task pilot with 150 BigCodeBench tasks (Level 1 first):

~~~bash
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml prepare-selection \
  --dataset bigcodebench --mode new --task-count 150 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/new_pilot150.json

python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml estimate \
  --dataset bigcodebench --level 1 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/new_pilot150.json

python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml collect \
  --dataset bigcodebench --level 1 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/new_pilot150.json
~~~

Scale Level 1 to 1,000 BigCodeBench tasks, then collect a nested 250-task Level-2 subset:

~~~bash
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml prepare-selection \
  --dataset bigcodebench --mode all --task-count 1000 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/scale_l1_1000.json

python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml collect \
  --dataset bigcodebench --level 1 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/scale_l1_1000.json

python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml prepare-selection \
  --dataset bigcodebench --mode all --task-count 250 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/scale_l2_250.json

python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml collect \
  --dataset bigcodebench --level 2 \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/scale_l2_250.json
~~~

Resume uses the exact same collect command. Completed compatible roots, transitions and trajectories
are skipped. Final validation/export/report for a chosen manifest:

~~~bash
python scripts/cbv_bench.py --config configs/cbv_bench_bigcodebench_v2.yaml finalize \
  --dataset bigcodebench --level 2 --deep-privacy \
  --selection-manifest results/cbv_bench_v2_bigcodebench/selections/scale_l2_250.json
~~~

Set --max-teacher-calls and/or --max-teacher-tokens on estimate and collect when a hard collection cap
is needed. A cap produces an explicit incomplete run; export refuses it.
