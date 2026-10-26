# Phase 3 — LiveCodeBench v6

Pipeline này giữ nguyên workflow `Planner → CP1 → Coder → CP2 → Reviewer → CP3`, counterfactual SKIP/VERIFY và resume append-only của HumanEval. Adapter mới chỉ thay task format và evaluator.

## Protocol đã khóa

- Dataset: `livecodebench/code_generation_lite`, config tăng dần `v6`.
- Dataset revision: `175d0ac20500fd7aa0813f7a2e6ae7ecad3c76c9`.
- Official evaluator repository commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
- Date filter: `contest_date >= 2025-01-01`.
- Selection seed: `42`; 100 IDs được ghi một lần vào `results/livecodebench/selected_tasks.json`.
- Eligible v6: 175 tasks — 43 Easy, 52 Medium, 80 Hard.
- Locked 100: 43 Easy, 52 Medium, 5 Hard; 55 AtCoder và 45 LeetCode.
- Selection dùng difficulty/platform/ID và seed, không dùng output hoặc correctness của Student/Teacher.

`v6` được dùng thay cho cumulative `release_v6` để chỉ tải shard 2025 cần thiết (~124 MB), tránh download nhiều GB dữ liệu cũ rồi mới lọc ngày.

## Cài đặt

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
conda env update -n agent_verify -f environment.yml
bash scripts/setup_livecodebench.sh
```

Script setup clone official evaluator vào `.vendor/LiveCodeBench` và checkout đúng commit đã pin.

## Commands

Xem trước và khóa danh sách 100 tasks, không gọi LLM:

```bash
bash scripts/run_livecodebench_pilot.sh --prepare-only
```

Smoke pilot 4 tasks:

```bash
bash scripts/run_livecodebench_pilot.sh \
  --task-count 4 \
  --seed 42 \
  --repeated-pairs 1 \
  --resume
```

Chạy đủ 100 tasks, một repetition:

```bash
bash scripts/run_livecodebench_pilot.sh \
  --task-count 100 \
  --seed 42 \
  --repeated-pairs 1 \
  --resume
```

Nếu bị ngắt, chạy lại nguyên command trên. `baseline.jsonl` và `intervention_pairs.jsonl` đã hoàn tất sẽ được bỏ qua theo deterministic run/pair IDs.

Nếu source code đã được sửa sau một run chưa hoàn chỉnh, tiếp tục chính run cũ bằng explicit audited recovery ID:

```bash
bash scripts/run_livecodebench_pilot.sh \
  --task-count 100 --seed 42 --repeated-pairs 1 --resume \
  --resume-run-id h1_livecodebench_v6_qwen25_gpu2-20646527df-9e5e2e077f
```

Chế độ này chỉ được phép khi config fingerprint khớp manifest cũ. Nó ghi `run_recovery` event chứa software hash trước/sau và chỉ chạy các canonical pair chưa có `status=completed`.

### Chạy qua đêm: phục hồi CP3 và thu thập 75 task còn lại

```bash
bash scripts/run_livecodebench_overnight.sh
```

Script thực hiện tuần tự và fail-fast:

1. Khóa selection 175 task rồi kiểm tra 100 ID đầu trùng tuyệt đối selection cũ.
2. Resume đúng run 100-task cũ, chỉ lấp các pair còn thiếu (hiện tại: 1 CP2 và 60 CP3).
3. Chạy offset `100`, count `75` trong results directory riêng; 75 task này đều là Hard.
4. Tạo lại Markdown report, CSV metrics, dashboard PNG và master console log.

Không gộp trực tiếp Hard-75 vào thống kê của sample 100 ban đầu: hai thư mục kết quả riêng giúp giữ rõ sampling distribution và audit trail.

Override endpoint khi cần:

```bash
bash scripts/run_livecodebench_pilot.sh \
  --task-count 100 --seed 42 --repeated-pairs 1 --resume \
  --student-endpoint http://127.0.0.1:8011/v1 \
  --teacher-endpoint http://127.0.0.1:8012/v1
```

Phân tích H1 và so sánh riêng với HumanEval:

```bash
bash scripts/analyze_livecodebench.sh \
  --results-dir results/livecodebench \
  --humaneval-results results
```

Train predictor chỉ trên LiveCodeBench, dùng fixed 20% outer test:

```bash
bash scripts/train_livecodebench_predictor.sh \
  --mode livecodebench \
  --livecodebench-results results/livecodebench \
  --output-dir results/livecodebench/phase2_lcb \
  --seed 42
```

Train kết hợp HumanEval + LiveCodeBench và đánh giá riêng từng benchmark:

```bash
bash scripts/train_livecodebench_predictor.sh \
  --mode combined \
  --livecodebench-results results/livecodebench \
  --humaneval-results results \
  --humaneval-run-id h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f \
  --output-dir results/livecodebench/phase2_combined \
  --seed 42
```

Không dùng cùng `--output-dir` cho `livecodebench` và `combined`, vì mỗi mode có một task universe và fixed test split khác nhau.

## Evaluation isolation

`LiveCodeBenchRunner` gọi trực tiếp official `lcb_runner.evaluation.testing_util.run_test`, nhưng generated code chỉ chạy sau khi worker đã áp dụng:

- Landlock read allowlist; không cho ghi filesystem.
- seccomp cấm network, process creation, exec và escape syscalls.
- CPU, RAM, process-count, output-size và wall-clock limits.

Private/public evaluator cases chỉ tồn tại trong `LiveCodeBenchPrivateCase`. Model-facing `LiveCodeBenchPublicTask`, prompt, critic features và checkpoint logs không chứa private tests. Trạng thái chấm được chuẩn hóa thành `PASS`, `FAIL`, `TIMEOUT`, `RUNTIME_ERROR`, `FORMAT_ERROR`, `EVALUATOR_ERROR`. `EVALUATOR_ERROR` làm record thất bại để resume; nó không được chuyển thành model incorrect (`Y=0`).

## Outputs

`results/livecodebench/` chứa:

- `selected_tasks.json`, `dataset_manifest.json`, `run_config.json`.
- `baseline.jsonl`, `checkpoint_states.jsonl`, `intervention_pairs.jsonl`.
- `llm_invocations.jsonl`, `evaluations.jsonl`, `events.jsonl`, `manifests.jsonl`.
- `summary.md`, `h1_analysis.md`, `accuracy_by_checkpoint.csv`, `helpful_harmful.csv`.
- `logs/` cho console logs.

Training tạo fixed test manifest, serialized models, predictions, policy/predictor CSVs, run manifest và report trong output directory được chỉ định.

Run qua đêm bổ sung:

- `results/livecodebench/report/livecodebench_100_dashboard.png`.
- `results/livecodebench/report/livecodebench_100_dashboard_checkpoint_metrics.csv`.
- `results/livecodebench_hard_extension/h1_analysis.md`, `summary.md` và các CSV phân tích.
- `results/livecodebench_hard_extension/report/livecodebench_hard_extension_dashboard.png`.
- `results/livecodebench_hard_extension/logs/lcb_offset100_75x1.log`.
- `results/livecodebench_overnight_logs/overnight_<timestamp>.log`.

## Tests

```bash
conda activate agent_verify
pytest -q
ruff check .

python -m agent_verify.livecodebench_cli prepare --help
python -m agent_verify.livecodebench_cli run --help
python -m agent_verify.livecodebench_cli analyze --help
python scripts/train_livecodebench_predictor.py --help
```

