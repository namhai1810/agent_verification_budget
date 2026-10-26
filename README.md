# Downstream-Aware Verification — Phase 1

Triển khai nghiên cứu tái lập cho Modules A/B/C: workflow Planner → Coder → Reviewer, teacher verification tại đúng một checkpoint, nhánh paired SKIP/VERIFY, local critic độc lập, HumanEval evaluator cô lập, và báo cáo H1.

Đây chỉ là Phase 1. Repo **không** huấn luyện value predictor, routing policy hay chạy H2.

## Chạy nhanh

Môi trường được cố định ở Conda `agent_verify` (Python 3.10). Từ thư mục repo:

```bash
bash scripts/run_phase1.sh
```

Lệnh mặc định dùng các model `mock` và chạy 4 task để kiểm tra toàn bộ plumbing mà không phát sinh chi phí. Mock coder cố ý trả placeholder và mock teacher không sửa artifact, nên kết quả này **không phải bằng chứng thực nghiệm**.

Chạy config/model thật sau khi đã chỉnh `configs/real_example.yaml`:

```bash
export OPENAI_API_KEY='...'
export AGENT_VERIFY_ALLOW_SPEND=YES
bash scripts/run_phase1.sh configs/real_example.yaml --task-count 24 --repeated-pairs 1
```

Hai lớp khóa chi phí đều bắt buộc: `budget.allow_paid_calls: true` trong config và biến môi trường `AGENT_VERIFY_ALLOW_SPEND=YES`. Mỗi request còn được kiểm tra upper-bound trước khi gọi; tổng không được vượt `budget.max_spend_usd`.

## Model mẹ, model con và critic

Tất cả tên model nằm ở `configs/*.yaml`, không hardcode trong source:

| Vai trò | Ý nghĩa | Khóa config mặc định |
|---|---|---|
| Planner | model con tạo kế hoạch, checkpoint 1 | `models.planner.model_id: mock-child-planner-v1` |
| Coder | model con sinh implementation, checkpoint 2 | `models.coder.model_id: mock-child-coder-v1` |
| Reviewer | model con review và trả code cuối, checkpoint 3 | `models.reviewer.model_id: mock-child-reviewer-v1` |
| Teacher | model mẹ/strong verifier, chỉ can thiệp một lần | `models.teacher.model_id: mock-parent-teacher-v1` |
| Local critic | model nhẹ/heuristic ước lượng risk độc lập | `models.critic.model_id: deterministic-local-critic-v1` |

Đổi `provider` thành `openai_compatible` và thay `model_id` bằng ID chính xác của endpoint bạn dùng. Mỗi vai trò có `temperature`, `top_p`, `max_output_tokens`, `seed`, `prompt_version`, và giá input/output riêng để freeze thí nghiệm và tính cost. `provider.base_url` cho phép dùng OpenAI API hoặc một server tương thích (ví dụ vLLM); secret chỉ đọc từ biến có tên ở `provider.api_key_env`.

`real_example.yaml` kế thừa `default.yaml` qua `_base_`, giúp config thật ngắn và tránh lệch các tham số an toàn. Hãy copy nó thành một file config riêng trước khi sửa.

### Hai Qwen2.5-Coder trên GPU 2

Config chạy local nằm ở `configs/vllm_gpu2.yaml`:

- Planner/Coder/Reviewer (model con): `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`, port `8011`.
- Teacher (model mẹ): `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ`, port `8012`.
- Local critic vẫn là `deterministic-local-critic-v1` để giữ phép đo risk độc lập.

Port `8001` trong ví dụ ban đầu đang bị một service uvicorn khác chiếm trên máy này, nên config dùng
`8011/8012`. Source hỗ trợ `models.<role>.base_url`, vì vậy model con và model mẹ được route đến hai
OpenAI-compatible endpoints riêng.

Khởi động server từ một environment đã cài vLLM:

```bash
bash scripts/start_vllm_gpu2.sh start
bash scripts/start_vllm_gpu2.sh status
bash scripts/start_vllm_gpu2.sh logs
```

Khi cả hai endpoint báo `API ready`, mở terminal khác và chạy smoke pilot:

```bash
bash scripts/run_vllm_pilot.sh --task-count 4 --repeated-pairs 1
```

Để chạy pilot 24 task với progress bar, elapsed time, ETA, console log và tự động tạo visualization:

```bash
bash scripts/run_vllm_with_progress.sh --task-count 24 --repeated-pairs 1
```

Nếu bị ngắt, chạy lại đúng command trên. Append-only records đã hoàn tất được resume; console log nằm ở
`results/logs/phase1_24x1_console.log`.

Dừng đúng hai process do script quản lý bằng `bash scripts/start_vllm_gpu2.sh stop`. Log và PID được
lưu dưới `results/vllm_gpu2/`.

## Kiến trúc và luồng dữ liệu

```text
PublicTask -> Planner -> CP1 -> Coder -> CP2 -> Reviewer -> CP3 -> sandbox evaluator
                           \       paired SKIP / VERIFY at exactly one CP       /
                            -> frozen TeacherVerifier (VERIFY branch only) -----
```

- `src/agent_verify/agents/fixed.py`: ba agent tuần tự và prompt/schema cố định.
- `src/agent_verify/workflow.py`: baseline, restore checkpoint, continuation với seed matched giữa SKIP/VERIFY.
- `src/agent_verify/verification/teacher.py`: teacher trả verdict, artifact đã sửa, explanation và cờ `changed`; artifact được validate lại đúng schema checkpoint.
- `src/agent_verify/verification/local_critic.py`: heuristic hoặc model nhẹ; không nhận teacher output, test, canonical solution hay downstream outcome.
- `src/agent_verify/evaluation/humaneval_runner.py`: truyền gold tests duy nhất vào worker bị giới hạn bằng Landlock (read allowlist, cấm write) và seccomp (cấm network/fork/exec/escape), kèm timeout và rlimits.
- `src/agent_verify/experiments/phase1.py`: baseline + 3 checkpoint × số repeated pairs; giữ nguyên failure và hỗ trợ resume theo deterministic `run_id`.
- `src/agent_verify/analysis/analyze_h1.py`: delta, correlation, ranking disagreement, offline oracle diagnostic, case studies và bốn figure.

Loader chủ động tách `PublicTask(prompt, entry_point)` khỏi `PrivateEvaluationCase(test, entry_point)`. `canonical_solution` không được copy. Prompt/messages trong raw log vì vậy không có gold tests hay lời giải chuẩn.

## Output

`results/` chứa:

- `manifests.jsonl`: config đầy đủ, model IDs, phiên bản/hash dataset, task IDs, Python/platform và hash source.
- `llm_invocations.jsonl`: prompt/config/version/hash, input/output, token, cost, latency, retry/failure.
- `checkpoint_states.jsonl`: observable state, artifacts trước đó, message, model config và parent state.
- `baseline.jsonl`: final implementation, outcome và local risk tại ba checkpoint.
- `intervention_pairs.jsonl`: state gốc, original/corrected artifacts, Y_skip, Y_verify, delta, teacher verdict/cost, trajectory refs.
- `events.jsonl`: failures được giữ lại.
- `reports/<run_id>/summary.md`, `h1_analysis.csv`, `figures/*.png`: bundle phân tích H1 bất biến
  theo run. Các đường dẫn `results/summary.md`, `results/h1_analysis.csv`, `results/figures/` là alias
  của run được phân tích gần nhất.

Các raw JSONL chỉ append. `run_id` kết hợp fingerprint config và hash source; chạy lại cùng code/config sẽ resume các baseline/pair đã hoàn tất. Thay code/config tạo run mới, tránh trộn dữ liệu.

## Lịch chạy đề xuất

```bash
# 1) Kiểm tra config và sandbox
conda run -n agent_verify python -m agent_verify.cli validate-config --config configs/default.yaml
conda run -n agent_verify python -m agent_verify.cli check-sandbox --config configs/default.yaml

# 2) Smoke mock 4 task
bash scripts/run_phase1.sh

# 3) Pilot thật 24 task (chỉ sau khi duyệt model và ngân sách)
bash scripts/run_phase1.sh configs/my_real.yaml --task-count 24

# 4) Expanded pilot 48 task nếu đủ nguồn lực
bash scripts/run_phase1.sh configs/my_real.yaml --task-count 48
```

Task selection là shuffle xác định bởi `experiment.seed`; manifest ghi chính xác task IDs. Có thể đổi sang `ordered`. Repeated pair dùng cùng downstream seed cho hai nhánh, nhưng report vẫn ghi chú rằng kernel/provider bên ngoài có thể không deterministic tuyệt đối.

## Cài lại môi trường

```bash
conda env update -n agent_verify -f environment.yml
conda activate agent_verify
python -m pip install -e '.[dev]'
```

Dependency trực tiếp được pin. File `data/HumanEval.jsonl.gz` lấy từ commit chính thức
`b9453e8f1865df0a0057a1383767caa400acfbcf` của `openai/human-eval`; loader kiểm tra SHA-256
trước mỗi run. Chạy kiểm thử:

```bash
conda run -n agent_verify pytest
conda run -n agent_verify ruff check src tests
```

Không chạy pilot trả phí nếu chưa có credential và ủy quyền chi phí rõ ràng. Nếu Landlock/seccomp không hoạt động, runner fail-closed trước khi gọi model.

## Extract và visualize dữ liệu

```bash
bash scripts/visualize_results.sh
```

Script mặc định chọn manifest mới nhất, flatten raw JSONL thành CSV và tạo dashboard tại
`results/extracted/<run_id>/`. Có thể chọn run cụ thể bằng:

```bash
bash scripts/visualize_results.sh --run-id <run_id>
```

## Phase 3 — LiveCodeBench

LiveCodeBench v6 đã có adapter, official evaluator bridge, sandbox, resume/progress, H1 reports và fixed-test predictor training riêng. Hướng dẫn đầy đủ, protocol đã pin và exact commands nằm tại [docs/livecodebench.md](docs/livecodebench.md).

Chuẩn bị dataset/evaluator mà không gọi LLM:

```bash
conda activate agent_verify
bash scripts/setup_livecodebench.sh
bash scripts/run_livecodebench_pilot.sh --prepare-only
```

Chạy đủ 100 tasks chỉ khi hai vLLM endpoints đã sẵn sàng:

```bash
bash scripts/run_livecodebench_pilot.sh \
  --task-count 100 --seed 42 --repeated-pairs 1 --resume
```

Phần implementation/test setup không tự chạy experiment 100 tasks và không tự train predictor.
