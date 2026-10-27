# Báo cáo tổng hợp — Downstream-Aware Verification, Phase 1–CBV-Bench V2

**Ngày khởi tạo báo cáo:** 24/10/2026  
**Cập nhật gần nhất:** 27/10/2026  
**Datasets:** HumanEval, LiveCodeBench v6; BigCodeBench đã lên lịch nhưng chưa bắt đầu collection  
**Trạng thái hiện tại:** Phase 1–3 và các controller ML đã được phân tích; CBV-Bench V2 hoàn tất HumanEval Level 2, còn LiveCodeBench Level 2 ở trạng thái partial 171/175.

> Báo cáo này được tổng hợp trực tiếp từ các JSONL records và run-specific reports. Các lần lặp `R=3` chỉ được dùng để kiểm tra tính tái lập, không được tính như ba tập task độc lập.

![Dashboard kết quả](../results/final_report_241026/final_results_dashboard.png)

## 1. Tóm tắt điều hành

Hệ thống so sánh hai nhánh bắt cặp tại ba checkpoint của workflow Planner → Coder → Reviewer:

- **SKIP:** giữ nguyên artifact tại checkpoint.
- **VERIFY:** gọi Teacher để kiểm tra và có thể sửa artifact.
- Hai nhánh tiếp tục với cùng decoding settings và được đánh giá bằng cùng hidden HumanEval tests.
- Hiệu ứng của verification được định nghĩa là `delta = Y_VERIFY - Y_SKIP`.

Kết quả held-out trên 139 tasks paired-complete:

| Chỉ số | Kết quả |
|---|---:|
| Baseline/paired SKIP pass | 113/139 = **81.3%** |
| VERIFY pass | 353/417 = **84.7%** |
| SKIP pass | 339/417 = **81.3%** |
| Helpful interventions | **19/417 = 4.6%** |
| Harmful interventions | **5/417 = 1.2%** |
| Unchanged outcomes | **393/417 = 94.2%** |
| Net gain | **+14/417 = +3.4 percentage points** |
| Teacher artifact changes | **101/417 = 24.2%** |
| Pipeline generation failures | **1/140 = 0.7%** |

Kết luận chính: Teacher verification tạo ra lợi ích dương trên held-out set. Tuy nhiên, lợi ích tập trung ở một số task, 76.2% các artifact được Teacher sửa không làm thay đổi binary outcome, và critic hiện chưa đủ tốt để routing chính xác.

## 2. Cấu hình hệ thống

| Thành phần | Cấu hình |
|---|---|
| Child model | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` |
| Child roles | Planner, Coder, Reviewer |
| Child endpoint | vLLM OpenAI-compatible, `127.0.0.1:8011` |
| Teacher model | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` |
| Teacher endpoint | vLLM OpenAI-compatible, `127.0.0.1:8012` |
| Critic | `deterministic-local-critic-v1`, heuristic local |
| GPU | Physical GPU 2, RTX 5880 Ada 48 GB |
| Decoding | `temperature=0.0`, `top_p=1.0` |
| Structured output | JSON Schema + Pydantic validation |
| Evaluator isolation | Landlock + seccomp; filesystem/network/process escape denied |
| Dataset selection | Shuffle seed `20261024`; dev offset 0/count 24; held-out offset 24/count 140 |

### Luồng baseline

```text
HumanEval problem
      │
      ▼
Planner ──CP1──► Coder ──CP2──► Reviewer ──CP3──► sandbox evaluation
```

### Luồng paired intervention tại mỗi checkpoint

```text
Restored checkpoint artifact
       ├── SKIP ─────────────────────────► downstream agents ─► evaluate
       └── Teacher verify/correct ───────► downstream agents ─► evaluate

delta = pass(VERIFY) - pass(SKIP)
```

Mỗi pair chỉ can thiệp tại **một** checkpoint. Đây không phải luồng gọi Teacher liên tiếp ở cả CP1, CP2 và CP3 trên cùng trajectory.

## 3. Những cải thiện đã thực hiện

### 3.1 Tách model và cấu hình thực nghiệm

- Tách Child 7B và Teacher 32B thành hai vLLM endpoints.
- Đưa tên model, endpoint, seed, temperature, token limit và prompt version vào YAML config.
- Dùng task selection seed cố định để dev/held-out không trùng nhau.
- Thêm `task_offset` để giữ 24 tasks đầu làm development và 140 tasks còn lại làm held-out.

### 3.2 Nâng Teacher prompt từ v1 lên v2

Prompt v1 dùng policy verification chung. Prompt v2 bổ sung:

- Tự suy ra contract từ đề bài và public examples.
- Xem plan, code và review trước đó là nguồn có thể sai, không phải ground truth.
- Tôn trọng literal task specification, kể cả khi khác bài toán “quen thuộc”.
- Không phát minh requirement hoặc hidden test.
- Nếu artifact đúng, giữ nguyên thay vì rewrite theo style.
- Nếu sai, sửa tối thiểu và có căn cứ.
- Guidance riêng cho CP1/CP2/CP3 và mô tả downstream còn lại sau checkpoint.

Kết quả trên cùng 24-task dev set tăng từ net `0/72` với v1 lên `+1/72` với v2.

### 3.3 Độ bền vận hành

- Append-only JSONL logs cho manifest, baseline, checkpoints, invocations, evaluations, pairs và events.
- Resume theo content-derived run/pair ID, không chạy lại các completed records.
- Progress bar hiển thị calls, baselines, pairs, evaluations, errors, warnings, elapsed và ETA.
- Tự phát hiện mismatch giữa model-declared `changed` và artifact diff; dùng derived diff làm giá trị chính thức.
- Tách log theo `task_offset` để các split không ghi lẫn nhau.
- Kiểm tra API readiness của cả Child và Teacher trước khi chạy.

### 3.4 An toàn evaluation

- Generated code chạy trong Landlock + seccomp sandbox.
- Chặn filesystem, network và process escape.
- Timeout và memory limit được cấu hình cho mỗi evaluation.

## 4. Ma trận thí nghiệm đã chạy

| Giai đoạn | Prompt | Tasks | Repetitions | Pairs | Helpful | Harmful | Net | Ghi chú |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Mock smoke test | v1 | 4 | 1 | 12 | — | — | — | Kiểm tra pipeline, không dùng kết luận model |
| Real pilot đầu, chưa hoàn tất | v1 | 24 dự kiến | 1 | 60 hoàn tất | 0 | 1 | -1 | Phát hiện thiếu resume/progress robustness; loại khỏi comparison chính |
| Dev hoàn chỉnh | v1 | 24 | 1 | 72 | 1 | 1 | **0** | Baseline trước prompt improvement |
| Dev hoàn chỉnh | v2 | 24 | 1 | 72 | 2 | 1 | **+1** | Thêm rescue `HumanEval/26` tại CP1 |
| Stability check | v2 | 24 | 3 | 216 | 6 | 3 | **+3** | Ba repetitions tái tạo đúng 2 helpful/1 harmful; không độc lập |
| Held-out | v2 | 140 attempted / 139 complete | 1 | 417 | 19 | 5 | **+14** | Một generation failure tại `HumanEval/38` |

Run IDs dùng trong phân tích:

- v1 dev: `h1_phase1_qwen25_gpu2-b08c6b8c6d-2af13ca7b2`
- v2 dev R=1: `h1_phase1_qwen25_gpu2-72700d2d7f-f68c87f537`
- v2 dev R=3: `h1_phase1_qwen25_gpu2-79205f0481-f68c87f537`
- v2 held-out: `h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f`

## 5. Kết quả held-out chi tiết

### 5.1 Hiệu ứng theo checkpoint

| Checkpoint | SKIP | VERIFY | Helpful | Harmful | Net | Delta |
|---|---:|---:|---:|---:|---:|---:|
| CP1 — Plan | 113/139 | 115/139 | 5 | 3 | +2 | **+1.4 pp** |
| CP2 — Code | 113/139 | 118/139 | 6 | 1 | +5 | **+3.6 pp** |
| CP3 — Final review | 113/139 | 120/139 | 8 | 1 | +7 | **+5.0 pp** |

CP3 có effect size quan sát lớn nhất trên held-out set. Điều này đảo ngược ranking của 24-task dev pilot, nơi CP1 đứng đầu. Kết quả cho thấy không nên chọn checkpoint chỉ dựa trên pilot nhỏ.

### 5.2 Khi Teacher thực sự sửa artifact

Trong 101 artifact changes:

| Outcome sau sửa | Count | Tỷ lệ trên changes |
|---|---:|---:|
| Helpful | 19 | **18.8%** |
| Không đổi binary outcome | 77 | **76.2%** |
| Harmful | 5 | **5.0%** |

Teacher có precision dương nhưng còn sửa thừa. Việc thêm một acceptance gate hoặc confidence/calibration layer có tiềm năng giảm 5 harmful cases mà vẫn giữ phần lớn rescue cases.

### 5.3 Critic và routing

| Metric | Dev 24 | Held-out 139 |
|---|---:|---:|
| Risk–delta Pearson correlation | 0.032 | 0.223 |
| Highest-risk policy gain, earliest tie-break | +4.2 pp | +1.4 pp |
| Offline-oracle mean delta | +8.3 pp | +8.6 pp |
| Maximum-risk ties ảnh hưởng lựa chọn | 2/24 | 11/139 |

Critic có tín hiệu dương nhẹ trên held-out nhưng score vẫn quá rời rạc. Audit sau Phase 1 xác định con số legacy `134/139` đã gộp tie ở maximum risk **hoặc** maximum observed Delta; nó không phải số risk-ranking ties. Có 136/139 tasks lặp ít nhất một risk score, nhưng chỉ 11/139 ties tại maximum risk thực sự có thể đổi lựa chọn. `HumanEval/93` helpful và `HumanEval/137` harmful đều từng nhận risk `0.15`, nên critic hiện chưa phân biệt tốt “Teacher sẽ cứu” với “Teacher sẽ phá”.

## 6. Tổng hợp dev + held-out

Để tránh pseudo-replication, phần này chỉ ghép hai run `R=1`: 24 dev tasks và 139 held-out tasks paired-complete. Run `R=3` bị loại khỏi tổng hợp.

| Metric | Combined R=1 |
|---|---:|
| Analyzable tasks | 163 |
| Paired interventions | 489 |
| SKIP pass | 396/489 = **81.0%** |
| VERIFY pass | 411/489 = **84.0%** |
| Helpful / harmful | **21 / 6** |
| Net gain | **+15/489 = +3.1 pp** |

Nếu tính `HumanEval/38` generation failure là failure của toàn pipeline, baseline trên toàn bộ 164 tasks là `132/164 = 80.5%`. Task đó không có đủ checkpoint artifacts nên không thể đưa vào paired effect estimate.

Theo checkpoint trên 163 analyzable tasks:

| Checkpoint | Helpful | Harmful | Net | Delta |
|---|---:|---:|---:|---:|
| CP1 | 6 | 3 | +3 | +1.8 pp |
| CP2 | 7 | 2 | +5 | +3.1 pp |
| CP3 | 8 | 1 | +7 | +4.3 pp |

## 7. Case studies từ artifact logs

![Các can thiệp helpful và harmful](../results/final_report_241026/teacher_intervention_case_studies.png)

### 7.1 Helpful — `HumanEval/26`, CP1

Plan ban đầu hiểu sai “remove duplicates” thành giữ một occurrence của mỗi value. Teacher đổi sang đếm frequency và chỉ giữ các value có `count == 1`. Coder downstream làm theo plan mới và outcome chuyển `FAIL → PASS`.

### 7.2 Helpful — `HumanEval/93`, CP2

Code ban đầu shift vowel nhưng quên đảo case:

```python
result.append(shifted_vowels[char])
```

Teacher sửa thành:

```python
result.append(shifted_vowels[char].swapcase())
```

Outcome chuyển `FAIL → PASS`.

### 7.3 Helpful — `HumanEval/19`, CP3

Final reviewer artifact gọi `num_map.inverse`, trong khi Python dictionary không có API này. Teacher tạo reverse mapping bằng `{v: k for k, v in num_map.items()}` và outcome chuyển `FAIL → PASS`.

### 7.4 Harmful — `HumanEval/137`, CP2

Code SKIP so sánh numeric value sau normalization nhưng trả về input gốc, ví dụ giữ chuỗi `"1,5"`. Teacher cố “preserve original” nhưng biến được gọi là `original_a` thực tế đã bị normalize thành `"1.5"`; output representation bị thay đổi và outcome chuyển `PASS → FAIL`.

Đây là failure mode quan trọng: Teacher xác định đúng loại invariant cần bảo vệ nhưng implementation của chính correction lại vi phạm invariant đó.

## 8. Phân tích thống kê

Trên 417 held-out pairs, có 19 helpful và 5 harmful discordant outcomes. McNemar/sign exact test nếu tạm coi mỗi task–checkpoint là độc lập cho `p ≈ 0.0066`.

Theo từng checkpoint:

| Checkpoint | Helpful | Harmful | Exact p-value |
|---|---:|---:|---:|
| CP1 | 5 | 3 | 0.727 |
| CP2 | 6 | 1 | 0.125 |
| CP3 | 8 | 1 | 0.039 |

Cần thận trọng với các con số này:

- Ba checkpoint của cùng một task có tương quan.
- CP3 `p=0.039` là uncorrected; Bonferroni cho ba checkpoint cho khoảng `0.117`.
- Combined R=1 có 21 helpful/6 harmful, exact paired `p ≈ 0.0059`, nhưng vẫn cần task-cluster bootstrap hoặc cluster-robust inference.
- R=3 dùng `temperature=0.0`; ba repetitions tạo outcome giống nhau hoàn toàn và không tăng effective sample size gấp ba.

Kết luận phù hợp là **có tín hiệu verification benefit đáng kể**, nhưng chưa đủ để tuyên bố checkpoint-optimal policy cuối cùng.

## 9. Lỗi, warnings và tính toàn vẹn dữ liệu

### 9.1 `HumanEval/38` generation failure

Coder bị rơi vào chuỗi escape/newline lặp lại. Cả hai API attempts chạm đúng `max_output_tokens=1600`, tạo JSON prefix chưa đóng và phát sinh:

```text
JSONDecodeError: Unterminated string starting at: line 2 column 11
```

Run vì vậy có:

- 140 tasks attempted.
- 139 completed baselines.
- 417/420 completed pairs.
- 973/980 evaluations.
- 2,087 recorded invocations thay vì 2,100 planned records.

Không được impute paired outcome cho task này. Trong reporting toàn-system có thể tính đây là một pipeline failure; trong causal paired analysis phải ghi là missing/non-analyzable.

### 9.2 Consistency warnings

Held-out run có 40 warnings:

- 29 Reviewer `changed` mismatches.
- 11 Teacher `changed` mismatches.

Pipeline so sánh artifact/code thực tế và dùng `derived_value`, nên các warnings không làm sai helpful/harmful counts. Tuy nhiên, warning rate cho thấy self-reported metadata của model không nên được dùng trực tiếp.

### 9.3 Compute usage

Held-out partial-complete run ghi nhận:

- 417 Teacher invocations.
- 684,881 Teacher tokens.
- Mean Teacher latency khoảng 6.39 giây.
- Khoảng 1.95 triệu total logged tokens trên mọi role.
- Wall time khoảng 1 giờ 22 phút.
- Configured API cost bằng 0 vì model self-hosted; GPU time và điện năng chưa được quy đổi thành USD.

## 10. Kết luận nghiên cứu

Các kết quả hiện tại ủng hộ ba kết luận:

1. **Teacher verification có giá trị downstream thực:** held-out paired pass rate tăng khoảng 3.4 pp, với 19 helpful so với 5 harmful.
2. **Vị trí verification quan trọng:** CP3 tốt nhất trên held-out, dù CP1 tốt nhất trên dev pilot. Checkpoint selection cần đánh giá trên held-out data, không chọn theo 24-task pilot.
3. **Routing là bottleneck tiếp theo:** offline oracle đạt +8.6 pp trong khi highest-risk heuristic với earliest tie-break chỉ đạt +1.4 pp. Cải thiện critic/acceptance gate có khả năng đem lại nhiều giá trị hơn chỉ thay Teacher model ngay lập tức.

## 11. Đề xuất thí nghiệm tiếp theo tại thời điểm kết thúc Phase 1

Ưu tiên theo thứ tự:

1. Thêm task-cluster bootstrap confidence intervals và McNemar exact statistics vào report generator.
2. Thêm acceptance gate để Teacher chỉ được áp dụng correction khi contract evidence đủ mạnh.
3. Cải thiện critic để có score liên tục hơn và giảm ranking ties.
4. Đóng băng prompt v2/pipeline, sau đó A/B Teacher hiện tại với model Teacher mới trên cùng split và seeds.
5. Tách rõ ba metric: rescue rate trên SKIP failures, damage rate trên SKIP passes, và beneficial precision trên Teacher changes.
6. Xử lý structured-output degeneration bằng bounded retry có corrective feedback; báo generation failure như một metric chính thức.

Không nên tune thêm trên 140-task held-out set rồi tiếp tục gọi nó là held-out. Nếu thay critic, prompt hoặc acceptance rule dựa trên kết quả này, cần tạo một test set mới hoặc báo rõ đây đã trở thành development evidence.

## 12. Artifact và log inventory

### Báo cáo và hình

- Dashboard: [`results/final_report_241026/final_results_dashboard.png`](../results/final_report_241026/final_results_dashboard.png)
- Case studies: [`results/final_report_241026/teacher_intervention_case_studies.png`](../results/final_report_241026/teacher_intervention_case_studies.png)
- Experiment summary CSV: [`results/final_report_241026/experiment_summary.csv`](../results/final_report_241026/experiment_summary.csv)
- Script tái tạo hình: [`scripts/build_final_report_241026.py`](../scripts/build_final_report_241026.py)

### Run-specific reports

- Prompt v1 dev: [`results/reports/h1_phase1_qwen25_gpu2-b08c6b8c6d-2af13ca7b2/summary.md`](../results/reports/h1_phase1_qwen25_gpu2-b08c6b8c6d-2af13ca7b2/summary.md)
- Prompt v2 dev R=1: [`results/reports/h1_phase1_qwen25_gpu2-72700d2d7f-f68c87f537/summary.md`](../results/reports/h1_phase1_qwen25_gpu2-72700d2d7f-f68c87f537/summary.md)
- Prompt v2 dev R=3: [`results/reports/h1_phase1_qwen25_gpu2-79205f0481-f68c87f537/summary.md`](../results/reports/h1_phase1_qwen25_gpu2-79205f0481-f68c87f537/summary.md)
- Prompt v2 held-out: [`results/reports/h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f/summary.md`](../results/reports/h1_phase1_qwen25_gpu2-09092ebd5e-d03859f54f/summary.md)

### Console logs

- 24-task R=1 runs: [`results/logs/phase1_24x1_console.log`](../results/logs/phase1_24x1_console.log)
- 24-task R=3 stability run: [`results/logs/phase1_24x3_console.log`](../results/logs/phase1_24x3_console.log)
- Held-out run: [`results/logs/phase1_offset24_140x1_console.log`](../results/logs/phase1_offset24_140x1_console.log)

### Append-only source records

- `results/manifests.jsonl`
- `results/baseline.jsonl`
- `results/checkpoint_states.jsonl`
- `results/intervention_pairs.jsonl`
- `results/llm_invocations.jsonl`
- `results/evaluations.jsonl`
- `results/events.jsonl`

## 13. Tái tạo visualization

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
python scripts/build_final_report_241026.py
```

Các hình và CSV sẽ được tạo lại trong `results/final_report_241026/`. Các biểu đồ định lượng được tính từ run-specific `h1_analysis.csv`; ba case-study annotations là snapshot có chủ đích của artifact logs.

---

# Phần bổ sung Phase 2 — Train Downstream Verification Value Predictor

> Phần này ghi lại **những thay đổi đã thực hiện, mục đích của từng thay đổi và intended flow**. Mục tiêu là để khi đọc lại có thể phân biệt rõ thiết kế dự kiến, cách evaluation thực sự hoạt động và kết quả thu được. Phase 2 chỉ dùng dữ liệu Phase 1 đã lưu; không gọi lại Student, Teacher hoặc HumanEval evaluator.

## 14. Vì sao chuyển sang Phase 2

Phase 1 cho thấy hai tín hiệu đồng thời:

1. Teacher có thể cải thiện kết quả cuối, nhưng phần lớn verification không đổi binary outcome.
2. Local heuristic risk không đủ khả năng xác định checkpoint nào có downstream value. Offline Oracle tốt hơn nhiều, nhưng Oracle biết trước outcome nên không thể triển khai.

Vì vậy thay đổi trọng tâm từ câu hỏi **“artifact hiện tại có lỗi không?”** sang:

> **“Nếu gọi Teacher tại checkpoint hiện tại, xác suất final success tăng bao nhiêu so với SKIP?”**

Mục tiêu Phase 2 không phải thay Teacher hoặc sửa workflow Planner–Coder–Reviewer. Mục tiêu là học một routing signal dùng trước verification:

```text
Q_skip   = P(final success | state hiện tại, SKIP)
Q_verify = P(final success | state hiện tại, VERIFY)
Delta_pred = Q_verify - Q_skip
```

## 15. Những thay đổi đã implement và mục đích

| Thay đổi | File chính | Mục đích |
|---|---|---|
| Leakage-safe data extraction | `src/agent_verify/analysis/phase2_data.py` | Chỉ tạo feature quan sát được trước verification; giữ label và task ID ngoài model input |
| Static artifact analysis | `phase2_data.py` | Trích syntax, AST depth, branches, loops, entry point, artifact length và metadata mà không chạy code |
| Two-head value predictor | `src/agent_verify/analysis/phase2_models.py` | Học riêng `Q_skip` và `Q_verify`, sau đó lấy hiệu để dự đoán intervention value |
| Metadata-only Logistic | `phase2_models.py` | Control baseline để đo semantic text có thực sự đóng góp hay không |
| TF-IDF Logistic | `phase2_models.py` | Mô hình chính, ưu tiên bias thấp và regularization vì chỉ có 417 pairs |
| SVD + HistGradientBoosting | `phase2_models.py` | Kiểm tra quan hệ phi tuyến giữa semantic representation và metadata |
| Nested GroupKFold | `phase2_models.py` | Tune hyperparameter/threshold mà không nhìn outer held-out fold |
| Sequential budget controller | `src/agent_verify/analysis/phase2.py` | Ra quyết định CP1→CP2→CP3, tối đa một Teacher call, cho phép SKIP cả ba |
| Paired offline policy evaluation | `phase2.py` | Dùng đúng counterfactual đã lưu tại checkpoint được chọn, không chạy inference lại |
| Task-level paired bootstrap | `phase2.py` | Tính CI mà vẫn giữ ba checkpoints cùng task phụ thuộc lẫn nhau |
| Leakage/controller unit tests | `tests/test_phase2_offline.py` | Bắt lỗi task leakage, post-treatment feature leakage và nhiều hơn một call/task |
| Reproduction environment | `pyproject.toml`, `scripts/train_phase2.py` | Pin `scikit-learn==1.7.2` và cung cấp entry point tái lập |

### Những feature được phép dùng

- HumanEval public task description; không dùng task ID làm feature.
- Checkpoint index, role và số downstream stages còn lại.
- Current artifact và preceding artifacts đã tồn tại tại thời điểm quyết định.
- Local heuristic risk.
- Static code/plan indicators và remaining verification budget.

### Những feature bị cấm

- `Y_skip`, `Y_verify`, observed Delta.
- Teacher-corrected artifact, verdict hoặc explanation.
- Final implementation và future checkpoint artifacts.
- Private HumanEval tests hoặc evaluator output.
- Task ID như một feature dự đoán.

## 16. Intended training flow

![Phase 2 training flow](../results/final_report_241026/phase2_training_flow.png)

Flow thực tế:

1. Nạp 417 paired records và khớp với 139 baseline task trajectories.
2. Tạo pre-verification features tại từng CP.
3. Outer GroupKFold chia theo `task_id`; cả CP1/CP2/CP3 của một task luôn nằm cùng fold.
4. Trong outer-train, inner 4-fold GroupKFold chọn hyperparameters và `Delta_pred` threshold.
5. Fit hai heads trên toàn outer-train.
6. Predict `Q_skip/Q_verify` cho outer-held-out tasks.
7. Ghép năm fold thành 1,251 OOF rows cho ba models.
8. Simulate controller tuần tự và dùng stored paired outcome tại checkpoint đã chọn.
9. Task-level paired bootstrap 20,000 lần để so sánh policies.

Điểm quan trọng: `final_model.joblib` được fit trên toàn bộ 139 tasks để phục vụ sử dụng sau này, nhưng **không được dùng để tính OOF metrics**. Mọi con số trong bảng Phase 2 đến từ held-out prediction của outer folds.

### Intended online controller

```text
CP1 state
  ├─ Delta_pred > threshold và còn budget? ─► VERIFY ─► stop routing
  └─ không ─► CP2 state
                ├─ Delta_pred > threshold? ─► VERIFY ─► stop routing
                └─ không ─► CP3 state
                              ├─ Delta_pred > threshold? ─► VERIFY
                              └─ không ─► SKIP all
```

Controller không được xem risk/prediction của checkpoint tương lai trước khi quyết định checkpoint hiện tại. Policy `Highest Risk (offline lookahead)` vẫn được giữ trong CSV nhưng được ghi rõ là diagnostic, không phải deployable baseline.

## 17. Quá trình training của ba model

![Phase 2 model training diagnostics](../results/final_report_241026/phase2_model_training_diagnostics.png)

Ba model được chọn vì ba mục đích khác nhau:

| Model | Input | Vai trò trong nghiên cứu |
|---|---|---|
| Metadata-only Logistic | Numeric/static metadata | Đo baseline không có semantic text |
| Ours Logistic | TF-IDF task/artifacts + metadata | Main small-data model; dễ regularize và diễn giải |
| Ours Boosting | TF-IDF→SVD + metadata | Kiểm tra nonlinear interactions với dense representation |

Mỗi model có hai classifiers dùng cùng pre-verification encoder:

- Head SKIP học từ `Y_skip`.
- Head VERIFY học từ `Y_verify`.
- Controller dùng hiệu hai probability heads, không trực tiếp train trên task ID hoặc Oracle choice.

Threshold khác nhau giữa các outer folds vì được tune độc lập trong inner validation. Đây là biểu hiện của sample nhỏ và intervention signal thưa, không nên hiểu threshold của một fold là giá trị universal.

## 18. Kết quả Phase 2

![Phase 2 policy comparison](../results/phase2/policy_comparison.png)

![Phase 2 accuracy versus cost](../results/phase2/accuracy_vs_cost.png)

| Method | Accuracy | Gain vs Never | Teacher calls | Teacher tokens | Helpful / Harmful |
|---|---:|---:|---:|---:|---:|
| Never Verify | 81.29% | +0.00 pp | 0 | 0 | 0 / 0 |
| Local Risk Controller | 82.73% | +1.44 pp | 7 | 11,817 | 2 / 0 |
| Metadata-only Logistic | 82.73% | +1.44 pp | 48 | 79,108 | 2 / 0 |
| **Ours Logistic** | **83.45%** | **+2.16 pp** | **65** | **103,355** | **3 / 0** |
| Ours Boosting | 82.73% | +1.44 pp | 87 | 136,577 | 3 / 1 |
| Always Verify CP3 | **86.33%** | **+5.04 pp** | 139 | 252,062 | 8 / 1 |
| Offline Oracle | 89.93% | +8.63 pp | 12 | 20,529 | 12 / 0 |

Paired comparisons:

- Ours Logistic − Always CP3: **−2.88 pp**, 95% CI **[−6.47, 0.00]**.
- Ours Boosting − Always CP3: **−3.60 pp**, 95% CI **[−8.63, +0.72]**.
- Ours Logistic − Local Risk: **+0.72 pp**, 95% CI **[−1.44, +2.88]**.
- Ours Boosting − Local Risk: **0.00 pp**, 95% CI **[−3.60, +3.60]**.

Ours Logistic giảm 74/139 Teacher calls, tương đương **53.2%**, và giảm khoảng **59.0% Teacher tokens** so với Always CP3. Tuy nhiên, point accuracy thấp hơn 2.88 pp và CI chạm 0; dữ liệu hiện tại chưa đủ để tuyên bố giữ nguyên accuracy.

### Predictor quality

| Predictor | Delta Pearson | Delta Spearman | Helpful AUROC | Helpful AP |
|---|---:|---:|---:|---:|
| Local Risk heuristic | 0.223 | 0.065 | 0.610 | 0.125 |
| Metadata-only Logistic | 0.126 | 0.077 | 0.598 | 0.098 |
| Ours Logistic | 0.142 | 0.102 | 0.592 | 0.083 |
| Ours Boosting | −0.014 | 0.019 | 0.487 | 0.048 |

Kết quả âm tính quan trọng: semantic predictors chưa dự đoán intervention benefit tốt hơn Local Risk theo Pearson hoặc Helpful AP. Logistic có policy accuracy cao hơn Local Risk 0.72 pp chủ yếu nhờ threshold/routing behavior, không phải vì đã học được Delta ranking mạnh.

## 19. Ý nghĩa của thay đổi và intended next step

### Những gì Phase 2 đã xác nhận

- Pipeline train/evaluate chống task leakage hoạt động và tái lập được.
- Có thể giảm mạnh Teacher calls bằng controller tuần tự.
- Phân tách `Q_skip/Q_verify` cho phép biểu diễn đúng câu hỏi decision value.
- OOF policy evaluation cho phép so sánh công bằng với Never/Always/Local Risk trên cùng tasks.

### Những gì Phase 2 chưa đạt được

- Learned predictor chưa vượt Always CP3 về final accuracy.
- Semantic features chưa tốt hơn heuristic risk về benefit ranking.
- Boosting không giúp và có một harmful selected intervention.
- Chỉ 19 helpful và 5 harmful pairs khiến threshold/hyperparameter thay đổi đáng kể giữa folds.
- Tất cả evidence vẫn đến từ một benchmark và một cặp Child/Teacher models.

### Intent khi tiếp tục

Không tự động chuyển sang Phase 3. Bước hợp lý tiếp theo là mở rộng data trước khi tăng model complexity:

1. Thu thập thêm tasks/benchmarks và nhiều repeated interventions thực sự đa dạng.
2. Giữ một external test set chưa từng dùng trong H1/Phase 2.
3. Cân nhắc target trực tiếp theo intervention class hoặc doubly robust/off-policy formulation khi data đủ lớn.
4. Thêm uncertainty/calibration và abstention thay vì chỉ threshold point estimate.
5. Chỉ tiếp tục nếu learned controller thắng baseline mạnh như Always CP3 hoặc tạo Pareto improvement rõ ràng giữa accuracy và cost.

## 20. Phase 2 artifacts và cách tái tạo

- Full Phase 2 report: [`results/phase2/phase2_report.md`](../results/phase2/phase2_report.md)
- OOF predictions: [`results/phase2/out_of_fold_predictions.csv`](../results/phase2/out_of_fold_predictions.csv)
- Predictor metrics: [`results/phase2/predictor_metrics.csv`](../results/phase2/predictor_metrics.csv)
- Policy comparison: [`results/phase2/policy_comparison.csv`](../results/phase2/policy_comparison.csv)
- Fold selections: [`results/phase2/fold_selections.json`](../results/phase2/fold_selections.json)
- Model checkpoints: `results/phase2/model_checkpoints/`
- Training implementation: [`src/agent_verify/analysis/phase2.py`](agent_verify/analysis/phase2.py)
- Feature extraction: [`src/agent_verify/analysis/phase2_data.py`](agent_verify/analysis/phase2_data.py)
- Predictors/nested CV: [`src/agent_verify/analysis/phase2_models.py`](agent_verify/analysis/phase2_models.py)
- Visualization script: [`scripts/build_phase2_report_visuals.py`](../scripts/build_phase2_report_visuals.py)

Tái tạo Phase 2 và hình bổ sung:

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify

python results/phase2/train_predictor.py --output-dir results/phase2
python scripts/build_phase2_report_visuals.py
```

Validation sau thay đổi:

- Toàn bộ **22 unit tests** pass.
- Ruff lint pass.
- 1,251 OOF rows = 3 models × 417 checkpoint pairs.
- Mỗi model–task có đúng ba checkpoints trong cùng outer fold.
- Mỗi controller chọn không quá một verification cho mỗi task.
- Run manifest xác nhận không gọi LLM, Teacher hoặc evaluator trong Phase 2.

## 21. Tối ưu exploratory cho Phase 2

Mục tiêu của vòng này là kiểm tra xem Boosting yếu do family model hay do search space và routing policy còn quá đơn giản. Toàn bộ quá trình chỉ dùng 417 paired records đã có; không gọi lại Child, Teacher hoặc evaluator.

![Phase 2 optimization comparison](../results/phase2_optimization/optimization_comparison.png)

### Những gì đã thay đổi và mục đích

| Thay đổi | Mục đích |
|---|---|
| Mở rộng Boosting từ 4 lên 45 cấu hình | Kiểm tra SVD dimension, tree capacity và L2 regularization kỹ hơn |
| Dùng ba ngưỡng riêng cho CP1/CP2/CP3 | Phản ánh verification sớm và muộn có downstream value khác nhau |
| Ràng buộc `threshold_CP1 ≥ threshold_CP2 ≥ threshold_CP3` | Chỉ spend budget sớm khi evidence mạnh; cho phép chờ checkpoint sau |
| Chọn config/ngưỡng trong inner 4-fold GroupKFold | Không dùng nhãn của outer-held-out task để ra quyết định |
| Đánh giá bằng 5-fold nested OOF + 20.000 task bootstraps | Giữ ba checkpoints cùng task trong một fold và đo uncertainty đúng đơn vị task |

Flow tối ưu:

```text
Outer train tasks
  └─ 4-fold inner GroupKFold
       ├─ 45 Boosting configurations
       ├─ 286 monotone checkpoint-threshold profiles
       └─ chọn accuracy cao nhất, rồi ưu tiên ít Teacher calls hơn
            └─ refit trên toàn outer-train
                 └─ predict outer-held-out tasks
                      └─ sequential controller: tối đa 1 VERIFY / task
```

### Kết quả

| Method | Accuracy | Gain vs Never | Teacher calls | Helpful / Harmful |
|---|---:|---:|---:|---:|
| Original Ours Logistic | 83.45% | +2.16 pp | 65 | 3 / 0 |
| Original Ours Boosting | 82.73% | +1.44 pp | 87 | 3 / 1 |
| **Optimized Boosting** | **84.89%** | **+3.60 pp** | **105** | **6 / 1** |
| Always Verify CP3 | 86.33% | +5.04 pp | 139 | 8 / 1 |

Optimized Boosting tăng **+2.16 pp** so với Boosting cũ và **+1.44 pp** so với Logistic cũ. So với Always CP3, nó thấp hơn **1.44 pp**, nhưng tiết kiệm **34 Teacher calls** và khoảng **70.429 Teacher tokens**.

Các paired intervals quan trọng:

- Optimized − Original Boosting: **+2.16 pp**, 95% CI **[−2.16, +7.19]**.
- Optimized − Local Risk: **+2.16 pp**, 95% CI **[−1.44, +5.76]**.
- Optimized − Always CP3: **−1.44 pp**, 95% CI **[−4.32, +1.44]**.

OOF predicted-Delta Pearson correlation tăng từ **−0.014** ở Boosting cũ lên **0.180**, nhưng vẫn thấp hơn Local Risk (**0.223**). Vì vậy phần tăng policy accuracy có vẻ đến đáng kể từ checkpoint-aware routing, chưa phải do model đã học một benefit ranking mạnh.

### Diễn giải khoa học

Đây là một **adaptive exploratory optimization trên cùng 139 tasks**, không phải independent validation. Point estimate tốt hơn nhưng mọi confidence interval so sánh với learned baselines đều chứa 0. Cấu hình được chọn cũng thay đổi nhiều giữa năm folds, phù hợp với việc chỉ có 19 helpful và 5 harmful checkpoint pairs.

Kết luận đúng ở thời điểm này:

- Pipeline tối ưu đã cải thiện point accuracy và predictor correlation.
- Chưa đủ bằng chứng thống kê để tuyên bố model tối ưu tốt hơn model cũ.
- Chưa vượt baseline mạnh Always CP3 về accuracy.
- Nên freeze specification hiện tại và test trên tasks/benchmark mới trước khi tune thêm.

### Artifacts và tái tạo

- Optimization report: [`results/phase2_optimization/optimization_report.md`](../results/phase2_optimization/optimization_report.md)
- Policy comparison: [`optimized_policy_comparison.csv`](../results/phase2_optimization/optimized_policy_comparison.csv)
- OOF predictions: [`optimized_oof_predictions.csv`](../results/phase2_optimization/optimized_oof_predictions.csv)
- Fold selections: [`fold_selections.json`](../results/phase2_optimization/fold_selections.json)
- Model checkpoints: `results/phase2_optimization/model_checkpoints/`
- Implementation: [`phase2_optimization.py`](agent_verify/analysis/phase2_optimization.py)
- CLI: [`optimize_phase2_models.py`](../scripts/optimize_phase2_models.py)

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
python scripts/optimize_phase2_models.py --output-dir results/phase2_optimization
```

Validation sau tối ưu:

- Toàn bộ **23 unit tests** pass và Ruff lint pass.
- 417 OOF checkpoint rows thuộc đúng 139 tasks.
- Controller chọn đúng 105 calls và không task nào bị VERIFY quá một lần.
- Cả năm serialized fold models load thành công.
- Manifest xác nhận không gọi LLM, Teacher hoặc evaluator.

## 22. Phase 3 — LiveCodeBench v6, 100 task đầu sau recovery hoàn chỉnh

Mục tiêu của Phase 3 là kiểm tra Downstream-Aware Verification trên một benchmark mới hơn và khó hơn HumanEval, đồng thời giữ nguyên workflow `Planner → CP1 → Coder → CP2 → Reviewer → CP3`. Run này sử dụng 100 task được khóa trước bằng seed 42: 43 Easy, 52 Medium và 5 Hard; Student là Qwen2.5-Coder-7B-Instruct-AWQ và Teacher là Qwen2.5-Coder-32B-Instruct-AWQ.

> **Trạng thái cuối: COMPLETE.** Baseline đủ **100/100** và intervention pairs đủ **300/300**. Pair `LiveCodeBench/abc400_d`, CP3 từng thiếu vì output Teacher chạm context 4096; sau khi tăng context lên 8192 và resume đúng run ID, pair cuối đã hoàn thành. Các estimate CP1/CP2/CP3 dưới đây là task-complete.

![LiveCodeBench 100-task dashboard](../results/livecodebench/report/livecodebench_100_dashboard.png)

### 22.1. Độ đầy đủ và recovery audit

| Checkpoint | Hoàn thành | Thiếu | Trạng thái |
|---|---:|---:|---|
| CP1 — Planner | 100/100 | 0 | Complete |
| CP2 — Coder | 100/100 | 0 | Complete |
| CP3 — Reviewer | 100/100 | 0 | Complete |
| **Tổng** | **300/300** | **0** | Có thể dùng cho downstream analysis |

Ở lần chạy đầu, 60 lỗi CP3 chủ yếu do Teacher trả `CodeArtifact` thay vì `ReviewArtifact`; CP2 còn thiếu một response. Sau khi tách response schema theo checkpoint và resume đúng run ID, pipeline phục hồi các pair thiếu mà không chạy lại những pair đã hoàn thành. Pair `abc400_d/CP3` sau đó được lấp bằng context 8192.

Event log vẫn giữ 62 `paired_intervention_failure` và 61 `teacher_verification_failure` lịch sử theo thiết kế append-only. Đây là audit trail của các lần thử cũ, không phải số pair còn thiếu. Trạng thái canonical nằm trong `intervention_pairs.jsonl` và hiện đủ 300 pair.

### 22.2. Baseline LiveCodeBench

| Nhóm | Correct / Total | Accuracy |
|---|---:|---:|
| Overall | 30/100 | **30.00%** |
| Easy | 24/43 | **55.81%** |
| Medium | 6/52 | **11.54%** |
| Hard | 0/5 | **0.00%** |
| AtCoder | 20/55 | 36.36% |
| LeetCode | 10/45 | 22.22% |

Baseline execution gồm 30 PASS, 58 FAIL, 8 RUNTIME_ERROR và 4 TIMEOUT. Khoảng cách Easy–Medium rất lớn, cho thấy distribution shift rõ so với HumanEval.

### 22.3. Verification effect theo checkpoint

| CP | Pairs | SKIP accuracy | VERIFY accuracy | Net gain | 95% task-bootstrap CI | Helpful / Harmful / No change |
|---:|---:|---:|---:|---:|---:|---:|
| CP1 | 100 | 30.00% | 30.00% | **0.00 pp** | [−5.00, +5.00] | 3 / 3 / 94 |
| CP2 | 100 | 30.00% | 37.00% | **+7.00 pp** | [+2.00, +13.00] | 8 / 1 / 91 |
| CP3 | 100 | 30.00% | 36.00% | **+6.00 pp** | [+2.00, +11.00] | 6 / 0 / 94 |

CP1 không tạo lợi ích ròng: ba helpful bị triệt tiêu bởi ba harmful. CP2 là checkpoint tốt nhất trên 100 task đầu, tăng bảy bài đúng và confidence interval không chứa 0. CP3 tăng sáu bài đúng, không có harmful. Toàn bộ net gain CP2/CP3 đến từ Easy; Medium và năm Hard task đầu không có cải thiện ròng.

Trong 300 pair có 17 helpful, 4 harmful và 279 no-change, tương ứng 5.67%, 1.33% và 93.00%. Mean paired Delta là **+4.33 pp**. Đây là trung bình counterfactual trên checkpoint-pairs, không phải accuracy của một policy triển khai.

### 22.4. Local Risk và H1

| CP | Risk support | Pearson risk–Delta | Spearman risk–Delta |
|---:|---|---:|---:|
| CP1 | Tất cả score = 0.20 | Không thể ranking | Không xác định |
| CP2 | 0.15 (45), 0.80 (54), 1.00 (1) | 0.009 | 0.005 |
| CP3 | 0.15 (45), 0.80 (54), 1.00 (1) | −0.027 | −0.028 |

CP1 hoàn toàn không ranking được task; CP2 và CP3 chỉ có ba mức risk và correlation với Delta gần 0. Kết quả tiếp tục ủng hộ động lực của H1: Local Error Risk hiện tại không đủ để biểu diễn Downstream Verification Value. Đây là evidence quan sát, không chứng minh một learned predictor chắc chắn generalize.

### 22.5. Chi phí thực tế

| Role | Calls | Tokens | Cumulative model latency |
|---|---:|---:|---:|
| Planner | 100 | 90,319 | 256.3 s |
| Coder | 300 | 326,208 | 734.6 s |
| Reviewer | 502 | 704,906 | 1,521.0 s |
| Teacher | 362 | 809,258 | 3,987.4 s |
| Local Critic | 300 | 384,680 | 0.1 s |
| **Tổng** | **1,564** | **2,315,371** | **6,499.4 s** |

Call count cao hơn canonical workflow vì bao gồm các lần recovery thất bại được giữ để audit. Giá USD bằng 0 vì hai model được phục vụ local qua vLLM.

### 22.6. Kết luận run 100 task

1. Baseline chỉ đạt 30%, chủ yếu nhờ Easy.
2. CP1 verification không có lợi ích ròng.
3. CP2 đạt +7 pp và CP3 đạt +6 pp; cả hai confidence interval đều không chứa 0.
4. Lợi ích chỉ xuất hiện ở Easy; chưa có evidence lợi ích trên Medium/Hard.
5. Local Risk có resolution thấp và correlation gần 0 với Delta.
6. Run đã đủ 300/300 pair và có thể dùng cho downstream analysis.

Artifacts:

- [`results/livecodebench/h1_analysis.md`](../results/livecodebench/h1_analysis.md)
- [`results/livecodebench/accuracy_by_checkpoint.csv`](../results/livecodebench/accuracy_by_checkpoint.csv)
- [`results/livecodebench/helpful_harmful.csv`](../results/livecodebench/helpful_harmful.csv)
- [`livecodebench_100_dashboard.png`](../results/livecodebench/report/livecodebench_100_dashboard.png)

## 23. Phase 3 — 75 Hard tasks mở rộng

Run Hard extension sử dụng 75 task còn lại của frozen LiveCodeBench v6 selection 175 task. Vì 100 task đầu đã chứa 5 Hard task, 75 task này cùng 5 task trước tạo thành toàn bộ **80 Hard tasks** trong selection. Workflow, model, seed và counterfactual protocol được giữ nguyên để kết quả có thể đối chiếu với run 100 task.

> **Trạng thái: PRELIMINARY / 1 PAIR MISSING.** Baseline đủ **75/75**; intervention pairs đạt **224/225**. CP2 và CP3 đủ 75/75; CP1 có 74/75. Pair thiếu là `LiveCodeBench/arc191_d`, CP1, do response JSON bị cắt với `JSONDecodeError: Unterminated string`. Không chuyển pair này thành `Y=0`.

![LiveCodeBench Hard-75 dashboard](../results/livecodebench_hard_extension/report/livecodebench_hard_extension_dashboard.png)

### 23.1. Recovery evaluator timeout

Lần chạy đầu dừng ở `LiveCodeBench/abc391_f`. Student sinh lời giải ba vòng lặp `O(N³)` cho `N ≤ 200,000`; worker chạm CPU limit nhưng runner cũ gắn nhãn `EVALUATOR_ERROR`. Evaluator đã được sửa để:

- cấp CPU budget cho wrapper theo số test;
- giữ per-test timeout của official evaluator;
- ánh xạ `SIGXCPU`/`SIGKILL` do resource limit thành model outcome `TIMEOUT`;
- chỉ dừng experiment với lỗi hạ tầng thật.

Sau bản vá, chính artifact `abc391_f` được kiểm thử lại và trả `TIMEOUT` sau 9.68 giây. Resume giữ nguyên 38 baseline đã có và hoàn tất 75 baseline. Event `baseline_failure` cũ vẫn tồn tại trong audit log nhưng không còn là missing record.

### 23.2. Baseline Hard-75

| Nhóm | Correct / Total | Accuracy |
|---|---:|---:|
| Hard extension | 3/75 | **4.00%** |
| Toàn bộ Hard trong frozen-175 | 3/80 | **3.75%** |

Baseline Hard-75 gồm 3 PASS, 52 FAIL, 10 RUNTIME_ERROR, 7 TIMEOUT và 3 FORMAT_ERROR. So với 100 task đầu đạt 30%, mức 4% xác nhận difficulty là yếu tố chính và Student 7B đang ở gần floor trên tập này.

### 23.3. Verification effect trên Hard-75

| CP | Pairs | SKIP accuracy | VERIFY accuracy | Net gain | 95% task-bootstrap CI | Helpful / Harmful / No change |
|---:|---:|---:|---:|---:|---:|---:|
| CP1* | 74/75 | 4.05% | 4.05% | **0.00 pp** | [0.00, 0.00] | 0 / 0 / 74 |
| CP2 | 75/75 | 4.00% | 4.00% | **0.00 pp** | [0.00, 0.00] | 0 / 0 / 75 |
| CP3 | 75/75 | 4.00% | 4.00% | **0.00 pp** | [0.00, 0.00] | 0 / 0 / 75 |

Dấu `*` nghĩa là CP1 thiếu `arc191_d`. Trong **224 pair đã hoàn thành**, không có helpful và cũng không có harmful; toàn bộ Delta bằng 0.

Điểm quan trọng là `Delta=0` không có nghĩa Teacher không làm gì. Teacher thay artifact ở:

| CP | Artifact changed | Tỷ lệ | Final correctness flips |
|---:|---:|---:|---:|
| CP1* | 72/74 | 97.30% | 0 |
| CP2 | 65/75 | 86.67% | 0 |
| CP3 | 59/75 | 78.67% | 0 |
| **Tổng** | **196/224** | **87.50%** | **0** |

Teacher can thiệp rất thường xuyên nhưng các sửa đổi không đủ để đưa lời giải qua private tests. Đây là khác biệt giữa **artifact-level correction** và **downstream verification value**. Trên hard cases, bottleneck có thể nằm ở năng lực lập thuật toán/triển khai của cả Student lẫn Teacher, không chỉ ở việc chọn sai checkpoint.

### 23.4. Chi phí Hard-75

| Role | Calls | Tokens | Cumulative model latency |
|---|---:|---:|---:|
| Planner | 76 | 83,319 | 229.6 s |
| Coder | 226 | 309,406 | 841.3 s |
| Reviewer | 376 | 673,854 | 1,736.2 s |
| Teacher | 225 | 521,361 | 2,545.5 s |
| Local Critic | 228 | 341,550 | 0.1 s |
| **Tổng** | **1,131** | **1,929,490** | **5,352.7 s** |

Các call vượt canonical count đến từ recovery của `abc391_f`. Có 129 consistency warnings ở trường `Reviewer.changed`; pipeline dùng code diff thực tế làm nguồn sự thật nên warning không làm mất record. Teacher chiếm 521,361 tokens và gần 47.6% cumulative model latency nhưng tạo 0 correctness flips trên Hard-75.

### 23.5. Tổng quan frozen-175

| Metric | 100 task đầu | Hard-75 extension | Toàn bộ frozen-175 |
|---|---:|---:|---:|
| Baseline correct | 30/100 | 3/75 | **33/175 (18.86%)** |
| Completed pairs | 300/300 | 224/225 | **524/525** |
| Helpful | 17 | 0 | **17** |
| Harmful | 4 | 0 | **4** |
| No change | 279 | 224 | **503** |

| CP | Completed pairs | SKIP accuracy | VERIFY accuracy | Net gain | Helpful / Harmful / No change |
|---:|---:|---:|---:|---:|---:|
| CP1* | 174/175 | 18.97% | 18.97% | **0.00 pp** | 3 / 3 / 168 |
| CP2 | 175/175 | 18.86% | 22.86% | **+4.00 pp** | 8 / 1 / 166 |
| CP3 | 175/175 | 18.86% | 22.29% | **+3.43 pp** | 6 / 0 / 169 |

Việc thêm Hard-75 làm hiệu quả tổng thể bị pha loãng: CP2 giảm từ +7.00 pp xuống +4.00 pp và CP3 giảm từ +6.00 pp xuống +3.43 pp, vì 75 Hard task bổ sung không có correctness flip nào. Không nên diễn giải đây là Teacher “không bao giờ hữu ích”; kết luận đúng là lợi ích quan sát được hiện tập trung ở Easy và không transfer sang Hard distribution.

### 23.6. Kết luận nghiên cứu và bước tiếp theo

1. Student 7B gần floor trên Hard: baseline 4%, nên downstream agents và Teacher thường không đủ sức biến sửa đổi cục bộ thành lời giải đúng.
2. Teacher sửa 87.5% artifact nhưng tạo 0 final correctness flips. Điều này trực tiếp củng cố lý do cần đo downstream value thay vì dùng `changed` hay verdict làm proxy.
3. Hard-75 không cung cấp positive Delta examples cho việc học predictor; nếu trộn không kiểm soát, class imbalance sẽ tăng mạnh.
4. Chưa được gọi frozen-175 là complete dataset vì còn thiếu `arc191_d/CP1`.
5. Bước kỹ thuật gần nhất là phục hồi đúng một pair còn thiếu. Sau đó đóng băng 525/525 pairs, tái tạo report, rồi mới quyết định train lại predictor theo difficulty-aware sampling hoặc đổi Teacher mạnh hơn cho Hard tasks.

Artifacts:

- [`results/livecodebench_hard_extension/h1_analysis.md`](../results/livecodebench_hard_extension/h1_analysis.md)
- [`results/livecodebench_hard_extension/accuracy_by_checkpoint.csv`](../results/livecodebench_hard_extension/accuracy_by_checkpoint.csv)
- [`results/livecodebench_hard_extension/helpful_harmful.csv`](../results/livecodebench_hard_extension/helpful_harmful.csv)
- [`livecodebench_hard_extension_dashboard.png`](../results/livecodebench_hard_extension/report/livecodebench_hard_extension_dashboard.png)
- [`livecodebench_hard_extension_dashboard_checkpoint_metrics.csv`](../results/livecodebench_hard_extension/report/livecodebench_hard_extension_dashboard_checkpoint_metrics.csv)

Rebuild dashboard (không gọi LLM):

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
python scripts/build_livecodebench_report_visuals.py \
  --results-dir results/livecodebench_hard_extension \
  --output-dir results/livecodebench_hard_extension/report \
  --run-id h1_livecodebench_v6_hard_extension_qwen25_gpu2-d6cd8c8193-28e3450171 \
  --output-name livecodebench_hard_extension_dashboard.png
```

## 24. Phase 2 mở rộng — Gộp HumanEval + LiveCodeBench và tối ưu ML

### 24.1. Mục tiêu thay đổi

Bước này tiếp tục flow Phase 2 nhưng mở rộng training universe từ HumanEval sang dữ liệu hợp nhất HumanEval + LiveCodeBench. Mục tiêu là kiểm tra learned Downstream Verification Value predictor khi benchmark và difficulty đa dạng hơn, đồng thời giữ một frozen outer test độc lập để đánh giá trước/sau tối ưu.

Không chạy lại Student, Teacher hoặc evaluator. Tất cả labels đều lấy từ các Verify/Skip pairs đã thu thập.

### 24.2. Data hợp nhất và xử lý task thiếu

| Data | Tasks | Checkpoint rows | Helpful | Harmful | No change | Baseline accuracy |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval | 139 | 417 | 19 | 5 | 393 | 81.29% |
| LiveCodeBench | 174 | 522 | 17 | 4 | 501 | 18.97% |
| **Combined** | **313** | **939** | **36** | **9** | **894** | **46.65%** |

`LiveCodeBench/arc191_d` bị loại toàn bộ khỏi ML dataset vì chỉ có CP2/CP3 và thiếu CP1. Không giữ hai checkpoint lẻ vì policy sequence cần đủ CP1/CP2/CP3 trên cùng một task; loại nguyên task cũng tránh selection bias và sai lệch cost simulation.

![Combined Phase-2 data overview](../results/combined_phase2_optimized/combined_data_overview.png)

### 24.3. Leakage-safe protocol

- Tổng task hợp lệ: 313; mỗi task đúng ba checkpoint rows.
- Train: 250 tasks / 750 rows.
- Frozen test: 63 tasks / 189 rows.
- Test gồm 28 HumanEval, 9 LiveCodeBench Easy, 10 Medium và 16 Hard.
- Split deterministic theo `task_id`, stratify theo benchmark/difficulty và không dùng outcome.
- CP1/CP2/CP3 của cùng task luôn nằm cùng split và cùng inner fold.
- Train/test overlap bằng 0.
- `task_id`, `Y_skip`, `Y_verify`, `Delta`, Teacher output và future artifacts không phải model features.
- Hyperparameter và threshold chỉ được chọn trong 4-fold `GroupKFold(task_id)` trên train.
- Frozen test chỉ được evaluate sau khi khóa toàn bộ lựa chọn.

Audit được lưu tại `leakage_audit.json`, `fixed_test_tasks.json` và `split_assignments.csv`.

### 24.4. Model và tối ưu

Ba model compact được train trước:

1. Metadata-only Logistic.
2. Semantic Logistic dùng TF-IDF + metadata.
3. Semantic Boosting dùng TF-IDF → SVD + metadata.

Sau đó Optimized Boosting mở rộng từ bốn lên 45 cấu hình, kết hợp 286 monotone checkpoint-threshold profiles với ràng buộc `threshold_CP1 ≥ threshold_CP2 ≥ threshold_CP3`.

Cấu hình được chọn từ train-only inner folds:

- SVD components = 24.
- Max leaf nodes = 3.
- L2 = 1.0.
- Thresholds = `(CP1 0.30, CP2 0.10, CP3 −0.05)`.

### 24.5. Frozen-test result — 63 combined tasks

| Method | Accuracy | Gain vs Never | 95% gain CI | Calls | Teacher tokens | Helpful / Harmful |
|---|---:|---:|---:|---:|---:|---:|
| Never Verify | 39.68% | +0.00 pp | [0.00, 0.00] | 0 | 0 | 0 / 0 |
| Local Risk | 39.68% | +0.00 pp | [0.00, 0.00] | 22 | 49,474 | 0 / 0 |
| Ours Logistic | 42.86% | +3.17 pp | [−3.17, +9.52] | 54 | 100,859 | 3 / 1 |
| Compact Boosting | 42.86% | +3.17 pp | [0.00, +7.94] | 15 | 28,861 | 2 / 0 |
| **Optimized Boosting** | **44.44%** | **+4.76 pp** | **[−1.59, +12.70]** | **62** | **135,504** | **4 / 1** |
| Always Verify CP2 | 44.44% | +4.76 pp | [0.00, +11.11] | 63 | 121,856 | 3 / 0 |
| Always Verify CP3 | 42.86% | +3.17 pp | [−3.17, +9.52] | 63 | 141,723 | 3 / 1 |
| Offline Oracle | 49.21% | +9.52 pp | [+3.17, +17.46] | 6 | 10,082 | 6 / 0 |

![Combined Phase-2 model optimization](../results/combined_phase2_optimized/combined_model_optimization.png)

### 24.6. Trước và sau expanded optimization

| Scope | Compact | Optimized | Difference | Paired 95% CI | Calls compact → optimized |
|---|---:|---:|---:|---:|---:|
| Combined (63) | 42.86% | 44.44% | **+1.59 pp** | [−6.35, +9.52] | 15 → 62 |
| HumanEval (28) | 71.43% | 75.00% | **+3.57 pp** | [−10.71, +17.86] | 5 → 28 |
| LiveCodeBench (35) | 20.00% | 20.00% | **0.00 pp** | [−8.57, +8.57] | 10 → 34 |

Optimized Boosting tăng đúng một task trên combined test, nhưng paired CI chứa 0. Nó dùng thêm 47 Teacher calls và 106,643 tokens so với Compact Boosting. Threshold CP3 âm khiến controller gần như luôn verify; do đó point estimate tốt hơn nhưng không cost-efficient.

### 24.7. Distribution shift và predictor quality

| Frozen scope | Never | Logistic | Compact Boosting | Optimized Boosting | Always CP2 | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval (28) | 67.86% | 67.86% | 71.43% | **75.00%** | 71.43% | 82.14% |
| LiveCodeBench (35) | 17.14% | **22.86%** | 20.00% | 20.00% | **22.86%** | 22.86% |

Optimization chỉ tạo point gain trên HumanEval subset. Trên LiveCodeBench, Semantic Logistic và Always-CP2 đạt 22.86%, cao hơn hai Boosting variants.

| Predictor | Delta Pearson | Helpful AUROC | Helpful AP | Joint log loss |
|---|---:|---:|---:|---:|
| Local Risk | −0.074 | 0.394 | 0.041 | — |
| Ours Logistic | 0.035 | 0.617 | 0.075 | 0.510 |
| Compact Boosting | **0.110** | 0.568 | **0.134** | 0.479 |
| Optimized Boosting | 0.092 | **0.701** | 0.110 | **0.478** |

Expanded optimization cải thiện Helpful AUROC và log loss rất nhỏ, nhưng Delta correlation và average precision không tăng. Policy gain phần lớn đến từ threshold mạnh tay hơn, chưa phải bằng chứng predictor hiểu downstream value tốt hơn.

### 24.8. Kết luận và quyết định khoa học

1. Data gộp rất mất cân bằng: 894/939 rows có `Delta=0`.
2. Leakage audit đạt yêu cầu; outer test thực sự độc lập với train/tuning.
3. Expanded optimization có point estimate +1.59 pp so với Compact, nhưng CI [−6.35, +9.52] chưa loại trừ không cải thiện.
4. Expanded model gần như Always Verify và không có gain trên LiveCodeBench; không nên chọn nó làm cost-sensitive policy hiện tại.
5. Compact Boosting phù hợp hơn nếu ưu tiên chi phí; Semantic Logistic là learned policy tốt nhất trên LiveCodeBench frozen subset.
6. Không tiếp tục tune trên cùng frozen test sau khi đã xem kết quả. Vòng confirmatory tiếp theo cần một untouched split hoặc benchmark slice mới.
7. Kết quả HumanEval-only optimized cũ 84.89% và HumanEval frozen subset 75.00% không được trừ trực tiếp vì task universe/protocol khác nhau; so sánh hợp lệ là Compact và Optimized trên cùng 63 frozen tasks.

Artifacts:

- [`phase2_report.md`](../results/combined_phase2_optimized/phase2_report.md)
- [`policy_comparison_by_benchmark.csv`](../results/combined_phase2_optimized/policy_comparison_by_benchmark.csv)
- [`predictor_metrics_by_benchmark.csv`](../results/combined_phase2_optimized/predictor_metrics_by_benchmark.csv)
- [`optimization_delta.csv`](../results/combined_phase2_optimized/optimization_delta.csv)
- [`dataset_summary.csv`](../results/combined_phase2_optimized/dataset_summary.csv)
- [`leakage_audit.json`](../results/combined_phase2_optimized/leakage_audit.json)
- [`run_manifest.json`](../results/combined_phase2_optimized/run_manifest.json)
- [`combined_data_overview.png`](../results/combined_phase2_optimized/combined_data_overview.png)
- [`combined_model_optimization.png`](../results/combined_phase2_optimized/combined_model_optimization.png)
- [`scripts/train_combined_phase2.sh`](../scripts/train_combined_phase2.sh)

Reproduction:

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
bash scripts/train_combined_phase2.sh
```

## 25. LiveCodeBench-only training và so sánh với Combined

### 25.1. Thiết kế thí nghiệm

Để kiểm tra distribution shift quan sát ở mục 24, cùng pipeline được train lại chỉ bằng LiveCodeBench:

- Data hợp lệ: 174 tasks / 522 checkpoint rows.
- Train: 139 tasks / 417 rows.
- Frozen test: 35 tasks / 105 rows.
- 35 test tasks giống hoàn toàn với LiveCodeBench subset trong run Combined.
- Feature, model families, seed, 4-fold `GroupKFold`, 45 Boosting configurations, 286 monotone threshold profiles và 20,000 bootstrap repetitions được giữ nguyên.
- Không gọi Student, Teacher hoặc evaluator.

### 25.2. Cách đọc Model, Combined train và LCB-only train

Trong mục này, **Model** không phải Student Qwen2.5-Coder-7B hay Teacher Qwen2.5-Coder-32B. Đây là model ML nhỏ đứng ở tầng controller, nhận trạng thái trước verification và dự đoán:

```text
Q_skip   = P(final answer đúng | không verify checkpoint hiện tại)
Q_verify = P(final answer đúng | verify checkpoint hiện tại)
Predicted Delta = Q_verify - Q_skip
```

Ba dòng model có ý nghĩa:

- **Semantic Logistic:** Logistic Regression dùng TF-IDF của đề bài/artifact kết hợp metadata.
- **Compact Boosting:** Gradient Boosting cấu hình nhỏ, ưu tiên accuracy–cost trade-off.
- **Optimized Boosting:** Gradient Boosting với search space và checkpoint-specific thresholds lớn hơn.

Hai cột training mode chỉ khác nhau ở nguồn training data:

| Training mode | Dữ liệu train | Dữ liệu đánh giá |
|---|---|---|
| Combined train | 111 HumanEval + 139 LiveCodeBench = 250 tasks | 35 frozen LCB tasks |
| LCB-only train | 139 LiveCodeBench tasks | Cùng 35 frozen LCB tasks |

Hai phía dùng cùng model families, features, seed, tuning protocol và test set. Vì vậy phép so sánh trả lời câu hỏi: thêm HumanEval vào training có giúp policy quyết định checkpoint tốt hơn trên LiveCodeBench hay không.

Accuracy trong bảng bên dưới là **final coding accuracy sau khi controller chọn Verify/Skip**, không phải classification accuracy nội bộ của model ML. Ví dụ Compact Boosting 20.00% so với 22.86% nghĩa là Combined giải đúng 7/35 task, còn LCB-only giải đúng 8/35 task.

### 25.3. Cấu hình được chọn từ train-only

| Model | Cấu hình | Threshold |
|---|---|---:|
| Semantic Logistic | C=10, TF-IDF 4,000, no class weight | 0.05 |
| Compact Boosting | leaves=7, L2=1, SVD=24 | 0.30 |
| Optimized Boosting | leaves=15, L2=10, SVD=48 | CP1=0.30, CP2=0.05, CP3=−0.05 |

### 25.4. So sánh trên cùng frozen test

| Model | Combined train | LCB-only train | LCB-only − Combined | Paired 95% CI | Calls Combined → LCB-only | Tokens Combined → LCB-only |
|---|---:|---:|---:|---:|---:|---:|
| Semantic Logistic | **22.86%** | 20.00% | **−2.86 pp** | [−8.57, 0.00] | 27 → 15 | 57,322 → 29,178 |
| Compact Boosting | 20.00% | **22.86%** | **+2.86 pp** | [0.00, +8.57] | 10 → 6 | 20,022 → 11,681 |
| Optimized Boosting | 20.00% | **22.86%** | **+2.86 pp** | [0.00, +8.57] | 34 → 35 | 84,785 → 86,890 |

Các reference trên cùng 35 task: Never Verify = 17.14%, Always-CP2 = 22.86% với 35 calls và Offline Oracle = 22.86% với 2 calls.

![LiveCodeBench-only vs Combined](../results/livecodebench_only_phase2_optimized/livecodebench_training_mode_comparison.png)

### 25.5. Vì sao Compact Boosting tăng một task

LiveCodeBench-only Compact Boosting chọn đúng cả hai helpful interventions:

- `LiveCodeBench/3708` tại CP2, observed Delta +1.
- `LiveCodeBench/3811` tại CP2, observed Delta +1.

Combined Compact Boosting cũng bắt được `3708`, nhưng tại `3811` nó verify CP1 trước, nơi Delta bằng 0, nên đã tiêu single-verification budget trước khi tới helpful CP2. Đây là minh họa cụ thể rằng ranking và checkpoint-specific threshold ảnh hưởng trực tiếp đến sequential policy.

### 25.6. Predictor quality

| Model | Helpful AUROC Combined → LCB-only | Helpful AP Combined → LCB-only |
|---|---:|---:|
| Semantic Logistic | 0.772 → **0.917** | 0.058 → **0.146** |
| Compact Boosting | **0.951** → 0.947 | **0.341** → 0.208 |
| Optimized Boosting | 0.859 → **0.888** | **0.283** → 0.113 |

Metric ranking và policy accuracy không tăng đồng thời. LCB-only Compact có AUROC/AP thấp hơn nhẹ nhưng tạo quyết định tuần tự tốt hơn trên `3811`.

### 25.7. Kết luận

1. **LCB-only Compact Boosting là lựa chọn point-estimate tốt nhất hiện tại:** accuracy 22.86%, bằng Always-CP2 và Oracle, nhưng chỉ dùng 6 Teacher calls.
2. Expanded Optimized Boosting cũng đạt 22.86% nhưng verify cả 35 tasks; vì vậy không có lợi ích cost-sensitive so với Compact.
3. Semantic Logistic train Combined vẫn tốt hơn bản LCB-only về accuracy, dù tốn nhiều call hơn.
4. Chênh lệch Compact chỉ là đúng một task và paired CI vẫn chứa 0. Frozen test chỉ có hai helpful interventions, nên chưa thể kết luận LCB-only chắc chắn vượt Combined.
5. Không tune tiếp trên 35 frozen tasks sau khi đã xem kết quả. Xác nhận tiếp theo cần một untouched LiveCodeBench split hoặc data mới.

Artifacts:

- [`training_mode_comparison.csv`](../results/livecodebench_only_phase2_optimized/training_mode_comparison.csv)
- [`livecodebench_training_mode_comparison.png`](../results/livecodebench_only_phase2_optimized/livecodebench_training_mode_comparison.png)
- [`policy_comparison_by_benchmark.csv`](../results/livecodebench_only_phase2_optimized/policy_comparison_by_benchmark.csv)
- [`predictor_metrics_by_benchmark.csv`](../results/livecodebench_only_phase2_optimized/predictor_metrics_by_benchmark.csv)
- [`leakage_audit.json`](../results/livecodebench_only_phase2_optimized/leakage_audit.json)
- [`scripts/train_livecodebench_only_phase2.sh`](../scripts/train_livecodebench_only_phase2.sh)

Reproduction:

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
bash scripts/train_livecodebench_only_phase2.sh
```

## 26. CBV-Bench V2 — Level 2 multi-verification

### 26.1. Mục tiêu và protocol

CBV-Bench V2 mở rộng counterfactual collection từ single intervention sang toàn bộ tám
chuỗi Verify/Skip tại ba checkpoint:

```text
SSS  VSS  SVS  SSV  VVS  VSV  SVV  VVV
```

Trong đó **S** là Skip Teacher và **V** là Verify bằng Teacher. Bốn chuỗi đầu là Level 1;
bốn chuỗi cuối bổ sung multi-verification. Các nhánh dùng shared-prefix cache để tránh
chạy lại phần workflow giống nhau.

Phân tích ngày 27/10/2026 chỉ đọc dữ liệu đã thu thập, không gọi Student/Teacher và
không train model. Để so sánh tám chuỗi công bằng, metric chỉ dùng task có đủ toàn bộ
tám trajectory.

### 26.2. Trạng thái collection

| Dataset | Complete Level 2 | Dự kiến | Trạng thái |
|---|---:|---:|---|
| HumanEval | **163/163** | 163 | Hoàn chỉnh, validation và export thành công |
| LiveCodeBench | **171/175** | 175 | Partial; bốn task lỗi collection |
| BigCodeBench Level 1 | **0/1000** | 1000 | Chưa bắt đầu do pipeline dừng tại LiveCodeBench |
| BigCodeBench Level 2 | **0/250** | 250 | Chưa bắt đầu do pipeline dừng tại LiveCodeBench |

LiveCodeBench đã lưu 1,380 trajectories: 171 task có đủ tám nhánh, ba task chỉ có đủ
Level 1 và một task chưa có trajectory hợp lệ. Vì vậy mọi bảng Level 2 bên dưới dùng
matched subset 171 task; không trộn các sequence có cỡ mẫu khác nhau.

![CBV-Bench Level 2 dashboard](../results/cbv_bench_analysis_26102026/figures/cbv_level2_dashboard.png)

### 26.3. Accuracy theo fixed sequence

| Dataset | Sequence | Accuracy | Delta so với SSS | Helpful | Harmful | Mean Teacher tokens/task |
|---|---|---:|---:|---:|---:|---:|
| HumanEval | SSS | 80.98% | +0.00 pp | 0 | 0 | 0 |
| HumanEval | VSS | 82.82% | +1.84 pp | 6 | 3 | 1,540 |
| HumanEval | SVS | 84.05% | +3.07 pp | 7 | 2 | 1,567 |
| HumanEval | SSV | 85.28% | +4.29 pp | 8 | 1 | 1,809 |
| HumanEval | VVS | 84.66% | +3.68 pp | 10 | 4 | 2,853 |
| HumanEval | VSV | 85.89% | +4.91 pp | 11 | 3 | 3,142 |
| HumanEval | SVV | 86.50% | +5.52 pp | 12 | 3 | 3,173 |
| HumanEval | **VVV** | **87.73%** | **+6.75 pp** | 13 | 2 | 4,461 |
| LiveCodeBench | SSS | 19.30% | +0.00 pp | 0 | 0 | 0 |
| LiveCodeBench | VSS | 19.30% | +0.00 pp | 3 | 3 | 1,951 |
| LiveCodeBench | SVS | 23.39% | +4.09 pp | 8 | 1 | 2,158 |
| LiveCodeBench | SSV | 22.81% | +3.51 pp | 6 | 0 | 2,525 |
| LiveCodeBench | VVS | 24.56% | +5.26 pp | 11 | 2 | 3,985 |
| LiveCodeBench | VSV | 22.22% | +2.92 pp | 7 | 2 | 4,467 |
| LiveCodeBench | **SVV** | **26.32%** | **+7.02 pp** | 12 | 0 | 4,647 |
| LiveCodeBench | VVV | 25.15% | +5.85 pp | 12 | 2 | 6,501 |

HumanEval đạt fixed-sequence accuracy cao nhất khi verify cả ba checkpoint. Trên
LiveCodeBench, **SVV** tốt hơn **VVV** dù dùng ít token hơn: việc thêm CP1 verification
không tạo gain trung bình và có thể làm kết quả xấu đi.

![CBV-Bench accuracy heatmap](../results/cbv_bench_analysis_26102026/figures/accuracy_heatmap.png)

### 26.4. Conditional checkpoint effects

Mỗi conditional contrast chỉ thay đổi một checkpoint và giữ nguyên hai action còn lại.
Mỗi task đóng góp bốn paired histories cho từng checkpoint.

| Dataset | Checkpoint | Paired edges | Helpful | Harmful | No change | Net effect |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval | CP1 Planner | 652 | 18 | 11 | 623 | +1.07 pp |
| HumanEval | CP2 Coder | 652 | 20 | 7 | 625 | +1.99 pp |
| HumanEval | CP3 Reviewer | 652 | 26 | 5 | 621 | **+3.22 pp** |
| LiveCodeBench | CP1 Planner | 684 | 10 | 11 | 663 | **−0.15 pp** |
| LiveCodeBench | CP2 Coder | 684 | 28 | 1 | 655 | **+3.95 pp** |
| LiveCodeBench | CP3 Reviewer | 684 | 18 | 1 | 665 | +2.49 pp |

Kết quả cho thấy giá trị checkpoint phụ thuộc dataset. CP3 mạnh nhất trên HumanEval,
trong khi CP2 mạnh nhất trên LiveCodeBench. CP1 trên LiveCodeBench có net effect âm
nhẹ. Đây là bằng chứng thực nghiệm cho động cơ downstream-aware routing: số lần verify
không đủ để quyết định, vị trí và history của verification cũng quan trọng.

### 26.5. Multi-verification và oracle headroom

| Dataset | Best fixed | Accuracy | Offline any-branch oracle | Multi tốt hơn / bằng / kém best Level 1 |
|---|---|---:|---:|---:|
| HumanEval | VVV | 87.73% | 90.18% | 1 / 161 / 1 |
| LiveCodeBench | SVV | 26.32% | 27.49% | 4 / 166 / 1 |

Multi-verification tạo thêm một số rescue cases nhưng phần lớn task có kết quả bằng
best single-intervention branch. Oracle chỉ là upper bound hậu nghiệm, không phải policy
có thể triển khai và không chứng minh rằng controller có thể dự đoán đúng nhánh tốt nhất.

### 26.6. Collection cost

| Dataset | Unique Teacher calls đã collect | Unique Teacher tokens |
|---|---:|---:|
| HumanEval complete | 1,141 | 1,800,443 |
| LiveCodeBench partial | 1,211 | 2,806,349 |

Các cột token theo sequence ở mục 26.3 là chi phí triển khai trung bình của trajectory.
Bảng trên là chi phí collection theo unique shared-prefix edges; hai đại lượng không được
cộng hoặc so sánh trực tiếp như cùng một metric.

### 26.7. LiveCodeBench failure audit

Bốn lỗi đều là model response không parse được thành JSON và đều có **retryable=true**:

- LiveCodeBench/3754: unterminated string.
- LiveCodeBench/abc395_e: thiếu delimiter.
- LiveCodeBench/arc191_d: unterminated string.
- LiveCodeBench/3777: unterminated string.

Cache tương thích vẫn được giữ. Tuy nhiên, missingness có thể liên quan đến generation
khó, nên kết quả 171-task chỉ là available-data analysis và chưa được trình bày như
LiveCodeBench benchmark hoàn chỉnh.

### 26.8. Kết luận tạm thời

1. Level 2 xác nhận verification có lợi trên cả HumanEval và LiveCodeBench.
2. VVV tốt nhất trên HumanEval nhưng không cost-efficient nhất trên LiveCodeBench.
3. CP2/CP3 tạo phần lớn gain trên LiveCodeBench; CP1 có conditional net effect gần 0.
4. Harmful edges vẫn tồn tại, vì vậy Always Verify không phải chiến lược tối ưu chung.
5. Multi-verification chỉ vượt best Level 1 trên 1 HumanEval task và 4 LiveCodeBench tasks.
6. Cần phục hồi bốn LiveCodeBench task trước khi chốt số cuối và trước khi bắt đầu
   BigCodeBench trong overnight pipeline.

Artifacts:

- [CBV Level 2 available-data report](../results/cbv_bench_analysis_26102026/CBV_Level2_available_data_report.md)
- [Sequence metrics](../results/cbv_bench_analysis_26102026/sequence_metrics.csv)
- [Checkpoint effects](../results/cbv_bench_analysis_26102026/checkpoint_effects.csv)
- [Task outcomes](../results/cbv_bench_analysis_26102026/task_outcomes.csv)
- [Case studies](../results/cbv_bench_analysis_26102026/case_studies.csv)
- [Machine-readable summary](../results/cbv_bench_analysis_26102026/summary.json)
- [Analysis script](../scripts/analyze_cbv_level2_available.py)

Reproduction (offline analysis only):

```bash
cd /home/llm/hainn/hainn/agent_verification_budget
conda activate agent_verify
python scripts/analyze_cbv_level2_available.py \
  --humaneval-dir results/cbv_bench_v2_26102026 \
  --livecodebench-dir results/cbv_bench_v2_livecodebench_26102026 \
  --output-dir results/cbv_bench_analysis_26102026
```

## 27. Cập nhật cuối — CBV-Bench sau BigCodeBench recovery

Mục này thay thế trạng thái tạm thời ở phần 26 đối với BigCodeBench. Collection
BigCodeBench đã hoàn thành sau khi audit và thay thế các task lỗi hạ tầng bằng selection
recovery có provenance. Phân tích dưới đây không chạy thêm inference và dùng paired
task-rollouts khi so sánh các verification sequence.

![CBV-Bench all-dataset dashboard](../results/cbv_bench_all_26102026/figures/cbv_bench_dashboard.png)

| Collection | Validation | Complete tasks | Trajectories | Teacher calls | Teacher tokens |
|---|---:|---:|---:|---:|---:|
| HumanEval L2 | Valid | 163/163 | 1,304/1,304 | 1,141 | 1,800,443 |
| LiveCodeBench L2 | **Partial** | 171/175 | 1,380/1,400 | 1,211 | 2,806,349 |
| BigCodeBench L1 | Valid | 200/200 | 800/800 | 600 | 978,419 |
| BigCodeBench L2 | Valid | 25/25 | 200/200 | 175 | 322,034 |

Tổng cộng có **538 unique selected task IDs**, **3,684/3,704 trajectories** và
**5,907,245 Teacher tokens**. LiveCodeBench vẫn thiếu 20 trajectories trên bốn task;
do đó toàn bộ collection chưa thể gọi là 100% full-valid, dù ba collection còn lại đã
hoàn chỉnh.

| Collection | Paired n | SSS | Best single verify | Best observed sequence |
|---|---:|---:|---:|---:|
| HumanEval L2 | 163 | 80.98% | SSV: 85.28% (+4.29 pp) | VVV: 87.73% (+6.75 pp) |
| LiveCodeBench L2 | 171 | 19.30% | SVS: 23.39% (+4.09 pp) | SVV: 26.32% (+7.02 pp) |
| BigCodeBench L1 | 200 | 31.50% | SVS: 35.50% (+4.00 pp) | SVS: 35.50% (+4.00 pp) |
| BigCodeBench L2 | 25 | 32.00% | VSS: 44.00% (+12.00 pp) | VVS: 48.00% (+16.00 pp) |

Kết quả chính:

1. Checkpoint tốt nhất thay đổi theo dataset: CP3 trên HumanEval, CP2 trên
   LiveCodeBench/BigCodeBench L1 và CP1 trên BigCodeBench L2 pilot.
2. Trên 359 task Level 2 hoàn chỉnh, multi-verification chỉ tốt hơn best single ở
   5 task, kém hơn ở 2 task và bằng nhau ở 352 task.
3. BigCodeBench L2 chỉ có 25 task nên mức gain +16 pp là pilot estimate, chưa phải
   kết luận ổn định.
4. Khi training combined phải lọc complete counterfactual groups và group split theo
   `task_id`; BigCodeBench L2 là subset của BigCodeBench L1 nên không được coi là độc lập.

Artifacts tổng hợp:

- [Full cross-dataset report](../results/cbv_bench_all_26102026/CBV_Bench_all_report.md)
- [Collection summary](../results/cbv_bench_all_26102026/collection_summary.csv)
- [Paired sequence accuracy](../results/cbv_bench_all_26102026/sequence_accuracy_paired.csv)
- [Checkpoint effects](../results/cbv_bench_all_26102026/checkpoint_effects.csv)
- [Machine-readable summary](../results/cbv_bench_all_26102026/summary.json)
- [Reproduction script](../scripts/report_cbv_bench_all.py)
