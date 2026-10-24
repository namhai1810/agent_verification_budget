# Báo cáo tổng hợp — Downstream-Aware Verification, Phase 1

**Ngày chốt dữ liệu:** 24/10/2026  
**Dataset:** OpenAI HumanEval, 164 tasks  
**Trạng thái:** 24 tasks development hoàn tất; 140 tasks held-out đã được thử, trong đó 139 tasks có đầy đủ paired interventions và 1 task gặp generation failure.

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
| Highest-risk policy mean delta | +4.2 pp | +2.2 pp |
| Offline-oracle mean delta | +8.3 pp | +8.6 pp |
| Tasks có ranking ties | 22/24 | 134/139 |

Critic có tín hiệu dương nhẹ trên held-out nhưng score vẫn quá rời rạc và nhiều ties. `HumanEval/93` helpful và `HumanEval/137` harmful đều từng nhận risk `0.15`, nên critic hiện chưa phân biệt tốt “Teacher sẽ cứu” với “Teacher sẽ phá”.

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
3. **Routing là bottleneck tiếp theo:** offline oracle đạt +8.6 pp trong khi highest-risk heuristic chỉ đạt +2.2 pp. Cải thiện critic/acceptance gate có khả năng đem lại nhiều giá trị hơn chỉ thay Teacher model ngay lập tức.

## 11. Đề xuất thí nghiệm tiếp theo

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
