# Deep Research Agent Harness 基准评测报告

本地全量验证结果。评测通过Mock故障注入、脚本模型与真实DeepAgents图验证工具契约、跨源恢复和证据预算。

- 样本：24；行为断言通过：24。
- 对照提交：`18e0552a17f4caaf290767a7de6f4afba00c7c88`；当前HEAD：`18e0552a17f4caaf290767a7de6f4afba00c7c88`。
- 代码SHA-256：`aa666c2e2950a2801d892ca7dd2ec9bafe75e5387c85469ac88e17e973da9e9b`。
- 数据集SHA-256：`425214c98d3ea5d671c33efe8d3abaa7b927572b1d9fe4c8e105bc522683bbda`。
- 计数器：`utf8-byte-budget-v1 (proxy, conservative)`。

## 实验 A：容错与跨源恢复

| 执行策略 | 完成数 | 完成率 |
|---|---:|---:|
| 基础异常捕获，无跨源恢复策略 | 4/7 | 57.1% |
| ToolWrapper 错误分类 + DelegationHarnessMiddleware 跨源恢复 | 6/7 | 85.7% |

评测要求至少两个不同可用来源类别提供证据。基础组执行已有工具异常捕获逻辑；增强组运行真实专家工具、ProviderHarnessMiddleware与DelegationHarnessMiddleware，逐例保留实际委派轨迹。

| 样本 | 注入故障 | 基础组 | 增强组 | 实际专家委派次数 |
|---|---|---|---|---:|
| F01 | {"tavily": "TimeoutError"} | 中止 | 完成 | 4 |
| F02 | {"tavily": "ServiceUnavailable"} | 中止 | 完成 | 4 |
| F03 | {"mysql": "TimeoutError"} | 完成 | 完成 | 3 |
| F04 | {"mysql": "ServiceUnavailable"} | 完成 | 完成 | 3 |
| F05 | {"ragflow": "TimeoutError"} | 完成 | 完成 | 4 |
| F06 | {"ragflow": "ServiceUnavailable"} | 完成 | 完成 | 4 |
| F07 | {"tavily": "TimeoutError", "ragflow": "ServiceUnavailable"} | 中止 | 安全中止 | 5 |

F07同时失去Tavily与RAGFlow，仅剩MySQL，按双来源证据要求安全中止；错误类型、来源与委派轨迹记录在results.json。
辅助诊断：显式无捕获对照完成0/7，完整数据保存在experiment_a[].unprotected。

## 实验 B：两级预算与证据去重

### 计量口径

主指标为**单次调研任务累积注入主控的证据与消息上下文体积**。每条首次进入主控输入的user/assistant/tool消息计一次；不同工具返回中重复出现的证据仍计入实际注入体积。固定系统提示词和工具定义不纳入主指标，同一历史消息在后续模型调用中重放时不重复计量。

计算式：`effective_context_units = sum(injected_messages[].units)`；`units = counter.count(canonical_json(role, content, tool_calls))`。消息ID只用于识别历史重放，不参与计数；保留每条消息的计数和SHA-256以便审计。

样本为分层合成压力样本：紧凑3例、标准长页6例、扩展长页3例，包含追踪URL镜像、相似正文和重复核验检索。工作流固定为首轮三源取证、第二轮网页与文档核验，共5个子任务、3次主控模型调用。样本分层、检索计划和阈值存于数据集，结果由同一计量函数直接计算。

关闭组同时关闭工具观察与子任务返回裁剪；启用组使用真实ProviderHarnessMiddleware的6000输入保护、DelegationHarnessMiddleware的1500子任务预算和URL/内容指纹/SimHash去重。工具Schema、取证事实、问题与轨迹保持一致。

| 指标 | 关闭裁剪 | 两级预算 + 去重 |
|---|---:|---:|
| 平均每任务有效上下文体积（计数单位） | 14,083.4 | 8,031.4 |
| 上下文超阈次数 / 12 | 9 | 0 |
| 平均体积缩减 | — | 42.97% |

阈值规则：`effective_context_units > 12000`。该值是本压力集的动态上下文预算，超阈计数用于衡量预算压力；真实生成延迟和网络超时采用独立在线测量。

| 样本 | 分层 | 关闭裁剪 | 两级预算 + 去重 | 超阈（前→后） | 关键事实保留 |
|---|---|---:|---:|---|---|
| N01 | compact | 8967 | 7425 | False→False | True |
| N02 | compact | 9016 | 7450 | False→False | True |
| N03 | compact | 9010 | 7444 | False→False | True |
| N04 | standard | 15008 | 8208 | True→False | True |
| N05 | standard | 14816 | 8316 | True→False | True |
| N06 | standard | 14819 | 8305 | True→False | True |
| N07 | standard | 14690 | 8002 | True→False | True |
| N08 | standard | 14857 | 8073 | True→False | True |
| N09 | standard | 14995 | 8331 | True→False | True |
| N10 | long | 17605 | 8205 | True→False | True |
| N11 | long | 17529 | 8351 | True→False | True |
| N12 | long | 17689 | 8267 | True→False | True |

### 辅助全量输入统计

包含框架系统提示词、工具定义和重复历史的每次模型调用平均代理计数为41,295.8→38,278.5。逐轮、累计和最终调用值保存在experiment_b[].before/after，便于同时评估固定开销与动态证据。

计数单位由tokenizer字段明确标识：有本地cl100k词表时使用该编码代理计数，缺失时使用UTF-8字节预算；实际API计费Token以目标模型usage为准。以上结果适用于公开的合成压力集及固定工作流。

## 24例行为断言

| 样本 | 分类 | 实际行为 | 断言通过 |
|---|---|---|---|
| N01 | normal_multi_source | complete | True |
| N02 | normal_multi_source | complete | True |
| N03 | normal_multi_source | complete | True |
| N04 | normal_multi_source | complete | True |
| N05 | normal_multi_source | complete | True |
| N06 | normal_multi_source | complete | True |
| N07 | normal_multi_source | complete | True |
| N08 | normal_multi_source | complete | True |
| N09 | normal_multi_source | complete | True |
| N10 | normal_multi_source | complete | True |
| N11 | normal_multi_source | complete | True |
| N12 | normal_multi_source | complete | True |
| F01 | single_source_fault_injection | complete | True |
| F02 | single_source_fault_injection | complete | True |
| F03 | single_source_fault_injection | complete | True |
| F04 | single_source_fault_injection | complete | True |
| F05 | single_source_fault_injection | complete | True |
| F06 | single_source_fault_injection | complete | True |
| F07 | single_source_fault_injection | abort | True |
| C01 | conflict_and_out_of_scope | complete | True |
| C02 | conflict_and_out_of_scope | complete | True |
| C03 | conflict_and_out_of_scope | reject | True |
| C04 | conflict_and_out_of_scope | reject | True |
| C05 | conflict_and_out_of_scope | abort | True |

## 验证边界

- 执行闭环：Mock与脚本模型固定输入轨迹，真实运行工具封装、专家图、两级预算和跨源委派。
- 事实质量：校验关键事实保留、冲突证据及未解决标记、SQL拒绝行为和私有事实来源边界。
- 指标解释：任务完成率、行为断言通过率和上下文预算超阈分别统计。
- 在线验证：模型回答质量、API计费、端到端延迟及真实超时率作为独立评测项。
- 回归入口：python -m unittest覆盖工具适配、原生Skill、取消、并发隔离和HTTP/WebSocket。
