# MobileVLA-R1 RGB RL on BW + vLLM 0.18.1 DAS（session007 开发交付草稿）

## 1. 简介

本报告记录固定 `AIGeeksGroup/MobileVLA-R1` RGB RL checkpoint 在超算互联网单张 BW 64GB 上的原生 vLLM 适配。模型由 Llama3 8B、SigLIP 视觉塔和发布方 2×2 downsample projector 组成。新增专用模型类、processor、tokenizer 哨兵映射和组件权重加载路径；Llama 与 SigLIP 的执行仍使用原生 vLLM / DAS 后端。

**冻结 002 已产生真实 BW 模型加载及 HTTP pilot 原件，controller 和 stop 均退出 0；独立本地审计已通过源码归属、原件一致性、七项功能、记录中的清理与覆盖卷回执检查。随后首次同输入 HTTP 实验发现实际268 / 参考269输入ID不一致，控制器按设计停止并清理。独立003候选已按发布方真实BOS / EOS调用链修复，正在等待完整重跑；正式精度、正式性能、官方验收和最终提交仍未完成。**

独立审计为 `evidence/session007/mobile-pilot001-independent-audit-b.json`，由 `scripts/audit_mobilevla_pilot007.py` 生成。它审计下载的记录，不是在本地重新加载远端权重或重新探测当时设备；该边界保留在报告和审计 JSON 中。

| 来源 / 对象 | 固定身份 |
| --- | --- |
| 权重仓库 | `AIGeeksGroup/MobileVLA-R1` |
| 权重 revision | `0c757f3e9d1b95d8416b42e7089e1c04f3e6118a` |
| 发布方代码 commit | `9d98cce3b623dfddf479740d78869ec30364b75a` |
| 实际组件路径 | `weight/rl/`；仅 RGB，不包含 depth / point tower |
| 赛事别名 | `BAAI/MobileVLA-R1`；官方别名等价关系尚未取得证明 |
| GPU 实测候选 | `session006-mobilevla-tools-002.zip`；SHA-256 `e116b10853b11fc81e5dcdc0dbd3c2d85fe36c4b8b38500a120a88c4552b92d2` |
| 实际 pilot run ID | `9c715239832bb23ab2270ab248ef0d8b` |
| 原始归档 | `000-session007-mobile-pilot001.zip`；26,522 B；SHA-256 `f417ddaf51c2a312db922fd755fbefe53cbad3975002d4ba97149a4fc8898414` |

标准目录为 `tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft/`，提供 `env.sh`、`env_check.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh`、`stop.sh`。这些包装入口委托给冻结原任务；**实际 GPU 运行对象是冻结 002 原任务，不能把新包装入口写成已在 GPU 重放**。

源包保留独立的候选源码、固定来源与许可、原任务、标准包装入口、开发对照控制器、CPU 测试和原始 pilot ZIP。完整 OCI 镜像引用、最终私有仓库commit、评审访问方式及最终表单仍待闭合；参赛者手动提交最终表单。

私有仓库根地址已由主流程创建为 [zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm](https://github.com/zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm)。当前报告不预先声明某个commit已push或评审已取得访问；最终代码保存和clone全SHA核验由主流程完成。

## 2. 验证环境

下表来自本次原始包的 `mobile-pilot001/env.log`、`config.json` 和 `server.log`。环境采集时间为 `2026-10-24T06:52:24.486517Z`。`image_effective=null`，因此不能把候选镜像名称当成完整镜像证据。

| 项目 | 实际记录 |
| --- | --- |
| Python | `3.10.12` |
| Torch | `2.10.0+das.opt1.dtk2604.20260325.g6b060a` |
| vLLM | `0.18.1+das.fa71803.dtk2604` |
| vllm-hcu | `0.18.1+das.dtk2604` |
| Transformers / tokenizers | `5.5.0` / `0.22.2` |
| Triton | `3.6.0+gitc73250c4.staging` |
| HIP / device | `6.3.26093` / `BW`；`gpu_available=true` |
| 并行与精度 | 单卡 TP1、BF16、`quantization=None` |
| 上下文与调度 | 开发 profile 4096；`max_num_seqs=1`；eager |
| GPU 内存比例 | 0.80；不是实测峰值显存 |
| 原生服务 | `127.0.0.1:30139`；本轮无外部映射 |
| 原权重目录 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-rl006/weight/rl` |
| 派生视图目录 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-derived006` |
| DTK 完整版本文件 | `/opt/dtk/.info/version` 不存在，原始失败保留；包版本带 `dtk2604` 不能替代该文件或镜像 digest |
| 平台规格、完整镜像与截图 | 待主流程补入经核验的独立平台原件 |

原生日志确认 `hcu` 平台插件激活，并解析出 `MobileVLAForConditionalGeneration006`。环境没有被本适配入口覆盖安装；最终报告仍需平台证据和原生包前后元数据，不能由单份 `env.log` 推断整个环境从未变化。

## 3. 适配优化方案

本实现使用标准 `LlavaConfig` 容器表达组件尺寸，并注册实际专用类；模型计算、projector、占位 token、模板和权重布局由专用实现处理。

| 文件 / 组件 | 具体改动与原因 | 验证及边界 |
| --- | --- | --- |
| `mobilevla_bw006/backend.py` | 使用原生 Llama/SigLIP，注册专用 processor；每帧转 RGB 后按发布方顺序 resize 至 384×384 | 原生 API / processor 预检查通过；本轮实际服务成功，不等同正式精度 |
| SigLIP 路径 | 从 27 层中取 `hidden_states[-2]`，等价保留前 26 层的全部 729 patch，避免误取末层 norm / pool 输出 | 固定发布方定义、原生源码和 CPU 数值证据；未删减真实 checkpoint |
| `mobilevla_bw006/projector.py` | 奇数 27×27 网格补零后 2×2 downsample；LN(4608)、Linear(4608→4096)、GELU、Linear(4096→4096) | 真实 safetensors header 和发布方 AST 对照；196 token / 帧 |
| `contract.py` / tokenizer | 将源 `-200` 哨兵映射为运行时输入占位 `128259`；原有 token ID 不变，LLM embedding 不新增行 | 原 tokenizer 的 1/8 帧序列和模板 CPU 对照；实际 HTTP expanded IDs 待新对照补证 |
| `prepare.py` | 四个 LLM shard 以硬链接组成派生视图；vision/projector 通过原生 secondary sources 加载 | 原 16 文件 SHA、6 份 header、8 份派生小文件封存；不改模型权重 |
| `backend.load_weights` | 返回 `AutoWeightsLoader` 的 loaded set，非量化原生 `DefaultModelLoader` 对未初始化参数抛错 | 先前 native 预检查封存过 loader 字节；本次 raw 未再次记录该文件 SHA。4+1+1 shard、warmup、HTTP 与绑定源码共同支持严格路径推断，未输出独立逐参数 runtime 计数 |
| `runtime.py` | 继承 owned supervisor，绑定 run ID、PID start token、命令、配置和最大运行时间 | 本轮 stop 回执与独立 `/proc` / port 记录已通过下载原件审计；未在本地重复远端 probe |
| comparison006-002历史 / 003修订 | 002统一双EOS的开发比较发现native268 / reference269；003按真实publisher调用链保留双BOS，并使用显式单EOS128009、greedy、max32 | 旧失败原件保留；003的27项原回归、6项来源 / token路径、7项生命周期测试通过，实际重跑待证 |

003新增 `_apply_hf_processor_tokens_only`，修复 OpenAI renderer 传入 token list 时默认原样返回、绕过专用文本 tokenization 的缺陷。它无cleanup解码对应文本，再执行同一个发布方 `tokenizer_image_token` offset流程；不是删除回包或参考token。固定source链与真实tokenizer-only执行详见 `docs/mobilevla-bos003-source-alignment.md` 和 `evidence/session007/mobilevla-bos003-source-diagnosis.json`。

003使用独立 `/root/private_data/guoxin-mobilevla-tools-007-003` 和新派生 `/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-bos003`；新receipt绑定7个适配模块SHA，拒绝旧receipt和源码漂移，原derived006保持原件。`--generation-config vllm` 配合明确的 EOS128009，遵循publisher `inference.py sample_prepared` 的显式覆盖语义；checkpoint中的双EOS文件未改。

RL 16 项权重提取已记录 CRC、长度和 SHA。新增 covering proof 回执标记 14 个覆盖卷完整 published LFS SHA 通过，并将 16 项 RL 压缩缓存绑定到这些卷；补齐网络 gap 为 552,355,933 B。该证明保留原提取 receipt 不变，`all_26_full_volume_bytes_verified=false`；不能据此声称整个 26 卷仓库和 SFT 权重都已核验。

保持平台与 general plugin 发现，不能把 `VLLM_PLUGINS` 单独设成 MobileVLA 插件名；原生 HCU / hcu_ops 必须继续可见。TP1、BF16、最多 8 帧、4096 是当前开发范围。没有扩展原 Llama 的 8192 最大位置，也没有为满足性能表格擅自改权重、结构或 tokenizer 语义。

## 4. 服务启动脚本

本次 GPU 使用 `/root/private_data/guoxin-mobilevla-tools-006-002/tasks/mobilevla-r1-rl-bw-vllm/pilot.sh`。原件记录 supervisor 210 / server 211，start token 分别为 `285201725` / `285201735`；这些只属于本次历史运行，不能在后续会话复用 PID 停止服务。

标准草稿入口的 CPU 计划检查与实际运行方法如下：

```bash
cd tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft
bash run.sh --dry-run --run-id 654118905a9e4fd8a5d5abe88a0ca97a
bash run.sh --run-id <new-32-hex-run-id>
bash env_check.sh > env.log
bash smoke.sh > smoke-stdout.log
bash stop.sh --run-id <same-owned-run-id>
```

`<new-32-hex-run-id>` 为每次运行新生成的值。建议同输入实验直接使用独立 `mobilevla_comparison_orchestrator007.py`：它生成新 ID、核验冻结 70 项与 comparison 6 项源文件、启动并等待 readiness、采原生 1/8 帧、在 `finally` 精确停止，独立检查 terminal state、`/proc` start token、live 进程组和端口，再执行 HF / token 对照，最后再次清理。总预算最多 1800 秒，每步有 timeout，输出必须为新目录。

原生标准日志仍产生于原任务 `runtime/`；标准包装入口没有移动或覆盖旧日志。每次运行后应复制到独立 evidence 目录并封存同 run ID 配置与日志。原生 server / supervisor 日志采用 append 模式，审计必须按本 run marker 截取，不能把其他运行的警告或失败混入结论。

## 5. Smoke 测试

下表读取自本轮原始 `smoke.log`；smoke 自身 ID 为 `317c43d1ef1841cabf8cc778010e9beb`，与服务 owned run ID 的作用不同。五次 chat 的 HTTP200 还可与同 run server access log 对照。所有输出仅证明开发功能和非空文本，不证明描述事实或导航动作正确。

| 检查 | 原始记录 | 客户端 wall，仅本次开发样例 |
| --- | --- | ---: |
| `GET /health` | HTTP200 | 未作为性能指标记录 |
| `GET /v1/models` | HTTP200，返回对应 served name 与模型目录 | 未作为性能指标记录 |
| 单图 | HTTP200，非空 assistant 文本，stop | 0.6693 s |
| 双图 | HTTP200，非空 assistant 文本，stop | 1.7947 s |
| 多轮第一轮 | HTTP200，非空 assistant 文本，stop | 0.4316 s |
| 多轮追问 | HTTP200，非空 assistant 文本，stop | 0.7546 s |
| 图像流式 | HTTP200，非空 content，最终 stop 和 `[DONE]` | 0.4354 s |

非流式 usage 的 `prompt_tokens` 分别为 259、460、259、298，`completion_tokens` 为 12、52、12、22。原 smoke 没有请求 `return_token_ids`，因此 `prompt_token_ids=null`、`token_ids=null` 是预期缺项，不能把 usage 计数当成 token ID 对照证据。多图响应中也有需要正式评测判定的描述内容，不能从“非空”推出准确。

state 记录服务于 `06:50:33.641485Z` 创建、`06:52:24.098463Z` ready、`06:52:57.436473Z` 结束；服务生命周期 143.795 s。stop 回执记录精确停止对应 supervisor 和 server group；另有 `06:54:38.971914Z` 的独立 `/proc` 不存在与 30139 关闭记录。独立本地审计已核对归属和原件哈希；本草稿不把该记录推广为新 run 已清理。

## 6. 精度验证

当前没有与本题明确绑定的官方 evaluator、自检、固定样本、参考动作、GPU baseline、误差容忍契约和 `BAAI` 别名映射证明。RGB-only 发布配置与赛事期待的 RGB-D / point 范围也需正式确认。本报告不生成虚构 `verify_result.json`、导航准确率、success rate 或正式 PASS。

标准 `verify.py` 委托冻结官方材料 gate。只有取得 `MOBILE_OFFICIAL_VERIFY_SCRIPT`、`MOBILE_OFFICIAL_DATA_DIR`、`MOBILE_OFFICIAL_BASELINE` 等实际材料后才能保留其脚本 / 数据 / baseline 哈希，执行自检并按真实契约解释结果；缺材料时退出2。

同输入 HF 组件对照是额外的工程参考：保留原始权重和 projector，记录相同帧字节、源 `-200` token、展开后的原生 ID、生成 raw IDs，以及 HF 像素 / 视觉 / projector 摘要。完全相等也只能证明这两个开发 fixture 下的输出一致，不构成发布方 wrapper baseline 或正式导航精度。首次实际比较的输入检查未通过；003重跑尚未完成。旧helper统一checkpoint双EOS只是历史开发策略；003明确使用publisher实际wrapper的单EOS128009。

## 7. 性能测试

本次开发服务是 4096 上下文、并发1、eager、最多8帧。原始日志中的 3.52 s 表示 loader 报告的权重读取耗时，7.77 s 表示 profile / KV cache / warmup 的初始化耗时；它们不等于完整冷启动 wall。第5节请求 wall 包括传输和客户端处理，只是单次 smoke 诊断，不能计算或冒称正式 TTFT、TPOT、tokens/s 或优化收益。

标准 `bench.sh` 当前对实际正式执行明确拒绝这个 pilot profile；`official_performance_verified=false`。原生 Llama 配置最大位置为8192，正式 T02 的 16K 输入加1K输出没有被当前模型配置容纳。该限制必须按主办方任务要求解决或明确拒绝，不能静默截断、扩展位置或以4K开发结果填充16K档。

正式性能仍需固定输入 / 图像分辨率、输出长度、1/8/64并发、4K/16K档、warmup、两轮负载和第二轮原始测量 / 汇总。原日志估算的 KV cache capacity 或理论 concurrency 不作为实测性能填表。

## 8. 注意事项

- 草稿与源包尚未最终封存；同输入实测结果回来后应新增修订与证据，不能改写冻结002或失败原件。
- 原生日志含 tokenizer incorrect regex pattern 提示。当前没有自动添加 `fix_mistral_regex=True`，因为那会影响输入 token 语义；先以相同输入真实 expanded IDs 对照判断，再决定经证据支持的修正。
- eager、ROCm GELU fallback、sparse op未出现、generation config默认采样提示和 NCCL `destroy_process_group()` 警告均保留。服务 shutdown 完成与独立进程 / 端口记录能支持本次清理，但不应删除警告或单凭日志宣称未来无资源风险。
- 覆盖14卷 LFS证明与26卷全仓证明分开；原 receipt、prepare receipt、覆盖卷新proof和实际运行证据分别保留。
- 发布方代码保留 Apache-2.0 及文件内 LLaVA / NVIDIA copyright notice；模型卡 `cc-by-sa-4.0` 只按该发布记录陈述，不能覆盖底座权利。LLM配置能证明 LlamaForCausalLM 架构，视觉配置能证明 SiglipVisionModel 架构；当前锁中尚缺对应精确底座仓库 / revision / 许可原文，不擅自推断所有权重适用同一许可。新编代码保留版权说明，不为整仓自选MIT或其它许可证。分项清单见 `docs/mobilevla-license-provenance007.md`。
- 完整OCI镜像、私有代码保存 / 评审访问、平台截图、官方 evaluator / data / baseline、正式性能、干净环境重放和最终表单仍有缺口。`submission_ready=false`、`submitted=false`；最终表单由参赛者手动提交。
