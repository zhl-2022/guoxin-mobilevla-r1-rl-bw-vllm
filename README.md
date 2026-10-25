# MobileVLA-R1 RGB RL on BW + 原生 vLLM DAS

## 1. 简介

本工程将固定 `AIGeeksGroup/MobileVLA-R1` RGB RL checkpoint 适配到超算互联网单张 BW 64GB、原生 vLLM/HCU 后端，保持 Llama3 8B、SigLIP 和发布方 2×2 downsample projector，新增专用模型类、processor、严格权重布局与输入协议处理。

**004 已完成真实 BW 全模型加载、七项 HTTP 功能复跑、1/8 帧同输入 HF 组件参考比较，以及自然停止边界比较。1/8 帧的输入分别269/1648 token，各32个生成 raw ID逐项相等；短问句输入263 token，两端13个 raw ID逐项相等，索引11的128001后继续生成，末 token128009真实触发 stop。两次运行的源码、回执、完整原件与 owned 清理均通过本地独立审计。正式导航准确率、正式性能、官方验收和最终提交仍未完成。**

| 对象 | 固定身份或实际记录 |
| --- | --- |
| 发布权重 | `AIGeeksGroup/MobileVLA-R1`，revision `0c757f3e9d1b95d8416b42e7089e1c04f3e6118a`，`weight/rl/` |
| 发布方代码 | commit `9d98cce3b623dfddf479740d78869ec30364b75a` |
| 实测修订 | `007-eos004`；部署工具 ZIP SHA-256 `f248a692df9ae69af29dae264e4ef6e659269593b25057b26eb17e18e2ad2336` |
| 同输入比较 run | `b10e5daa57e83d115258adf97d496377` |
| 比较原件 | `dist/000-session007-mobile-comparison003.zip`，64,762 B，SHA-256 `deba391bfd4da82ddc1799e4c303507ed88a41782efb4bd9f4c0e3bcd0595057` |
| 最终 HTTP pilot run | `4171b0333672dac2f12a25dbbba9de60` |
| HTTP 原件 | `dist/000-session007-mobile-pilot004.zip`，52,432 B，SHA-256 `e2ad816cc70109f4ba83646eb4fa59e105c159ce261ef60987b11899bc360530` |
| 私有仓库 | [zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm](https://github.com/zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm)；push、实际 commit 与 fresh clone 全 SHA 核验由封包后的独立交付回执记录 |
| 赛事别名 | `BAAI/MobileVLA-R1`；尚缺主办方对别名等价关系和 RGB-only 任务范围的确认 |

标准入口在 `tasks/mobilevla-r1-rl-bw-vllm-standard007/`，包含 `env.sh`、`env_check.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh`、`stop.sh` 和本报告。薄包装委托同仓库 `tasks/mobilevla-r1-rl-bw-vllm/` 的实际原生任务；原生任务完成上述 GPU 运行，标准薄包装另有 shell 语法和 dry-run 检查，不把它写成一次额外 GPU 重放。

## 2. 验证环境

| 组件或配置 | 实际记录与边界 |
| --- | --- |
| 平台与设备 | 超算互联网 BW64GB 单卡；运行日志记录设备 `BW`，HCU 平台插件激活 |
| Python | 3.10.12，入口 `/usr/bin/python` |
| Torch | distribution metadata `2.10.0+das.opt1.dtk2604.20260325.g6b060a`；HF 进程 `torch.__version__` 为 `2.10.0`，两种口径分别保留 |
| vLLM / vllm-hcu | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604` |
| Transformers / tokenizers | `5.5.0` / `0.22.2`；源与派生 tokenizer 实际类均为 `TokenizersBackend` |
| Triton / HIP | `3.6.0+gitc73250c4.staging` / `6.3.26093` |
| 精度与并行 | BF16、TP1、非量化、eager |
| 开发 profile | 上下文4096、并发1、最多8帧 RGB、GPU 内存比例0.80 |
| 服务 | `127.0.0.1:30139`；稳定 served name `mobilevla-r1-rgb-rl006`，实际修订由004源码 SHA 和新 receipt 绑定 |
| 原始权重 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-rl006/weight/rl` |
| 独立004派生视图 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-eos004` |
| 镜像与 DTK 身份 | 完整 OCI repository:tag/digest 尚缺；`/opt/dtk/.info/version` 不存在的失败保留，不能以 wheel 后缀替代完整镜像证据 |

适配入口保留原生 GPU 栈与 `hcu`、`hcu_ops` 插件发现，不执行发布方 CUDA 环境安装脚本。实际 processor 源 SHA 为 `900283f4fabd65449ac76b8a08d3d8e546321d4c88d046c3d510cfc5fabde1bf`，backend SHA 为 `5d529bd662867463d2f7f3fabb24dc5c8629976a136cbdeab5aff8b349e2c0d8`。实际 renderer、InputProcessor、SamplingParams 源 SHA 与锁定公开参考一致；实际 ModelConfig SHA 为 `82c0879b0670a5883e1c5e844e1ebf20bf21ee1f5764ff6a799ca97856d5ad38`，与下载的公开参考不同，未对该文件差异作完整源码归因。EOS 结论同时使用实际安装方法执行结果和真实生成 raw ID，不宣称整个 DAS 框架与公开版本字节一致。

## 3. 适配优化方案

| 位置 | 实际行为与原因 | 验证范围 |
| --- | --- | --- |
| 专用 backend / registry | 注册 `MobileVLAForConditionalGeneration006`，Llama 与 SigLIP 由原生 vLLM 执行；LlavaConfig 承载组件配置 | 服务真实解析到专用类，专用实现覆盖 Mobile 图像投影、输入和权重规则 |
| SigLIP | checkpoint 保留27层；`hidden_states[-2]` 选择对应前26层输出的729个 patch，无末层 pooling 替代 | 固定发布源码、6个 safetensors header、CPU 合同及真实 BW 执行 |
| Projector | 奇数网格补零后 2×2 downsample；LN4608、Linear4608→4096、GELU、Linear4096→4096 | 发布方真实 AST 与数值对照，每帧196 token，权重不改写 |
| RGB processor | 逐帧 RGB 转换、resize384×384，再执行 SigLIP 预处理并保留帧顺序 | CPU 原 processor 对照，以及真实1/8帧 input ID 与 HF 组件比较 |
| 源 token 语义 | 源 `-200` 映射为输入占位128259，LLM embedding/head仍128259行 | 不新增可生成 embedding 行；每帧196个图像占位 |
| 双 BOS 修复 | renderer 给出 token list 时，003开始在真实 tokens-only 路径执行发布方相同 tokenization，004保留 | 源模板字面 BOS 与 checkpoint 自动 BOS 均保留；修复前268/269差异原件保留，没有删除输入或生成 token |
| EOS004修复 | 发布方 `sample_prepared` 显式传入 tokenizer EOS128009；独立派生 generation config 将 `[128001,128009]` 映射为128009，root/text config 原值保持128009 | 实际 native 配置方法测得 stop set仅128009，历史源 generation config 重演 stop set为双EOS；真实请求自然跨128001并以128009结束 |
| Prepare 与严格加载 | 原16文件、6个 header、8个派生小文件 SHA；4个 LLM shard 硬链接，vision/projector 由原生 secondary sources 加载，返回 loaded set 交给非量化 DefaultLoader | 实际4+1+1 shard加载与推理通过；未独立枚举全部 GPU 参数张量 |
| 新派生 receipt | `mobilevla-derived-hf-view-007-eos004-v1`，绑定7模块 SHA、8派生文件 SHA、16源文件 SHA和 EOS 映射 | Runtime 拒绝旧 schema、模块漂移、源内容漂移及额外 EOS；004源16项与已核003提取回执身份一致 |
| Owned 生命周期 | 新32hex owner、PID start token、精确配置、bounded subprocess、finally cleanup | native 和 final 两次核验通过；不按进程名或端口号批量杀其他服务 |

RL16项提取回执记录 CRC、长度和 SHA。14个覆盖卷的 published LFS 完整 SHA 证明进一步将 RL16缓存绑定到发布字节，补齐 gap552,355,933 B；独立审计检查固定锁、回执及逐段无洞、无重叠映射。本地没有重新读取远端17GB大字节，`all_26_full_volume_bytes_verified=false`，不把覆盖卷证明写成26卷全仓或 SFT 完整性证明。

EOS修复依据见 `docs/mobilevla-eos004-stop-contract.md`，历史 BOS 来源链见 `docs/mobilevla-bos003-source-alignment.md`。本地4.57.3 tokenizer CPU 执行、实际5.5.0 native probe、完整 GPU 比较分别记录，没有声称运行发布方4.37.2完整环境或官方完整 wrapper baseline。

## 4. 服务启动脚本

### 提交前审计修订008

新增仓库根目录的标准入口，委托原标准目录和已封存的004原生任务。根目录可执行 `bash run.sh --dry-run --run-id 654118905a9e4fd8a5d5abe88a0ca97a`、`bash env_check.sh --dry-run`、`python verify.py --selfcheck --dry-run` 和 `bash bench.sh --dry-run`。正式材料缺失时verify/bench应返回2，不能将此退出码改成通过。

这些入口修订的验证范围为CPU和Linux脚本合同；没有把旧GPU日志改标成008 GPU复跑。原适配器、checkpoint与实验原件未修改。新增 `verify_bundle.py` 核对本修订的精确文件集合，旧manifest和旧README保存在 `evidence/prior-delivery008/`。完整镜像、干净环境GPU复跑、官方精度/性能和评审访问仍待补齐，`submission_ready=false`。

源码树的主要目录如下。权重保存在外部 `private_data`，不包含在代码包或 Git 仓库中。

| 目录 | 用途 |
| --- | --- |
| `candidates/mobilevla-r1-session006/mobilevla_bw006/` | 当前004专用模型、processor、prepare、runtime 与 HF 组件参考 |
| `tasks/mobilevla-r1-rl-bw-vllm/` | 实际运行并绑定源码的原生任务 |
| `tasks/mobilevla-r1-rl-bw-vllm-standard007/` | 标准脚本入口与最终报告 |
| `scripts/` | Owned 比较控制器、严格解包、下载证明、原件审计、封包与 clone 校验 |
| `tests/` | 45项 CPU 合同、源码等价、EOS 与生命周期回归 |
| `evidence/session007/` | 独立审计及按索引提取的原始副本 |
| `dist/` | 已冻结部署包、历史草稿及未修改的成功/失败原件 ZIP |
| `docs/` | 当前报告、来源说明、许可记录及保留的历史草稿 |

已有源16项权重和原 extraction receipt 时，在 `/root/private_data` 内的新源码树执行完整比较；输出目录必须不存在，且设备应由本实验独占使用：

```bash
/usr/bin/python -B scripts/mobilevla_comparison_orchestrator007_003.py \
  --frozen "$PWD" --comparison "$PWD" --python /usr/bin/python \
  --output /root/private_data/mobilevla-comparison007-eos004-replay001
```

控制器按需新建004派生视图，核验70项候选和5项 helper/probe 源锁，执行实际安装的 token 与 EOS 配置方法，再启动 owned run。原生1/8帧各 max32及短问句 max128完成后，精确停止、核验 state/start token/进程组/端口，才执行三个 HF reference，最后再次 cleanup。总预算最多1800秒，工作步骤预留65秒清理，不设 min_tokens、不忽略 EOS、不强迫 token 输出。

标准服务入口：

```bash
cd tasks/mobilevla-r1-rl-bw-vllm-standard007
bash run.sh --dry-run --run-id 654118905a9e4fd8a5d5abe88a0ca97a
bash run.sh --run-id <new-32-hex-owned-run-id>
bash env_check.sh > env.log
bash smoke.sh > smoke-stdout.log
bash stop.sh --run-id <same-owned-run-id>
```

Stop 缺少精确32hex ID时退出2。原生 runtime 日志在原任务目录中保存；按 run marker 封存该轮 state/config/server/supervisor，不将 append 日志中的其他 run 混入结论。运行依赖使用现有原生镜像和权重目录，额外 requirements 不覆盖 GPU 栈。

源码与原件完整性可在解包或 fresh clone 后核验；manifest SHA取外部封包回执的 `manifest_sha256`：

```bash
python -B scripts/verify_mobilevla_standard004_007.py \
  --root "$PWD" --manifest-sha256 <package-receipt-manifest-sha256> \
  --check-git-index
```

该检查覆盖 manifest 中所有文件 SHA、精确文件集合和 Git tracked 集合；Git commit/branch由实际 clone 输出。纯解包树检查时省略 `--check-git-index`。最终 manifest 不将自身列入 files，外部 receipt 单独锁定它的 SHA，避免自引用。

## 5. Smoke 测试

最终标准原生任务的 health、models、单图、双图、多轮首轮、多轮追问和图像 SSE 共七项功能检查全部通过。Owner `4171b0333672dac2f12a25dbbba9de60` 与 smoke client `d8ff059ab7a44c1e96e65ff64d4a81a6` 分别记录，controller/stop均退出0。

| 最终004用例 | 实际结果 | 原始结束信息 |
| --- | --- | --- |
| Health / models | HTTP200，模型身份与004派生路径匹配 | GET检查通过 |
| 单图 | HTTP200，非空文本，13生成 token | stop，stop_reason=null，无 length 截断 |
| 双图 | HTTP200，非空文本，50生成 token | stop，stop_reason=null，无 length 截断 |
| 多轮首轮 / 追问 | HTTP200，均非空文本，各13生成 token | stop，stop_reason=null，无 length 截断 |
| 图像 SSE | 14个事件，非空delta、终止帧和 `[DONE]` | stop，stop_reason=null，无流错误或 length 截断 |

完整53项 raw envelope 的 manifest/CRC及选定 Mobile run 的9个源码 SHA通过；其他 session 记录按原 ZIP保留，不参与本模型结论。实际4+1+1 shard加载，日志记录加载5.40 s。独立清理 UTC `2026-10-24T08:25:04.877407+00:00`：supervisor19454/server19455均 absent，start token285740963/285740973与owner一致，30139关闭。详见 `evidence/session007/mobile-pilot004-independent-audit.json`。

同输入运行 `b10e5daa57e83d115258adf97d496377` 的52项原件完整 SHA/CRC、75项前后源码锁、新派生 receipt、API原始 prompt/choice IDs、usage及比较输入文件 SHA均通过独立重算：

| 输入 | 展开输入 token | 原生与HF生成 raw ID | 结束原因 | 停止边界 |
| --- | ---: | --- | --- | --- |
| 1帧 red，原问题 | 269，双BOS、196图像占位 | 两端各32，逐项全等，首分歧null | 两端 length | 此组未触发 EOS |
| 8帧 red/blue交替，原问题 | 1648，双BOS、1568图像占位 | 两端各32，逐项全等，首分歧null | 两端 length | 此组未触发 EOS |
| 1帧 red，中文短问，max128 | 263，双BOS、196图像占位 | 两端各13，逐项全等，首分歧null | 两端 stop，native stop_reason=null | 索引11的128001后继续，索引12的128009终止 |

第三组双方完整 raw IDs 为：

```text
[32,2579,9518,374,279,1925,1665,304,279,2217,13,128001,128009]
```

实际配置 probe 执行安装的 ModelConfig、BaseRenderer、SamplingParams 方法，测得 primary EOS128009、有效停止集合 `[128009]`；HF AutoConfig及vLLM实际 text-config对象也均为128009。以未修改源 generation config重演得到 `[128001,128009]`。该配置证据与自然生成共同确认004修复。HTTP七项原件未请求尾 token IDs，因此其中的 null stop_reason单独不用于识别 EOS；真正的128009触发证据来自上述独立同输入 raw IDs。

HF张量形状为 pixel `[N,3,384,384]`、vision `[N,729,1152]`、projector `[N,196,4096]`，展开输入为 `[1,T,4096]`，张量摘要随原件保留。Native stop和HF完成后final stop均核验owner、PID/start token、空live进程组及30139连接拒绝：supervisor13136/server13137，start token285650841/285650851，final UTC `2026-10-24T08:12:16.503847+00:00`。完整15步骤均PASS、实际613.055 s小于1800 s，HF启动晚于独立native清理。详见 `evidence/session007/mobile-comparison003-independent-audit-b.json`。

历史002功能结果、第一次268/269输入差异、003短前缀成功及003b错误128001停止均保留，不用后续成功覆盖旧记录。旧003b的四例额外停止由 `mobile-pilot003-independent-audit.json` 明确标为停止语义不一致。

## 6. 精度验证

目前缺少本题绑定的官方 evaluator、自检、固定导航样本/动作参考、GPU baseline、误差容忍合同以及赛事别名/RGB-only范围确认。`official_accuracy_verified=false`、`official_acceptance=false`，没有导航准确率、success rate或正式PASS。

标准 `verify.py` 委托官方材料 gate；缺少 `MOBILE_OFFICIAL_VERIFY_SCRIPT`、`MOBILE_OFFICIAL_DATA_DIR`、`MOBILE_OFFICIAL_BASELINE` 等材料时退出2。有材料后需封存脚本、数据、baseline SHA、自检和原始输出，再按真实评分合同解释。

本次三组比较使用独立标准组件 HF工程参考，没有执行官方完整 wrapper或 benchmark。32/32/13个 raw ID相等证明这些开发 fixtures、同一greedy策略下的输入和生成一致；第三组补足了真实停止边界。它不证明导航动作正确、所有场景或正式任务精度，也不证明全部长上下文和采样策略均一致。

## 7. 性能测试

当前4096上下文、并发1、eager开发 profile没有正式 TTFT、TPOT、tokens/s或优化收益。原 checkpoint最大位置8192，不能容纳正式T02的16K输入加1K输出；没有静默扩展位置或截断以填性能表。

004控制器613.055 s包括111.663 s源验证/prepare、33.465 s配置及token probe、161.632 s启动、客户端模块导入、三个请求、三个HF组件进程及两次清理。HF进程内部记录57.769/15.649/21.485 s，覆盖其模型加载与生成；native客户端stage约24.756/24.352/23.301 s包含模块导入，HTTP smoke请求另有约0.46–1.69 s的日志区间。这些区间不同，不作为native/HF速度对比或正式性能。5.40 s权重加载同样只是特定日志字段。

标准 `bench.sh` 对当前pilot profile拒绝实际正式执行，`official_performance_verified=false`。正式性能仍需主办方固定输入/输出、并发1/8/64、4K/16K档、warmup、两轮测试和第二轮原始日志/汇总，KV cache理论容量不能替代吞吐实测。

## 8. 注意事项

- 本包封存开发源码和真实运行证据；完整OCI镜像身份、平台截图、私有仓库最终commit/fresh-clone回执、评审访问、正式评测和最终表单仍分别记录。`submission_ready=false`、`submitted=false`，最终表单由参赛者手动提交。
- 保留 tokenizer incorrect-regex、ROCm GELU(tanh)→exact fallback、eager、无效sparse-fusion及NCCL未显式destroy_process_group警告。没有自动启用 `fix_mistral_regex=True` 改写tokenizer；实际输入及raw输出一致是本次实测范围内的证据。
- 部署工具manifest及prepare receipt中的GPU字段记录其生成时的CPU阶段状态；后续真实GPU结果由原始运行包和独立审计记录，不回写已冻结原件。
- 发布方代码Apache-2.0及LLaVA/NVIDIA notices、Mobile权重模型卡 `cc-by-sa-4.0` 分别保留。LLM/vision精确底座repo、revision和许可原文尚未锁定，不能用Mobile仓库LICENSE覆盖所有权重；新增适配代码保留版权，不为整仓自选MIT。见 `docs/mobilevla-license-provenance007.md`。
- 历史draft/staging、002/003部署包、旧失败和成功原件保留。`docs`中的历史草稿保持当时内容，当前结果以本报告和相应run原件为准。冻结源码和原始证据不因报告更新而改字节。
- `.gitattributes` 关闭文本换行转换，以保持Windows/Linux clone后SHA一致；shell内容使用LF。`.gitignore`只在任务运行目录忽略smoke日志，封存的evidence日志保留。权重、运行缓存和原生二进制不入代码仓库。
- 最终源码目录另含 `STANDARD-MANIFEST-007-004.json`，由外部封包receipt锁定manifest SHA。Push后应从私有仓库fresh clone并逐文件核SHA和tracked集合；该Git闭环由后续实际回执证明，不以本地包成功代替。
