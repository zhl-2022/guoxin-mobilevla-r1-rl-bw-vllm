# MobileVLA-R1 RGB RL on BW + vLLM 0.18.1 DAS

## 1. 简介

本工程将固定 `AIGeeksGroup/MobileVLA-R1` RGB RL checkpoint 适配到超算互联网单张 BW 64GB、原生 vLLM / HCU 后端，保持 Llama3 8B、SigLIP 与发布方2×2 downsample projector，新增专用模型类、processor、权重布局与输入协议处理。

**004 已完成独立 EOS 修复、45 项 CPU 回归和严格部署包验证，真实 GPU 重跑尚待纳入。历史 003 的 1/8 帧同输入比较各 32 个 raw ID 与 HF 相等，均由 length 结束；003b 七项 HTTP 功能通过，但四例 stop_reason=128001 暴露额外 EOS 泄漏，因此不能作为发布方停止语义一致的证据。正式导航准确率、正式性能、官方验收和最终提交均未完成。**

| 对象 | 固定身份 |
| --- | --- |
| 发布权重 | `AIGeeksGroup/MobileVLA-R1`，revision `0c757f3e9d1b95d8416b42e7089e1c04f3e6118a`，`weight/rl/` |
| 发布方代码 | commit `9d98cce3b623dfddf479740d78869ec30364b75a` |
| 当前适配修订 | `007-eos004`；工具 ZIP SHA-256 `f248a692df9ae69af29dae264e4ef6e659269593b25057b26eb17e18e2ad2336`；当前仅 CPU 验证 |
| 历史003修订 | `007-bos003`；工具 ZIP SHA-256 `aa62f6e5dde19cd5fcd4ed6c1cfb599a8a7739d32b9a58bbd8e1fe83522018e6` |
| 同输入实验run | `d80323e361d3489d7b68090b7a2a3ac2` |
| 同输入原件 | `dist/000-session007-mobile-comparison002.zip`，82,473 B，SHA-256 `df35fd6707b67956b573130c71f394f84df57b496b6c6df7bbd870892a233634` |
| 私有仓库 | [zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm](https://github.com/zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm)；实际push / clone核验由最终交付回执记录 |
| 赛事别名 | `BAAI/MobileVLA-R1`；尚缺官方别名等价关系和RGB-only任务范围确认 |

标准入口在 `tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft/`，提供 `env.sh`、`env_check.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh`、`stop.sh`。这些薄包装委托同仓库的004原任务。历史003原任务已在BW运行；标准包装的shell语法 / dry-run已通过，不将其单独写成又一次GPU重放。

## 2. 验证环境

| 组件 / 配置 | 实际记录与边界 |
| --- | --- |
| 平台 / 卡 | 超算互联网BW64GB单卡；本轮设备记录为`BW`，HCU平台插件激活 |
| Python | 3.10.12，实际入口`/usr/bin/python` |
| Torch | 原生distribution metadata `2.10.0+das.opt1.dtk2604.20260325.g6b060a`；HF进程`torch.__version__`返回`2.10.0`，分别保留 |
| vLLM / vllm-hcu | `0.18.1+das.fa71803.dtk2604` / `0.18.1+das.dtk2604` |
| Transformers / tokenizers | `5.5.0` / `0.22.2`；真实源 / 派生tokenizer类均为`TokenizersBackend` |
| Triton / HIP | `3.6.0+gitc73250c4.staging` / `6.3.26093` |
| 精度 / 并行 | BF16、TP1、非量化、eager |
| 开发profile | `max_model_len=4096`、`max_num_seqs=1`、最多8帧RGB，GPU内存比例0.80 |
| 服务 | `127.0.0.1:30139`；served name `mobilevla-r1-rgb-rl006`为稳定接口名，实际源码修订由004源码 SHA 和新 receipt 绑定 |
| 原始权重 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-rl006/weight/rl` |
| 004独立派生视图 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-eos004`；原003目录保留 |
| 完整镜像 / DTK文件 | 完整OCI registry/repository:tag和digest尚缺；`/opt/dtk/.info/version`不存在的失败原样保留，不能用包名后缀替代镜像证据 |

适配入口不覆盖安装原生GPU栈，也不执行发布方CUDA环境安装脚本。真实native token处理文件SHA为 `900283f4fabd65449ac76b8a08d3d8e546321d4c88d046c3d510cfc5fabde1bf`，与固定vLLM源文件一致；003 backend SHA为 `5d529bd662867463d2f7f3fabb24dc5c8629976a136cbdeab5aff8b349e2c0d8`。

## 3. 适配优化方案

| 位置 | 实际行为 / 原因 | 验证与限制 |
| --- | --- | --- |
| 专用backend / registry | 注册 `MobileVLAForConditionalGeneration006`，Llama和SigLIP继续由原生vLLM执行；标准LlavaConfig仅承载组件配置 | 真实服务解析到专用类，非architecture改名后直接套标准LLaVA |
| SigLIP | 保留27层checkpoint，选择`hidden_states[-2]`对应前26层全部729patch；无末层pooling替代 | 固定源码、header和CPU合同；真实BW执行 |
| projector | 奇数网格补零后2×2 downsample；LN4608、Linear4608→4096、GELU、Linear4096→4096 | 发布方真实AST / 数值对照；每帧196token；未改权重 |
| RGB processor | 每帧按发布方先RGB转换、resize384×384，再SigLIP预处理，保持帧顺序 | CPU原processor对照、真实native token-only与1/8HTTP对照 |
| 源token语义 | 源`-200`映射为输入占位128259；LLM embedding / head仍为128259行，不增加可生成行 | 原tokenizer及组件header封存，196占位 / 帧 |
| 003 tokens-only修复 | OpenAI renderer产生token list时，默认分支绕过专用文本tokenization；003在真实tokens-only路径无cleanup解码对应文本，再执行同一发布方`tokenizer_image_token`流程 | 发布方模板自带BOS和checkpoint自动BOS均保留；修复前268 / 269差异原件保留；未删API / 参考 / 生成token |
| 004停止语义修复 | 发布方 `sample_prepared` 显式 EOS128009；vLLM 在 `--generation-config vllm` 下仍读取 generation config 的特殊 token。新派生 generation_config 把 `[128001,128009]` 映射为128009；root/text 原值保持128009 | 源文件不改；原生实际配置方法 probe 必须证实有效停止集合仅128009；长对照分别记录跨128001与末尾128009 |
| prepare / strict加载 | 原16文件、6个safetensors header和8个派生小文件SHA；4个LLM shard硬链接，vision / projector原生secondary sources；返回loaded set供非量化DefaultLoader严格检查 | 真实4+1+1 shard加载及随后推理支持全参数严格路径推断，未独立枚举全部GPU参数 |
| 新derived receipt | 独立004目录 / schema，绑定7个适配模块 SHA、8个派生小文件 SHA 和原16文件 SHA；runtime还拒绝任何额外EOS | 原derived006/003、冻结002/003、全部历史原件均保留 |
| owned生命周期 | 新run32hex、PID start token、精确配置 / 命令、bounded subprocess、finally cleanup | native与final两次独立记录均PASS，未按端口或进程名杀其他服务 |

RL16项提取记录CRC、长度和SHA；后续14个覆盖卷published LFS完整SHA证明将这16项缓存绑定到发布字节，补齐gap552,355,933 B。独立审计检查了固定锁、回执一致性及逐段无洞 / 无重叠映射；本地未重新读取远端17GB大字节，`all_26_full_volume_bytes_verified=false`。不能将14覆盖卷证明写成26卷全仓 / SFT证明。

004 EOS 来源链与真实配置探针见 `docs/mobilevla-eos004-stop-contract.md`。锁定源 LLM config / tokenizer EOS 均为128009，旧缺陷来自复制的 generation config 双EOS；不能把源 text config 误记为128001。历史 BOS 来源链、真实tokenizer执行与失败定位见 `docs/mobilevla-bos003-source-alignment.md`、`evidence/session007/mobilevla-bos003-source-diagnosis.json`。本地4.57.3 tokenizer执行、真实native5.5.0 probe和完整GPU比较分别记录，没有冒称运行了发布方4.37.2完整环境或官方wrapper baseline。

## 4. 服务启动脚本

已有原始16项权重与extraction receipt时，完整同输入实验可重放如下。项目必须位于`/root/private_data`下，输出目录必须全新，且先确认本实验设备没有其他服务占用。

```bash
/usr/bin/python -B scripts/mobilevla_comparison_orchestrator007_003.py \
  --frozen "$PWD" --comparison "$PWD" --python /usr/bin/python \
  --output /root/private_data/mobilevla-comparison-evidence007-eos004-replay001
```

控制器按需新建004派生视图，验证70项候选和5项 helper / probe 源锁，执行无权重的实际 native token 与 EOS 配置方法。随后启动新 owned run，运行1/8帧各 max32，以及1帧中文短问 max128。精确停止、检查 terminal state / start token / live进程组 / 端口后，才运行三个同输入 HF reference 和 raw ID 比较，最后再次 owned cleanup。长对照必须两端 raw 相等，并至少自然跨过128001或观察128009终止；否则不标停止边界验证通过。总预算最多1800秒，工作步骤预留65秒清理，日志和JSON均独立新建。

标准服务入口示例：

```bash
cd tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft
bash run.sh --dry-run --run-id 654118905a9e4fd8a5d5abe88a0ca97a
bash run.sh --run-id <new-32-hex-owned-run-id>
bash env_check.sh > env.log
bash smoke.sh > smoke-stdout.log
bash stop.sh --run-id <same-owned-run-id>
```

原生runtime日志仍由原任务保存，运行后按run marker连同精确state/config封存；不能把append日志内其他run混入本轮结论。标准stop缺少精确32hex run ID时退出2。

## 5. Smoke 测试

历史003b标准任务真实七项功能复跑已完整审计，owner `1ed4c0d6c350796e0b49df353ed04ba9`。原件 `dist/000-session007-mobile-pilot003b.zip` 为45,876 B，SHA-256 `cdb16ce39102beb8037ebf0f26b9527c61d3b3df84482f681f5b25cec19940e8`；42个payload完整SHA/CRC、9个实际源码SHA与冻结003匹配、原生BW和4+1+1权重分片加载均通过。权重加载日志7.42 s仅为该次记录。

| 003b用例 | 功能结果 | 原始结束信息 |
| --- | --- | --- |
| health / models | HTTP200，通过 | served模型路径与003派生目录一致 |
| 单图 / 多图 / 多轮首轮 | HTTP200、非空输出，通过 | stop，额外停止token128001 |
| 多轮追问 | HTTP200、非空输出，通过 | stop，stop_reason=null；无raw尾token，不能独立指定EOS |
| 图像SSE | 非空delta、最终stop与DONE，通过 | stop_reason128001，13个事件 |

上述七项功能通过，同时 `publisher_stop_semantics_consistent=false`。全部没有length截断，不能因此把错误的128001停止写成发布方EOS触发。审计见 `evidence/session007/mobile-pilot003-independent-audit.json`。独立清理UTC `2026-10-24T07:46:21.411132+00:00`：supervisor9302 / server9303均 absent，start token285500162 / 285500195与owner一致，30139已关闭。tokenizer regex、ROCm GELU fallback与未显式destroy_process_group警告保留。

004的完整同输入及标准HTTP原件尚待纳入；以下003短生成结果只作为历史输入/前缀证据，不替代004停止语义重跑。

003同输入比较已取得以下独立审计结果，详见 `evidence/session007/mobile-comparison002-independent-audit-b.json`：

| 输入 | 原生实际输入 | HF展开输入 | 生成raw ID | 结束原因 | 比较 |
| --- | ---: | --- | ---: | --- | --- |
| 1帧red | 269，双BOS、196图像占位 | `[1,269,4096]` | 两端各32 | 两端length | 逐项全等，首分歧null |
| 8帧red/blue交替 | 1648，双BOS、1568图像占位 | `[1,1648,4096]` | 两端各32 | 两端length | 逐项全等，首分歧null |

HF的pixel为`[N,3,384,384]`、vision为`[N,729,1152]`、projector为`[N,196,4096]`；形状与摘要随raw原件保留。API原回包的prompt IDs、choice IDs、usage与helper记录一致，comparison输入文件SHA通过独立重算。

原生停止及HF完成后最终检查均归属同run：supervisor4908 / server4909，start token285428146 / 285428157，两进程均absent、live进程组空、30139连接被拒绝，最终检查UTC `2026-10-24T07:32:27.487692Z`。这是记录中的owned清理证明，不冒称本地重新探测了远端设备或证明任意其他GPU进程不存在。

## 6. 精度验证

目前缺少本题绑定的官方evaluator、自检、固定导航样本 / 动作参考、GPU baseline、误差容忍契约、赛事别名及RGB-only范围确认。`official_accuracy_verified=false`、`official_acceptance=false`；不生成虚构导航准确率、success rate或正式PASS。

标准 `verify.py` 委托受控官方材料gate，缺少 `MOBILE_OFFICIAL_VERIFY_SCRIPT`、`MOBILE_OFFICIAL_DATA_DIR`、`MOBILE_OFFICIAL_BASELINE` 等材料时退出2。有材料后封存其脚本 / 数据 / baseline哈希、自检和原始输出，再按真实评分契约解释。

历史003的1/8帧比较是独立标准组件HF工程参考，未执行官方完整wrapper / benchmark。两组各32个raw ID相等仅覆盖相同开发fixture与同一greedy策略；输出均截断，不证明导航动作正确、完整回答、全部场景、长序列或EOS实际触发。

## 7. 性能测试

当前为4096上下文、并发1、eager开发profile，没有正式TTFT、TPOT、tokens/s或优化收益。原checkpoint最大位置8192，不能容纳正式T02的16K输入加1K输出；没有静默扩展位置或截断以填性能表。

历史003完整控制器359.706 s包含逐文件prepare、native tokenizer probe、服务加载、客户端 / HF模块导入、两组请求和清理。HF记录的11.969 / 14.903 s是其内部模型加载及生成区间；客户端stage墙钟又包含模块导入。这些区间不同，不作为native / HF速度比较或正式性能。

标准 `bench.sh` 对当前pilot profile拒绝实际正式执行，`official_performance_verified=false`。仍需主办方固定图像 / 输入、输出长度、并发1/8/64、4K/16K档、warmup、两轮和第二轮原始日志 / 汇总；KV cache理论容量不是实测吞吐。

## 8. 注意事项

- 完整OCI镜像、平台截图、私有仓库最终commit / clone全SHA、评审访问、正式评测和最终表单仍按实际缺项记录。`submission_ready=false`、`submitted=false`，最终表单由参赛者手动提交。
- 保留tokenizer incorrect regex、ROCm GELU(tanh)→exact fallback、eager、无效sparse fusion及NCCL `destroy_process_group()`警告；未自动加`fix_mistral_regex=True`改变tokenizer。两组实际输入和raw生成一致是本次实测范围内的证据。
- 发布方代码Apache-2.0及LLaVA / NVIDIA notice、Mobile权重模型卡`cc-by-sa-4.0`分别保留。LLM / vision精确底座repo、revision与许可原文尚未锁定，不能用Mobile仓库LICENSE覆盖所有权重。新编代码保留版权，不为整仓自选MIT；详见 `docs/mobilevla-license-provenance007.md`。
- 002、第一次输入不一致raw、003修订和成功raw均保留。源码 / 证据跨Windows和Linux保持原始字节，`.gitattributes`关闭换行转换；shell现有内容为LF。模型权重、运行目录和缓存不入私有代码仓库。
- 本标准交付等待004完整GPU比较及标准HTTP原件，未最终封存；004的45项本地CPU测试、另一代理独立45项CPU测试、104项部署payload的严格解包/回读/拒覆盖与75项源码锁已通过。003旧staging与草稿保留，当前独立004 staging继续补证，不修改已运行候选及权重原件。
