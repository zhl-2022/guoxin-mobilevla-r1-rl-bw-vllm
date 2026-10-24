# MobileVLA-R1 RGB RL on BW + vLLM 0.18.1 DAS

## 1. 简介

本工程将固定 `AIGeeksGroup/MobileVLA-R1` RGB RL checkpoint 适配到超算互联网单张 BW 64GB、原生 vLLM / HCU 后端，保持 Llama3 8B、SigLIP 与发布方2×2 downsample projector，新增专用模型类、processor、权重布局与输入协议处理。

**003已完成真实BW模型加载、1帧 / 8帧同输入HF组件参考比较。两组原生实际输入分别269 / 1648个token，均保留发布方双BOS；两组各32个生成raw ID与HF逐项相等，结束原因为length。独立下载原件审计通过，owned进程 / 进程组 / 端口清理通过。003标准七项HTTP复跑原件待补入本报告。正式导航准确率、正式性能、官方验收和最终提交均未完成。**

| 对象 | 固定身份 |
| --- | --- |
| 发布权重 | `AIGeeksGroup/MobileVLA-R1`，revision `0c757f3e9d1b95d8416b42e7089e1c04f3e6118a`，`weight/rl/` |
| 发布方代码 | commit `9d98cce3b623dfddf479740d78869ec30364b75a` |
| 实测适配修订 | `007-bos003`；工具ZIP SHA-256 `aa62f6e5dde19cd5fcd4ed6c1cfb599a8a7739d32b9a58bbd8e1fe83522018e6` |
| 同输入实验run | `d80323e361d3489d7b68090b7a2a3ac2` |
| 同输入原件 | `dist/000-session007-mobile-comparison002.zip`，82,473 B，SHA-256 `df35fd6707b67956b573130c71f394f84df57b496b6c6df7bbd870892a233634` |
| 私有仓库 | [zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm](https://github.com/zhl-2022/guoxin-mobilevla-r1-rl-bw-vllm)；实际push / clone核验由最终交付回执记录 |
| 赛事别名 | `BAAI/MobileVLA-R1`；尚缺官方别名等价关系和RGB-only任务范围确认 |

标准入口在 `tasks/mobilevla-r1-rl-bw-vllm-delivery007-draft/`，提供 `env.sh`、`env_check.sh`、`run.sh`、`smoke.sh`、`verify.py`、`bench.sh`、`stop.sh`。这些薄包装委托同仓库的003原任务。原任务已在BW运行；标准包装的shell语法 / dry-run已通过，不将其单独写成又一次GPU重放。

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
| 服务 | `127.0.0.1:30139`；served name `mobilevla-r1-rgb-rl006`为稳定接口名，实际源码修订由003源码SHA和receipt绑定 |
| 原始权重 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-rl006/weight/rl` |
| 003派生视图 | `/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-bos003` |
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
| 003采样语义 | 发布方`inference.py sample_prepared`显式EOS128009；native加`--generation-config vllm`，比较请求与HF固定greedy、max32、单EOS128009 | checkpoint原双EOS文件不改；两组比较均length，尚未实测EOS触发 |
| prepare / strict加载 | 原16文件、6个safetensors header和8个派生小文件SHA；4个LLM shard硬链接，vision / projector原生secondary sources；返回loaded set供非量化DefaultLoader严格检查 | 真实4+1+1 shard加载及随后推理支持全参数严格路径推断，未独立枚举全部GPU参数 |
| 新derived receipt | 独立003目录 / schema，绑定7个适配模块SHA，runtime拒绝旧receipt或源码漂移 | 原derived006、冻结002、第一次失败原件均保留 |
| owned生命周期 | 新run32hex、PID start token、精确配置 / 命令、bounded subprocess、finally cleanup | native与final两次独立记录均PASS，未按端口或进程名杀其他服务 |

RL16项提取记录CRC、长度和SHA；后续14个覆盖卷published LFS完整SHA证明将这16项缓存绑定到发布字节，补齐gap552,355,933 B。独立审计检查了固定锁、回执一致性及逐段无洞 / 无重叠映射；本地未重新读取远端17GB大字节，`all_26_full_volume_bytes_verified=false`。不能将14覆盖卷证明写成26卷全仓 / SFT证明。

来源链、真实tokenizer执行与失败定位见 `docs/mobilevla-bos003-source-alignment.md`、`evidence/session007/mobilevla-bos003-source-diagnosis.json`。本地4.57.3 tokenizer执行、真实native5.5.0 probe和完整GPU比较分别记录，没有冒称运行了发布方4.37.2完整环境或官方wrapper baseline。

## 4. 服务启动脚本

已有原始16项权重与extraction receipt时，完整同输入实验可重放如下。项目必须位于`/root/private_data`下，输出目录必须全新，且先确认本实验设备没有其他服务占用。

```bash
/usr/bin/python -B scripts/mobilevla_comparison_orchestrator007_002.py \
  --frozen "$PWD" --comparison "$PWD" --python /usr/bin/python \
  --output /root/private_data/mobilevla-comparison-evidence007-replay001
```

控制器按需新建003派生视图，验证70项候选和5项helper / probe源锁，运行只tokenizer的native preflight，再启动新owned run、采原生1/8帧、精确停止、检查terminal state / start token / live进程组 / 端口，确认通过后运行HF1/8和raw ID比较，最后再次owned cleanup。总预算最多1800秒，工作步骤预留65秒清理，日志和JSON均独立新建。

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

003标准七项HTTP复跑已由主流程启动；该运行的完整raw尚待纳入，当前不预填通过。历史002的health、models、单图、双图、多轮首轮、多轮追问和图像流式均通过transport / 非空输出检查，原始日志、独立审计和警告继续保留；历史002结果不能直接替代003标准复跑。

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

本次1/8帧比较是独立标准组件HF工程参考，未执行官方完整wrapper / benchmark。两组各32个raw ID相等仅覆盖相同开发fixture与同一greedy策略；输出均截断，不证明导航动作正确、完整回答、全部场景、长序列或EOS实际触发。

## 7. 性能测试

当前为4096上下文、并发1、eager开发profile，没有正式TTFT、TPOT、tokens/s或优化收益。原checkpoint最大位置8192，不能容纳正式T02的16K输入加1K输出；没有静默扩展位置或截断以填性能表。

本次完整控制器359.706 s包含逐文件prepare、native tokenizer probe、服务加载、客户端 / HF模块导入、两组请求和清理。HF记录的11.969 / 14.903 s是其内部模型加载及生成区间；客户端stage墙钟又包含模块导入。这些区间不同，不作为native / HF速度比较或正式性能。

标准 `bench.sh` 对当前pilot profile拒绝实际正式执行，`official_performance_verified=false`。仍需主办方固定图像 / 输入、输出长度、并发1/8/64、4K/16K档、warmup、两轮和第二轮原始日志 / 汇总；KV cache理论容量不是实测吞吐。

## 8. 注意事项

- 完整OCI镜像、平台截图、私有仓库最终commit / clone全SHA、评审访问、正式评测和最终表单仍按实际缺项记录。`submission_ready=false`、`submitted=false`，最终表单由参赛者手动提交。
- 保留tokenizer incorrect regex、ROCm GELU(tanh)→exact fallback、eager、无效sparse fusion及NCCL `destroy_process_group()`警告；未自动加`fix_mistral_regex=True`改变tokenizer。两组实际输入和raw生成一致是本次实测范围内的证据。
- 发布方代码Apache-2.0及LLaVA / NVIDIA notice、Mobile权重模型卡`cc-by-sa-4.0`分别保留。LLM / vision精确底座repo、revision与许可原文尚未锁定，不能用Mobile仓库LICENSE覆盖所有权重。新编代码保留版权，不为整仓自选MIT；详见 `docs/mobilevla-license-provenance007.md`。
- 002、第一次输入不一致raw、003修订和成功raw均保留。源码 / 证据跨Windows和Linux保持原始字节，`.gitattributes`关闭换行转换；shell现有内容为LF。模型权重、运行目录和缓存不入私有代码仓库。
- 本标准交付仍等待003七项HTTP新原件，未最终封存。获得原件后只更新交付报告与新增回执，不修改已运行003候选和它的权重 / source锁。
