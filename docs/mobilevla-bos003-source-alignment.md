# MobileVLA 007-bos003：发布方 BOS / EOS 调用语义对齐

第一次同输入实验的 HTTP 已实际生成32个token，但输入检查失败：原生返回268个输入ID，参考期望269个；图像占位均为196个，差异为开头少一个BOS。原始 ZIP SHA-256 为 `73bdfcb76a37e78ffa369bae1982a85396abcb4c18d5f29d733fd3905f040bdc`，原件保持不变，不能通过删除参考 / API token 将其改写成成功。

## 固定发布方执行链

固定源码 commit `9d98cce3b623dfddf479740d78869ec30364b75a` 的 `inference.py` 调用 `LlavaLlamaModel.from_pretrained`，经 `LlavaMetaModel.init_vlm` 进入 `llava/model/language_model/builder.py`。当前 Llama 分支以 `AutoTokenizer.from_pretrained(use_fast=False, legacy=False, padding_side="right")` 加载。

Checkpoint 的 `tokenizer_class` 实际为 `PreTrainedTokenizerFast`，`tokenizer.json` 的 `TemplateProcessing` 自动增加BOS，而 `llama_3` Conversation 文本已经包含字面BOS。发布方 `tokenizer_image_token` 的 offset 流程保留这两个BOS。真实 tokenizer 与发布方方法 AST 已执行：1帧源ID74 / 展开269，8帧源ID88 / 展开1648，前缀均为 `[128000,128000,128006,...]`。本地库为 Transformers4.57.3 / tokenizers0.22.1；未伪称运行了发布方依赖锁定的4.37.2完整环境。

每个关键源文件 SHA / Git blob 及实际 tokenizer 参数、ID数组保存在 `evidence/session007/mobilevla-bos003-source-diagnosis.json`。独立候选包含对应六份固定源码和来源锁；不运行发布方模型初始化代码，也未加载GPU权重。

## 003 修订

- 原生 `BaseMultiModalProcessor._apply_hf_processor_main` 在 OpenAI 已渲染为 `list[int]` 时调用默认 tokens-only 分支，原样返回，绕过专用文本 tokenization。003覆盖 `_apply_hf_processor_tokens_only`：无cleanup地解码对应文本，再执行同一个 source tokenization，随后由原生路径展开每帧196个占位。没有更改API回包或生成token来制造一致。
- 发布方 `inference.py sample_prepared` 显式传入 tokenizer 的单EOS `128009`。003服务加入 `--generation-config vllm`，新请求与HF参考明确为greedy、max32和 `[128009]`。Checkpoint文件中的双EOS仍保持原件；`128001` 不再额外提前停止。
- 新派生目录为 `/root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-bos003`。只新建视图和receipt，保留原权重与原derived006。新receipt / config标记 `007-bos003` 并封存7个适配模块SHA；runtime拒绝旧receipt和源码漂移。
- 自含候选位于新 `/root/private_data/guoxin-mobilevla-tools-007-003`；002原目录、原ZIP、原derived和第一次失败原件不覆盖。

## 完整重跑

严格展开新003包后执行：

```bash
/usr/bin/python -B /root/private_data/guoxin-mobilevla-tools-007-003/scripts/mobilevla_comparison_orchestrator007_002.py \
  --python /usr/bin/python \
  --output /root/private_data/mobilevla-comparison-evidence007-run002
```

输出目录必须全新。控制器先按需在总1800秒预算内prepare新derived，运行 native tokenizer-only preflight，封存真实native类与来源SHA；然后新owned run、native1/8、finally精确stop、独立PID start token / live进程组 / 端口检查、HF1/8与raw ID比较，最后再次精确清理。

首次失败的控制器行为符合预期：发现输入不等后停止，没有继续HF，最终cleanup通过。003尚未取得真实GPU结果，所有准确率 / 性能 / 验收 / 最终提交状态仍为未验证或false。新候选的CPU回归和来源对齐证据只支持进入实际重跑。
