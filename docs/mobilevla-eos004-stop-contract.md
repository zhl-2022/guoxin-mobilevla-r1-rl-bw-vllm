# MobileVLA 004 发布方 EOS 对齐修订

本修订只准备 CPU 证据、原生停止配置修复和有界重跑入口，004 的 GPU 结果尚未产生。002、003、原始权重、提取回执以及全部失败原件均保留。

## 1. 已证实的问题

003b 七项功能 smoke 通过，但单图、多图、首轮和 SSE 的原始 choice.stop_reason 为整数 128001。该结果不能证明发布方停止语义一致。003 的 1/8 帧对照各 32 个 raw token 相等，均由长度上限停止，也没有验证 EOS。

发布方 inference.py 的 sample_prepared 显式传入 tokenizer.eos_token_id=128009。锁定的原始 llm/config.json 与 tokenizer 也是 128009，原始 llm/generation_config.json 则包含 [128001, 128009]；不得把源 LLM 配置误记成 128001。

vLLM v0.18.1 的 ModelConfig.try_get_generation_config() 在 auto 和 vllm 两种模式都读取模型 generation_config.json。get_diff_sampling_param() 在 vllm 模式只清空普通采样默认值；InputProcessor 仍把 generation_config_fields 交给 SamplingParams.update_from_generation_config()，后者将额外的 128001 加入停止集合。BaseRenderer.get_eos_token_id() 读取 tokenizer EOS。scheduler 中 primary EOS 分支不设置整数 stop_reason，额外停止 token 分支设置整数 stop_reason。因此旧 raw 与额外 EOS 泄漏相符。

## 2. 独立修复

新目录 /root/private_data/guoxin-mobilevla-tools-007-004，派生模型 /root/private_data/guoxin-candidates/models/mobilevla-r1-derived007-eos004。新 prepare 回执使用 mobilevla-derived-hf-view-007-eos004-v1，绑定七个适配器模块 SHA 和八个派生小文件 SHA。源 16 文件与原提取回执仍逐字节验证，四个 LLM 分片硬链接，视觉和 projector 原样由严格 secondary loader 读取。

只有派生 generation_config.json 的 eos_token_id 从 [128001, 128009] 改为 128009；root config 与 text config 明确保持 128009。receipt.publisher_eos_mapping 记录原值与派生值。runtime 会拒绝任一层 EOS 不同的派生视图，包括重新封存但含额外 EOS 的回执。源文件不修改。

probe_mobilevla_tokens007_004.py 在不实例化权重的前提下执行实际安装的 ModelConfig、BaseRenderer、SamplingParams 方法，记录四个模块源码 SHA、HF/vLLM 实际 config 对象的 EOS、有效停止集合，并用源 generation config 复现旧额外 EOS 集合。它同时保留 1/8 帧双 BOS 的实际 native tokens-only 路径检查。

## 3. GPU 重跑入口

在新包安全展开后执行 scripts/mobilevla_comparison_orchestrator007_003.py --output <private_data 中不存在的新目录>，总预算最多 1800 秒。默认先自动 prepare004 和配置探针，再启动 exact owner run。依次运行 1 帧与 8 帧各 max32 的原对照，另加 1 帧中文短问句 max128；不增加 min_tokens，不忽略 EOS，不强迫输出 token。native 停止并独立验证 owner、PID start token、进程组和端口后，才运行三个同输入 HF reference 并比较 raw token IDs，最后再次 owned cleanup。

长对照必须 raw IDs 全部相等，而且双方至少观察到 128001 后仍有生成，或者末 token 为 128009 且 finish_reason=stop。若只有 length 前缀相等又没有跨停止边界，controller 返回非零并保留 completed_equal_without_stop_boundary_coverage。primary 128009 的 stop_reason 可以为 null；验证器允许该原生口径。若只跨过 128001 而未见 128009，会分别记录，不能称已证明 EOS 真实触发。

## 4. 证据与边界

public v0.18.1 配置、renderer、input processor、sampling 和 scheduler 源码及 SHA 锁位于 evidence/session007/mobile-eos004-vllm-source。CPU AST 测试复现旧泄漏及新单 EOS；真实配置方法执行、完整模型加载、HTTP 和长生成比较需由 GPU 重跑提供。此次没有导航准确率、官方性能、平台验收或最终提交结论。代码许可与底座权重来源分别记录，新增适配代码没有自选整仓许可证。
