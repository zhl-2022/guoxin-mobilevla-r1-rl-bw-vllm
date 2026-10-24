# MobileVLA 代码与权重许可来源清单

本清单分别记录当前固定原件能够支持的来源与声明。仓库不使用统一MIT文件覆盖所有代码或模型权重；新增适配代码保留版权说明。

| 对象 | 已有固定原件 | 当前能够陈述的声明 | 尚缺信息 |
| --- | --- | --- | --- |
| MobileVLA 发布方代码 | commit `9d98cce3b623dfddf479740d78869ec30364b75a` 的 `LICENSE` 与完整source文件锁 | 发布方仓库提供 Apache-2.0；原始文件内 LLaVA / NVIDIA copyright notice 继续保留 | 最终交付时保持对应原文与修改来源可追溯 |
| MobileVLA 发布权重 | revision `0c757f3e9d1b95d8416b42e7089e1c04f3e6118a` 的模型卡与HF metadata | 发布模型卡声明 `cc-by-sa-4.0` | 该模型卡声明不能代替各底座的独立来源与许可原文 |
| LLM 组件 | `weight/rl/llm/config.json`、原始四个safetensors与index / SHA | 配置记录 `LlamaForCausalLM`、8B规模相关结构；本适配未改权重 | 精确底座仓库、revision、上游权重来源映射与许可原文尚未随本包锁定；不自行用Mobile仓库LICENSE覆盖 |
| Vision 组件 | `weight/rl/vision_tower/config.json`、preprocessor与原始safetensors / SHA | 配置记录 `SiglipVisionModel`，27层、1152隐藏维度、384输入 / 14patch | 精确底座仓库、revision与上游许可原文尚未随本包锁定；不仅凭架构名推定许可 |
| 原生框架来源节选 | 固定 vLLM 0.18.1 源码与source锁；文件内原始声明 | 保留各节选原始 Apache-2.0 / copyright notice，DAS native实际包版本单独记录 | 原生镜像与发行物的完整来源、digest和对应许可随平台交付材料补齐 |
| 新编 adapter / 工具 / 控制器 | 独立003候选、comparison helpers、owned controller、源码SHA与变更说明 | 新编部分保留版权，不为整个混合仓库自选MIT或其他许可 | 最终私有代码保存与评审访问方式由主流程实际验证 |

模型卡原件位于 `evidence/mobilevla-preflight-20261023/hf-README.md` 与 `hf-model.json`；发布方固定代码原件及SHA位于同目录的 `source/` 与 `source-file-lock.json`。候选中的 `reference/UPSTREAM-LICENSE-APACHE-2.0.txt` 保留了发布方代码许可。精确底座来源缺项以本清单和最终报告第8节的状态为准。
