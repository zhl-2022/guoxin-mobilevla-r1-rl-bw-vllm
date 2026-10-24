"""Isolated MobileVLA RGB RL adapter; new adaptation copyright reserved."""

ARCHITECTURE = "MobileVLAForConditionalGeneration006"


def register():
    """vLLM general plugin entry point, also loaded in worker processes."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCHITECTURE,
        "mobilevla_bw006.backend:MobileVLAForConditionalGeneration006",
    )
