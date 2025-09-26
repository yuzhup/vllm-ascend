from dataclasses import dataclass, field

import torch
import torch_npu

from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_config import WeightPrefetchConfig
from vllm_ascend.utils import current_stream, prefetch_stream, npu_stream_switch

SUPPORTED_MODULES = ["attn", "mlp", "moe"]
MOE_PREFETCH_TOKEN_THRESHOLD = 96


@dataclass
class ModuleWeightPrefetchConfig:
    module_name: str
    enable: bool = False
    prefetch_ratio: dict = field(default_factory=dict)
    is_active_this_forward: bool = False

    def __post_init__(self) -> None:
        self.prefetch_ratio = {
            prefix: ratio
            for prefix, ratio in self.prefetch_ratio.items()
            if 0 <= ratio <= 1
        }

        assert self.module_name in SUPPORTED_MODULES, (
            f"Invalid module name {self.module_name}, should be one of {SUPPORTED_MODULES}")

        if self.module_name in SUPPORTED_MODULES:
            self.enable = self.enable and any(self.prefetch_ratio.values()) > 0


class WeightPrefetchMethod:
    """
    Unified weight prefetch method.
    """

    def __init__(self, weight_prefetch_config: WeightPrefetchConfig) -> None:
        self.calculation_stream = current_stream()
        self.prefetch_stream = prefetch_stream()

        self.attn = ModuleWeightPrefetchConfig(
            module_name="attn",
            enable=weight_prefetch_config.enabled,
            prefetch_ratio=weight_prefetch_config.prefetch_ratio.get("attn", {}),
        )
        self.moe = ModuleWeightPrefetchConfig(
            module_name="moe",
            enable=weight_prefetch_config.enabled,
            prefetch_ratio=weight_prefetch_config.prefetch_ratio.get("moe", {}),
        )

    def maybe_prefetch_attn_weight_preprocess(self,
                                              prefix: str,
                                              weight: torch.Tensor,
                                              start_flag: torch.Tensor) -> None:
        if not self.attn.enable:
            return

        weight_size = weight.data.element_size() * weight.data.numel() * self.attn.prefetch_ratio.get(prefix, 0)

        self.calculation_stream = torch_npu.npu.current_stream()
        self.weight_prefetch_impl(weight=weight,
                                  start_flag=start_flag,
                                  max_weight_size=int(weight_size))

    def maybe_prefetch_attn_weight_postprocess(self) -> None:
        if self.attn.enable and self.prefetch_stream is not None:
            self.calculation_stream.wait_stream(self.prefetch_stream)

    def update_forward_param(self, num_tokens: int):
        if self.moe.enable:
            self.moe.is_active_this_forward = num_tokens >= MOE_PREFETCH_TOKEN_THRESHOLD

    def maybe_prefetch_moe_weight_preprocess(self, prefix):
        if not self.moe.is_active_this_forward:
            return
        forward_context = get_forward_context()
        weight = forward_context.model_instance.model.layers[forward_context.layer_idx].mlp.experts.w13_weight
        weight_size = weight.data.element_size() * weight.data.numel() * self.moe.prefetch_ratio.get(prefix, 0)
        self.calculation_stream = torch_npu.npu.current_stream()
        self.weight_prefetch_impl(weight=weight,
                                  start_flag=None,
                                  max_weight_size=int(weight_size))
        forward_context.layer_idx += 1

    def maybe_prefetch_moe_weight_postprocess(self):
        if self.moe.is_active_this_forward and self.prefetch_stream is not None:
            self.calculation_stream.wait_stream(self.prefetch_stream)

    def weight_prefetch_impl(self,
                             weight: torch.Tensor,
                             start_flag: torch.Tensor,
                             max_weight_size: int) -> None:
        self.prefetch_stream.wait_stream(self.calculation_stream)
        with npu_stream_switch(self.prefetch_stream):
            torch.ops.vllm.maybe_npu_prefetch(inputs=weight,
                                              dependency=start_flag,
                                              max_size=max_weight_size)

