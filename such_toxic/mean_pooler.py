from typing import Optional

import torch
from torch import nn


def mean_pooling(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = mask.unsqueeze(-1).expand(inputs.size()).float()
    sum_embeddings = torch.sum(inputs * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class MeanPooler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return inputs.mean(dim=1)

        return mean_pooling(inputs, mask)
