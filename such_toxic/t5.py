import torch
import torch.nn as nn
from transformers import T5ForSequenceClassification
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput


class T5Toxic(nn.Module):
    def __init__(self, model_name: str = "t5-small", num_labels: int = 6) -> None:
        super(T5Toxic, self).__init__()
        self.t5 = T5ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Seq2SeqSequenceClassifierOutput:
        return self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
