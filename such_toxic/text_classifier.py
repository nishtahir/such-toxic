import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel

from such_toxic.mean_pooler import MeanPooler


class TextClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        hidden_size: int = 384,
        embedding_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dropout: float = 0.1,
    ):
        super(TextClassifier, self).__init__()
        self.embedding = AutoModel.from_pretrained(embedding_name)
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.mean_pooler = MeanPooler()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        embeddings = self.embedding(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooler(embeddings[0], attention_mask)
        normalized = F.normalize(pooled, p=2, dim=1)
        logits = self.linear(normalized)
        dropped = self.dropout(logits)
        out: Tensor = self.sigmoid(dropped)
        return out
