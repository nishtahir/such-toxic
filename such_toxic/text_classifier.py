from torch import Tensor, nn


class TextClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 384,
    ):
        super(TextClassifier, self).__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embedding: Tensor) -> Tensor:
        logits = self.linear(embedding)
        predictions: Tensor = self.sigmoid(logits)
        return predictions
