"""Official-HPL-compatible heteroscedastic uncertainty learner."""

from torch import nn


class UncertaintyLearner(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=1):
        super().__init__()
        layers = []
        for layer in range(num_layers):
            layers.extend((
                nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
            ))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.fc = nn.Sequential(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, value):
        return self.fc(value)
