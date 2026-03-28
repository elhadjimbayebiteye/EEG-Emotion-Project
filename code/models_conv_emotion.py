import torch
import torch.nn as nn

class CompactEmotionCNNArticle(nn.Module):
    """
    Version adaptée SEED du modèle 'Convolving Emotions'
    - Compact mais capable d'exploiter 62 canaux
    """

    def __init__(self, num_channels: int, seq_len: int, num_classes: int = 3):
        super().__init__()

        # Couche 1 : mélange initial des canaux
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=1
        )

        # Couche 2
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool1d(2)

        # Couche 3
        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.act3 = nn.Tanh()
        self.pool3 = nn.MaxPool1d(2)

        # Calcul automatique de la taille final pour FC
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, seq_len)
            out = self.conv1(dummy)
            out = self.pool2(self.act2(self.conv2(out)))
            out = self.pool3(self.act3(self.conv3(out)))
            flat = out.shape[1] * out.shape[2]

        self.fc = nn.Linear(flat, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        x = x.flatten(start_dim=1)
        return self.fc(x)
