import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ==============================
# 1) Attention spatiale explicable (ChannelGate)
# ==============================
class ChannelGate(nn.Module):
    

    def __init__(self, num_channels: int, hidden: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_channels, bias=True)
        )
        self.last_channel_weights: Optional[torch.Tensor] = None  # [B, C] sur CPU pour XAI

        self._init_weights()

    def _init_weights(self):
        # Init "Kaiming" stable pour MLP
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, C, T]
        desc = x.abs().mean(dim=-1)                 # [B, C]  moyenne temporelle de |signal|
        logits = self.mlp(desc)                     # [B, C]
        weights = torch.softmax(logits, dim=-1)     # [B, C]  somme=1 par échantillon
        self.last_channel_weights = weights.detach().cpu()
        return x * weights.unsqueeze(-1)            # [B, C, T]


# ==============================
# 2) Encodeur temporel (CNN résiduel + LayerNorm finale)
# ==============================
class TemporalCNN(nn.Module):
    """
    Encodeur temporel 1D avec résidus légers.
    Entrée :  [B, C, T]
    Sortie :  [B, E, T/2]  (MaxPool(2))
    """
    def __init__(self, in_ch: int, embed_dim: int = 128, pdrop: float = 0.30):
        super().__init__()
        E = embed_dim

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, E, kernel_size=5, padding=2),
            nn.BatchNorm1d(E),
            nn.ReLU(inplace=True),
            nn.Dropout(pdrop)
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(E, E, kernel_size=5, padding=2),
            nn.BatchNorm1d(E),
            nn.ReLU(inplace=True),
            nn.Dropout(pdrop)
        )
        self.pool1 = nn.MaxPool1d(2)  # T -> T/2

        self.block2 = nn.Sequential(
            nn.Conv1d(E, E, kernel_size=3, padding=1),
            nn.BatchNorm1d(E),
            nn.ReLU(inplace=True),
            nn.Dropout(pdrop)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(E, E, kernel_size=3, padding=1),
            nn.BatchNorm1d(E),
            nn.ReLU(inplace=True),
            nn.Dropout(pdrop)
        )

        # LayerNorm sur l'axe 'canaux features' après transpose
        self.final_norm = nn.LayerNorm(E)

        self._init_weights()

    def _init_weights(self):
        for m in [*self.stem, *self.block1, *self.block2, *self.block3]:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, C, T]
        x = self.stem(x)                 # [B, E, T]
        x = x + self.block1(x)           # résiduel léger
        x = self.pool1(x)                # [B, E, T/2]
        x = x + self.block2(x)           # résiduel
        x = x + self.block3(x)           # résiduel

        # LayerNorm sur l'axe E (on transpose B,E,T -> B,T,E puis retour)
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return x                         # [B, E, T/2]


# ==============================
# 3) Petite attention temporelle
# ==============================
class TinyTemporalAttention(nn.Module):
    """
    Multi-Head Self-Attention sur l'axe temporel (après CNN).
    Entrée :  [B, E, T] -> permute [T, B, E] pour nn.MultiheadAttention
    Sortie :  [B, E, T] (résiduel + LayerNorm)
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, pdrop: float = 0.15):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=pdrop, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(pdrop * 1.5)  # léger sur-régularisation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, E, T]
        x_t = x.permute(2, 0, 1)                  # [T, B, E]
        attn_out, _ = self.mha(x_t, x_t, x_t, need_weights=False)
        x_t = self.norm(x_t + self.dropout(attn_out))
        return x_t.permute(1, 2, 0)               # [B, E, T]


# ==============================
# 4) Modèle complet — EEGFormerHybridV2
# ==============================
class EEGFormerHybridV2(nn.Module):
    """
    Pipeline :
      (1) ChannelGate  -> attention spatiale explicable (poids par électrode)
      (2) TemporalCNN  -> encodeur temporel robuste
      (3) TinyTemporalAttention -> capte dépendances T
      (4) Classif MLP

    Paramètres :
      - num_channels : nombre d'électrodes (62, 48, 32, 16, 8…)
      - seq_len      : longueur temporelle (ex: 200)
      - num_classes  : nombre de classes cibles

    """

    def __init__(
        self,
        num_channels: int = 62,
        seq_len: int = 200,
        num_classes: int = 3,
        embed_dim: int = 128,
        gate_hidden: int = 64,
        pdrop_cnn: float = 0.30,
        pdrop_head: float = 0.15,
        heads: int = 4,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # 1) Attention spatiale explicable
        self.gate = ChannelGate(num_channels=num_channels, hidden=gate_hidden)

        # 2) CNN temporel
        self.temporal = TemporalCNN(in_ch=num_channels, embed_dim=embed_dim, pdrop=pdrop_cnn)

        # 3) MHA temporelle
        self.tiny_attn = TinyTemporalAttention(embed_dim=embed_dim, num_heads=heads, pdrop=pdrop_head)

        # 4) Normalisation + Classif
        self.norm = nn.LayerNorm([embed_dim, seq_len // 2])
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * (seq_len // 2), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

        # init MLP final
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, C, T]  avec C = num_channels, T = seq_len
        x = self.gate(x)           # attention spatiale explicable
        x = self.temporal(x)       # [B, E, T/2]
        x = self.tiny_attn(x)      # [B, E, T/2]
        x = self.norm(x)           # LayerNorm 2D
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits              # [B, num_classes]

    # ---------- utilitaires XAI ----------
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Retourne les poids d'attention moyens sur le batch si disponibles.
        - Si on vient de faire un forward, self.gate.last_channel_weights = [B, C] (CPU)
        - Ici on renvoie la moyenne batch -> [C] (pour tri/visualisation)
        """
        if self.gate.last_channel_weights is None:
            return None
        return self.gate.last_channel_weights.mean(dim=0)  # [C] (CPU)

    def get_attention_weights_per_sample(self) -> Optional[torch.Tensor]:
        """
        Retourne les poids par échantillon -> [B, C] (CPU) si disponibles.
        Utile pour corréler importance canaux <-> erreurs par exemple.
        """
        return self.gate.last_channel_weights

    @torch.no_grad()
    def rank_channels(
        self,
        electrode_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Renvoie la liste triée (desc) des canaux par importance,
        avec noms si fournis (sinon index).
        Nécessite un forward récent pour disposer des poids.
        """
        w = self.get_attention_weights()
        if w is None:
            return []
        w = w.numpy().tolist()
        if electrode_names is None:
            return [(str(i), w[i]) for i in range(len(w))]
        return [(electrode_names[i], w[i]) for i in range(len(w))]

    # ---------- fabrique ----------
    @staticmethod
    def from_data_shape(
        x_shape: Tuple[int, int, int],
        num_classes: int,
        embed_dim: int = 128,
        **kwargs
    ) -> "EEGFormerHybridV2":
        """
        Permet d’instancier rapidement le modèle à partir de X.shape :
          - x_shape = (B, C, T) -> utilise C et T
        """
        _, C, T = x_shape
        return EEGFormerHybridV2(
            num_channels=C,
            seq_len=T,
            num_classes=num_classes,
            embed_dim=embed_dim,
            **kwargs
        )
