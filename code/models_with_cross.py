import torch
import torch.nn as nn
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
        # Pour XAI : on garde les derniers poids 
        self.last_channel_weights: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, C, T]
        desc = x.abs().mean(dim=-1)          
        logits = self.mlp(desc)               
        weights = torch.softmax(logits, dim=-1)  
        self.last_channel_weights = weights.detach().cpu()
        return x * weights.unsqueeze(-1)      


# ==============================
# 2) Encodeur temporel (CNN résiduel)
# ==============================
class TemporalCNN(nn.Module):
    """
    Entrée :  [B, C, T]
    Sortie :  [B, E, T/2]
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
      
        x = self.stem(x)              
        x = x + self.block1(x)       
        x = self.pool1(x)             
        x = x + self.block2(x)       
        x = x + self.block3(x)       

        # LayerNorm sur E
        x = self.final_norm(x.transpose(1, 2)).transpose(1, 2)
        return x                    


# ==============================
# 3) Self-Attention temporelle (TinyTemporalAttention)
# ==============================
class TinyTemporalAttention(nn.Module):
    """
    Multi-Head Self-Attention sur l'axe temporel.
    Entrée : [B, E, T] -> [T, B, E] pour nn.MultiheadAttention
    Sortie : [B, E, T]
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, pdrop: float = 0.15):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=pdrop, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(pdrop * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, E, T]
        x_t = x.permute(2, 0, 1)          
        attn_out, _ = self.mha(x_t, x_t, x_t, need_weights=False)
        x_t = self.norm(x_t + self.dropout(attn_out))
        return x_t.permute(1, 2, 0)      


# ==============================
# 4) Cross-Attention temporelle
# ==============================
class CrossAttention(nn.Module):
    """
    Cross-Attention entre deux représentations temporelles.
    Ici, pour une première version, on fait Q = X_self, KV = X_self
    (architecture Self -> Cross, même source, mais bloc séparé).
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, pdrop: float = 0.15):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=pdrop, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
       
        q = x_q.permute(2, 0, 1)     
        kv = x_kv.permute(2, 0, 1)    

        out, _ = self.mha(q, kv, kv, need_weights=False)
        out = self.norm(q + self.dropout(out))
        return out.permute(1, 2, 0)  


# ==============================
# 5) Modèle complet — EEGFormerHybridV3 (Self + Cross)
# ==============================
class EEGFormerHybridV3(nn.Module):
    """
    Pipeline :
      (1) ChannelGate           -> attention spatiale (poids par électrode)
      (2) TemporalCNN           -> encodeur temporel
      (3) TinyTemporalAttention -> Self-Attention temporelle
      (4) CrossAttention        -> Cross-Attention temporelle (Self -> Cross)
      (5) LayerNorm + MLP       -> classification

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

        # 1) Attention spatiale explicable (canaux)
        self.gate = ChannelGate(num_channels=num_channels, hidden=gate_hidden)

        # 2) CNN temporel
        self.temporal = TemporalCNN(in_ch=num_channels, embed_dim=embed_dim, pdrop=pdrop_cnn)

        # 3) Self-Attention temporelle
        self.self_attn = TinyTemporalAttention(embed_dim=embed_dim, num_heads=heads, pdrop=pdrop_head)

        # 4) Cross-Attention temporelle (Self -> Cross)
        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=heads, pdrop=pdrop_head)

        # 5) Normalisation + Classif 
        self.norm = nn.LayerNorm([embed_dim, seq_len // 2])
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * (seq_len // 2), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes)
        )

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, C, T]
        x = self.gate(x)         
        x = self.temporal(x)      

        # Self-Attention temporelle
        x_self = self.self_attn(x)    

        # Cross-Attention : ici Q = self, KV = self (architecture Self -> Cross)
        x_cross = self.cross_attn(x_self, x_self) 

        # Normalisation + classif
        x_out = self.norm(x_cross)
        x_out = self.dropout(x_out)
        logits = self.classifier(x_out)
        return logits   # [B, num_classes]

    # ---------- utilitaires XAI (mêmes idées que V2) ----------
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        if self.gate.last_channel_weights is None:
            return None
        return self.gate.last_channel_weights.mean(dim=0)  # [C]

    def get_attention_weights_per_sample(self) -> Optional[torch.Tensor]:
        return self.gate.last_channel_weights

    @torch.no_grad()
    def rank_channels(
        self,
        electrode_names: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
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
    ) -> "EEGFormerHybridV3":
        _, C, T = x_shape
        return EEGFormerHybridV3(
            num_channels=C,
            seq_len=T,
            num_classes=num_classes,
            embed_dim=embed_dim,
            **kwargs
        )
