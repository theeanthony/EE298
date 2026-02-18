#!/usr/bin/env python3
"""
tcn_model.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Temporal Convolutional Network (TCN) for fungal bioelectrical signal classification.
Architecture: causal dilated convolutions with exponential dilation growth.
~15K parameters — trains on CPU in minutes, Colab T4 in seconds.

Input: (batch, 1, 600) — 60-second windows at 10 Hz
Output: (batch, n_classes) logits
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """
    Single TCN block: two causal dilated convolutions with residual connection.

    Causal padding ensures the output at time t only depends on inputs at times <= t.
    Like reading a signal left-to-right — the network can't peek into the future.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left side only
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection: 1x1 conv if channel dims differ, else identity
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # First causal conv: pad left, conv, chop right
        out = nn.functional.pad(x, (self.causal_pad, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second causal conv
        out = nn.functional.pad(out, (self.causal_pad, 0))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        res = self.residual(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.

    4 blocks with dilation [1, 2, 4, 8] and kernel=7 gives:
    Receptive field = 2 * (7-1) * (1+2+4+8) = 180 samples = 18 seconds at 10 Hz.
    That's enough to capture the slow fungal signal oscillations (0.01-1 Hz).
    """

    def __init__(self, in_channels=1, hidden=32, kernel_size=7, num_blocks=4, dropout=0.2):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden
            blocks.append(TemporalBlock(in_ch, hidden, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*blocks)
        self.hidden = hidden

    def forward(self, x):
        return self.network(x)


class FungalSignalTCN(nn.Module):
    """
    Complete TCN classifier: encoder + global average pooling + classification head.

    Supports 3-phase transfer learning:
      Phase 1: Pre-train all parameters on ECG (5-class)
      Phase 2: freeze_early_blocks(2), replace_head(2) for plant/fungal adaptation
      Phase 3: freeze_encoder(), train head only on target fungal data
    """

    def __init__(self, encoder, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.hidden, encoder.hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(encoder.hidden, n_classes),
        )

    def forward(self, x):
        # x shape: (batch, 1, 600)
        features = self.encoder(x)          # (batch, hidden, 600)
        pooled = features.mean(dim=2)       # (batch, hidden) — global avg pool
        return self.head(pooled)            # (batch, n_classes)

    def freeze_encoder(self):
        """Freeze all encoder parameters (Phase 3: head-only training)."""
        self.encoder.requires_grad_(False)

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        self.encoder.requires_grad_(True)

    def freeze_early_blocks(self, n=2):
        """Freeze the first n encoder blocks (Phase 2: adapt later blocks + head)."""
        for i, block in enumerate(self.encoder.network):
            if i < n:
                block.requires_grad_(False)
            else:
                block.requires_grad_(True)

    def replace_head(self, new_n_classes):
        """Swap classification head for a different number of classes."""
        hidden = self.encoder.hidden
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, new_n_classes),
        )

    def get_encoder_state_dict(self):
        """Get encoder weights for checkpoint transfer."""
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, state_dict):
        """Load encoder weights from a checkpoint."""
        self.encoder.load_state_dict(state_dict)


def build_tcn(n_classes=2, hidden=32, kernel_size=7, num_blocks=5, dropout=0.2):
    """
    Convenience factory: build a FungalSignalTCN with default architecture.

    Default is 5 blocks (dilations [1,2,4,8,16]):
      RF = 2 * (7-1) * (1+2+4+8+16) = 372 samples = 37 seconds at 10 Hz.
    This gives more temporal context than the 4-block (18 sec) baseline,
    better capturing long-duration word types (Adamatzky words span minutes).
    """
    encoder = TCNEncoder(
        in_channels=1, hidden=hidden, kernel_size=kernel_size,
        num_blocks=num_blocks, dropout=dropout,
    )
    return FungalSignalTCN(encoder, n_classes=n_classes)


# ============================================================
# TCN WITH SELF-ATTENTION POOLING (Tier 2 enhancement)
# ============================================================

class TCNWithAttention(FungalSignalTCN):
    """
    TCN with self-attention pooling instead of global average pooling.

    Learns which time steps to weight for classification, producing
    interpretable attention maps showing which spike events drove predictions.

    For a 60-second window at 10 Hz (600 samples), the model can focus
    attention on the spike bursts most diagnostic of each stimulus.

    Inherits freeze_encoder(), freeze_early_blocks(), replace_head(),
    get_encoder_state_dict(), load_encoder_state_dict() from FungalSignalTCN.
    """

    def __init__(self, encoder, n_classes=2, num_heads=4):
        super().__init__(encoder, n_classes)
        hidden = encoder.hidden
        # num_heads must divide hidden; with hidden=32, use num_heads=4
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden)

    def forward(self, x, return_attention=False):
        # x: (batch, 1, 600)
        features = self.encoder(x)           # (batch, hidden, 600)
        enc = features.permute(0, 2, 1)      # (batch, 600, hidden)
        attn_out, attn_weights = self.attention(enc, enc, enc)
        attn_out = self.attn_norm(attn_out + enc)  # residual + norm
        pooled = attn_out.mean(dim=1)        # (batch, hidden) — attended avg
        logits = self.head(pooled)
        if return_attention:
            return logits, attn_weights      # attn_weights: (batch, 600, 600)
        return logits


def build_tcn_with_attention(n_classes=2, hidden=32, kernel_size=7,
                              num_blocks=5, dropout=0.2, num_heads=4):
    """Build a TCNWithAttention model — interpretable attention-pooled TCN."""
    encoder = TCNEncoder(
        in_channels=1, hidden=hidden, kernel_size=kernel_size,
        num_blocks=num_blocks, dropout=dropout,
    )
    return TCNWithAttention(encoder, n_classes=n_classes, num_heads=num_heads)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Quick sanity check — standard TCN (5-block default)
    model = build_tcn(n_classes=5)
    total, trainable = count_parameters(model)
    print(f"TCN (5-block): {total:,} total params, {trainable:,} trainable")

    x = torch.randn(4, 1, 600)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")
    assert logits.shape == (4, 5), f"Expected (4, 5), got {logits.shape}"

    # Test freeze/replace
    model.freeze_early_blocks(2)
    _, trainable_partial = count_parameters(model)
    print(f"After freezing 2 blocks: {trainable_partial:,} trainable")

    model.replace_head(2)
    model.freeze_encoder()
    _, trainable_head = count_parameters(model)
    print(f"After freezing encoder: {trainable_head:,} trainable (head only)")

    # Sanity check — attention TCN
    attn_model = build_tcn_with_attention(n_classes=5)
    total_attn, _ = count_parameters(attn_model)
    print(f"\nTCN+Attention (5-block): {total_attn:,} total params")
    logits_attn = attn_model(x)
    assert logits_attn.shape == (4, 5), f"Expected (4, 5), got {logits_attn.shape}"
    logits_attn2, weights = attn_model(x, return_attention=True)
    print(f"Attention weights shape: {weights.shape}")  # (4, 600, 600)

    print("All checks passed.")
