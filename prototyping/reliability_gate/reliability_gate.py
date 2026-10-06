"""
PACE: Patch-Adaptive Cross-Attention for Efficient Multimodal Sensor Fusion.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Reliability Gate
# ============================================================================

class ReliabilityGate(nn.Module):
    """
    Predict an 8x8 binary reliability gate from RGB and IR quality maps.

    Input:
        rgb_quality: [B, 1, 8, 8]
        ir_quality:  [B, 1, 8, 8]

    Output:
        logits:    [B, 1, 8, 8]
        soft_gate: [B, 1, 8, 8]
        hard_gate: [B, 1, 8, 8]
        gate:      [B, 1, 8, 8]

    `gate` uses a straight-through estimator. Forward behavior is hard
    binary routing, while gradients behave like the sigmoid output.
    """

    def __init__(self, hidden_dim=16):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # Start near a 50/50 decision.
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, rgb_quality, ir_quality):
        if rgb_quality.shape[-2:] != (8, 8):
            raise ValueError(
                "rgb_quality must have spatial size 8x8."
            )

        if ir_quality.shape[-2:] != (8, 8):
            raise ValueError(
                "ir_quality must have spatial size 8x8."
            )

        quality = torch.cat(
            [rgb_quality, ir_quality],
            dim=1,
        )

        """
        This uses a trick.
        Given RGB and IR patch-quality maps, the reliability gate predicts a continuous activation probability 
        using a lightweight convolutional network. This probability is thresholded at 0.5 to produce a binary routing 
        decision for each spatial patch. To enable training with backpropagation despite this non-differentiable binary 
        operation, a straight-through estimator is used. Specifically, `detach()` prevents gradients from propagating 
        through the hard binary gate and the detached copies of the soft prediction, while the original sigmoid activation 
        remains connected to the computational graph. As a result, the forward pass uses the binary gate while the backward 
        pass uses the gradient of the underlying sigmoid activation to update the gate. A gate value of 1 activates 
        cross-attention, whereas a value of 0 bypasses cross-attention.
        """

        logits = self.network(quality)

        soft_gate = torch.sigmoid(logits)

        hard_gate = (soft_gate > 0.5).float()

        # This is the trick: equates to just hard_gate
        gate = hard_gate.detach() - soft_gate.detach() + soft_gate

        return gate