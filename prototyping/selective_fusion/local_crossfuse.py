
"""
PACE: Local Feature-Space CrossFuse.

Runs pretrained CrossFuse attention and decoding only on patches
selected by the reliability gate.

Supports precomputed encoder features so that the complete PACE
pipeline can share encoder computation between quality estimation
and selective fusion.

Inputs:
    rgb, ir:        [B, 1, H, W], normalized to [0, 1]
    rgb_quality:     [B, 1, 8, 8]
    ir_quality:      [B, 1, 8, 8]
    gate:            [B, 1, 8, 8], binary
    encoded_features:
        Optional tuple (rgb_sh, rgb_de, ir_sh, ir_de)

Outputs:
    out:             [B, 1, H, W]
    gate:            [B, 1, 8, 8]
    prefer_rgb:      [B, 1, 8, 8]
    fused_count:     Number of selected patches
    fusion_rate:     Fraction of patches selected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCrossFuse(nn.Module):
    def __init__(
        self,
        rgb_encoder,
        ir_encoder,
        baseline,
        grid_size=8,
        context_features=8,
        batch_size=16,
    ):
        super().__init__()

        self.rgb_encoder = rgb_encoder
        self.ir_encoder = ir_encoder
        self.baseline = baseline

        self.grid_size = grid_size
        self.context_features = context_features
        self.batch_size = batch_size

        if context_features < 4 or context_features % 2 != 0:
            raise ValueError(
                "context_features must be even and >= 4"
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        self.eval()

    @staticmethod
    def crop_padded(x, y, x0, size):
        """
        Extract a square feature crop.

        Regions outside the original feature map are filled
        using replication padding.
        """
        h, w = x.shape[-2:]

        top = max(0, -y)
        left = max(0, -x0)
        bottom = max(0, y + size - h)
        right = max(0, x0 + size - w)

        crop = x[
            ...,
            max(0, y):min(h, y + size),
            max(0, x0):min(w, x0 + size),
        ]

        return F.pad(
            crop,
            (left, right, top, bottom),
            mode="replicate",
        )

    def attention(self, ir_de, rgb_de):
        """
        Apply pretrained CrossFuse attention locally.

        Inputs:
            ir_de:  [N, 128, S, S]
            rgb_de: [N, 128, S, S]

        Output:
            fused:  [N, 128, S, S]

        Shifted attention is disabled for small local windows.
        """
        fused, *_ = self.baseline.cross_atten_block(
            ir_de,
            rgb_de,
            False,
        )

        return fused

    @staticmethod
    def normalize_individual(x):
        """
        Independently normalize each decoded context crop.

        This preserves the normalization behavior of the
        existing local fusion prototype.
        """
        low = x.amin(
            dim=(1, 2, 3),
            keepdim=True,
        )

        high = x.amax(
            dim=(1, 2, 3),
            keepdim=True,
        )

        return (
            (x - low) /
            (high - low).clamp_min(1e-6)
        )

    @torch.no_grad()
    def forward(
        self,
        rgb,
        ir,
        rgb_quality,
        ir_quality,
        gate,
        encoded_features=None,
    ):
        """
        Perform selective local fusion.

        If encoded_features is supplied, the pretrained
        encoders are NOT executed again.
        """

        # --------------------------------------------------
        # Validate image dimensions
        # --------------------------------------------------

        if rgb.shape != ir.shape or rgb.ndim != 4:
            raise ValueError(
                "RGB and IR must have matching [B,1,H,W] shapes"
            )

        b, c, h, w = rgb.shape
        g = self.grid_size

        if c != 1:
            raise ValueError(
                "Expected single-channel grayscale images"
            )

        if h % g != 0 or w % g != 0:
            raise ValueError(
                "Image dimensions must be divisible by grid_size"
            )

        expected = (b, 1, g, g)

        for name, tensor in (
            ("rgb_quality", rgb_quality),
            ("ir_quality", ir_quality),
            ("gate", gate),
        ):
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}"
                )

        if not torch.isfinite(gate).all():
            raise ValueError(
                "Gate contains non-finite values"
            )

        if not torch.all((gate == 0) | (gate == 1)):
            raise ValueError(
                "Gate must contain only 0 and 1"
            )

        ph = h // g
        pw = w // g

        # --------------------------------------------------
        # Initialize output using the preferred modality
        # --------------------------------------------------

        prefer_rgb = rgb_quality >= ir_quality

        preferred_pixels = F.interpolate(
            prefer_rgb.float(),
            size=(h, w),
            mode="nearest",
        ).bool()

        output = torch.where(
            preferred_pixels,
            rgb,
            ir,
        ).clone()

        # --------------------------------------------------
        # Identify patches selected for fusion
        # --------------------------------------------------

        selected = torch.nonzero(
            gate[:, 0] > 0.5,
            as_tuple=False,
        )

        fused_count = selected.shape[0]
        total_patches = b * g * g

        # If the gate selects nothing, skip all fusion work.
        if fused_count == 0:
            return {
                "out": output,
                "gate": gate,
                "prefer_rgb": prefer_rgb,
                "fused_count": 0,
                "fusion_rate": 0.0,
            }

        # --------------------------------------------------
        # Obtain encoder features
        # --------------------------------------------------

        if encoded_features is None:

            # Standalone mode: encode images once.
            rgb_sh, rgb_de = self.rgb_encoder(
                rgb * 255.0
            )

            ir_sh, ir_de = self.ir_encoder(
                ir * 255.0
            )

        else:

            # Integrated PACE mode:
            # Reuse features already computed for quality
            # estimation and reliability gating.
            if len(encoded_features) != 4:
                raise ValueError(
                    "encoded_features must contain "
                    "(rgb_sh, rgb_de, ir_sh, ir_de)"
                )

            (
                rgb_sh,
                rgb_de,
                ir_sh,
                ir_de,
            ) = encoded_features

        # --------------------------------------------------
        # Determine spatial feature dimensions
        # --------------------------------------------------

        dh, dw = rgb_de.shape[-2:]

        if dh % g != 0 or dw % g != 0:
            raise ValueError(
                "Deep feature dimensions must be divisible "
                "by grid_size"
            )

        fh = dh // g
        fw = dw // g

        if fh != fw:
            raise ValueError(
                "Expected square deep feature patches"
            )

        s = self.context_features

        if s < fh:
            raise ValueError(
                "Context is smaller than target feature patch"
            )

        if (s - fh) % 2 != 0:
            raise ValueError(
                "Context and target patch must be centered"
            )

        if h % dh != 0 or w % dw != 0:
            raise ValueError(
                "Image and deep feature dimensions are not "
                "integer multiples"
            )

        scale_h = h // dh
        scale_w = w // dw

        if scale_h != scale_w:
            raise ValueError(
                "Expected uniform encoder downsampling"
            )

        scale = scale_h

        margin = (s - fh) // 2
        shallow_size = s * scale

        # --------------------------------------------------
        # Build selected patch jobs
        # --------------------------------------------------

        jobs = []

        for bi, row, col in selected.tolist():

            dy = row * fh - margin
            dx = col * fw - margin

            jobs.append({
                "bi": bi,
                "row": row,
                "col": col,
                "dy": dy,
                "dx": dx,
            })

        # --------------------------------------------------
        # Process selected patches in batches
        # --------------------------------------------------

        for start in range(
            0,
            len(jobs),
            self.batch_size,
        ):

            batch_jobs = jobs[
                start:start + self.batch_size
            ]

            ir_de_crops = []
            rgb_de_crops = []
            ir_sh_crops = []
            rgb_sh_crops = []

            for job in batch_jobs:

                bi = job["bi"]
                dy = job["dy"]
                dx = job["dx"]

                # Deep features: local S x S windows
                ir_de_crops.append(
                    self.crop_padded(
                        ir_de[bi:bi + 1],
                        dy,
                        dx,
                        s,
                    )
                )

                rgb_de_crops.append(
                    self.crop_padded(
                        rgb_de[bi:bi + 1],
                        dy,
                        dx,
                        s,
                    )
                )

                # Shallow features: corresponding image region
                sy = dy * scale
                sx = dx * scale

                ir_sh_crops.append(
                    self.crop_padded(
                        ir_sh[bi:bi + 1],
                        sy,
                        sx,
                        shallow_size,
                    )
                )

                rgb_sh_crops.append(
                    self.crop_padded(
                        rgb_sh[bi:bi + 1],
                        sy,
                        sx,
                        shallow_size,
                    )
                )

            # Batch all selected feature crops.
            ir_de_batch = torch.cat(
                ir_de_crops,
                dim=0,
            )

            rgb_de_batch = torch.cat(
                rgb_de_crops,
                dim=0,
            )

            ir_sh_batch = torch.cat(
                ir_sh_crops,
                dim=0,
            )

            rgb_sh_batch = torch.cat(
                rgb_sh_crops,
                dim=0,
            )

            # ----------------------------------------------
            # Local pretrained cross-attention
            # ----------------------------------------------

            fused_features = self.attention(
                ir_de_batch,
                rgb_de_batch,
            )

            # ----------------------------------------------
            # Local pretrained decoder
            # ----------------------------------------------

            raw = self.baseline.decoder_fusion(
                ir_sh_batch,
                rgb_sh_batch,
                ir_de_batch,
                rgb_de_batch,
                fused_features,
            )

            if raw.shape[-2:] != (
                shallow_size,
                shallow_size,
            ):
                raise RuntimeError(
                    f"Decoder returned {raw.shape[-2:]}, "
                    f"expected "
                    f"{(shallow_size, shallow_size)}"
                )

            # ----------------------------------------------
            # Normalize and retain central patches
            # ----------------------------------------------

            offset = margin * scale

            # Normalize over each decoded context region,
            # then retain the central target patch.
            normalized = self.normalize_individual(raw)

            for i, job in enumerate(batch_jobs):

                bi = job["bi"]
                y = job["row"] * ph
                x = job["col"] * pw

                fused_patch = normalized[
                    i:i + 1,
                    :,
                    offset:offset + ph,
                    offset:offset + pw,
                ]

                # Replace only the selected 32x32 patch.
                output[
                    bi:bi + 1,
                    :,
                    y:y + ph,
                    x:x + pw,
                ] = fused_patch

        # --------------------------------------------------
        # Return reconstructed image and routing statistics
        # --------------------------------------------------

        return {
            "out": output.clamp(0.0, 1.0),
            "gate": gate,
            "prefer_rgb": prefer_rgb,
            "fused_count": fused_count,
            "fusion_rate": fused_count / total_patches,
        }
