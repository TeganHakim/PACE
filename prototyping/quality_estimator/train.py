"""
load saved feature maps + corresponding ground-truth quality .npy maps, 
train the model from model.py, validate it, and save the best weights to checkpoints/.
"""

"""

TODO once encoder features have been extracted:

1. Import SpatialQualityEstimator from model.py.

2. Create a PyTorch Dataset that loads:
       saved encoder feature tensor
       corresponding ground-truth 8x8 quality map

3. Create train and validation DataLoaders.

4. Determine encoder feature channel count C.

5. Initialize:
       model = SpatialQualityEstimator(in_channels=C)

6. Define regression loss.
   Initial choice: SmoothL1Loss or MSELoss.

7. Define optimizer.
   Initial choice: Adam.

8. Training loop:
       for each epoch:
           - load feature maps
           - predict 8x8 quality maps
           - calculate loss against GT maps
           - backpropagate
           - update estimator weights

9. Validation loop:
       - model.eval()
       - no gradients
       - calculate validation loss
       - calculate MAE / RMSE

10. Save best validation checkpoint:
        checkpoints/
            quality_rgb.pt
            quality_ir.pt

11. Evaluate quality predictions, especially performance on
    degraded patches vs clean patches.

12. Later:
    - threshold predicted quality/reliability maps
    - feed them into the reliability gate / patch router
"""