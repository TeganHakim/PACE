"""
load the frozen trained CrossFuse encoder, run the RGB/IR images through it, 
and save the relevant encoder feature maps to features/rgb/ and features/ir/. 
This is run once after encoder training.
"""

"""

TODO once encoder training is complete:

1. Load the trained CrossFuse encoder checkpoint.

2. Set encoder to evaluation mode and freeze parameters.

3. Load images from processed_dataset.
   - RGB image
   - corresponding IR image
   - preserve train / val / test split

4. Preprocess images exactly as expected by the CrossFuse encoder.

5. Pass each image through the encoder:
       shallow_features, deep_features = encoder(image)

6. Determine which encoder representation should be used by the
   quality estimator (expected: deep features).

7. Verify and record feature dimensions:
       [C, H, W]

8. Save features:
       features/
           rgb/
               train/
               val/
               test/
           ir/
               train/
               val/
               test/

9. Use matching sample IDs so features can later be paired with
   their ground-truth 8x8 quality maps.

10. Verify:
    - every feature tensor has a corresponding quality map
    - RGB/IR sample IDs remain aligned
    - no train/val/test leakage
"""