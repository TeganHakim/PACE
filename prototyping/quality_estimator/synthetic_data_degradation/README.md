### Synthetic Data Degradation for Training PACE's Quality & Reliability Estimation Models

The DroneVehicle dataset in its original form does not provide sufficient training data with poor quality for both modalities. 
The following degradations will be performed through random sampling on imagery:

- *blur*: convolution with Gaussian kernels
- *noise*: add sampled Gaussian noise to pixel values
- *obstruction*: generate irregular masks/blotches and replace pixels beneath
- *low light*: reduce intensity nonlinear transforms instead of simply multiplying everything

It's key to avoid matching these degradations to the 8x8 grid itself. True sensor degradations will not follow such patterns in operational environments. 

For each degradation environment: 
- Randomly choose a center \((x,y)\) anywhere in the image.
- Randomly choose a size, e.g. radius \(r\), width/height, etc.
- Randomly choose a severity \(s\).
- Apply the degradation to that spatial region.
- Afterward, overlay the 8×8 grid to determine which feature patches were affected and by how much --> this step is for the training
data, so we have the target telling the estimator what the quality score should be for each of the 8x8 patches. 