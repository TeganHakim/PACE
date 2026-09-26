### PACE's Quality & Reliability Estimator

The core of PACE's approach to adaptive fusion is driven by a quality & reliability score map
produced by an estimator model over an IR/RGB input. The estimator model will, for each 8x8 patch
in the input, compute from the encoded feature vector, a quality score. The input to the model is the 
feature vector representative of the 8px by 8px patch produced by the baseline CrossFuse encoders, 
trained on the DroneVehicle dataset. 

Data Requirements: supervised learning will be used for the estimator models, so for both modalities,
imagery with patches of poor quality (and labeled as such) are required. To achieve this, synthetic data degradation
is a necessity as the DroneVehicle dataset contains primarily iamgery with high sensor quality.  

Training Procedure: to be completed. 