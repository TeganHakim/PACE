# Progress Notes
A document to centralize design decisions, to-do's, and relevant notes we come up with through the research process. 

Note on encoders:
Because our quality estimator model 

Next steps:
for baseline:
- gpu access --> train feature encoders on dronevehicle dataset
- gpu access --> train transfuse on dronevehicle dataset
- gpu access --> train the detection head on the outputted 
- *we need to set up their architecture, trained on OUR dataset* 
- *in summary*: 1. train encoders independently, and freeze 2. train transfuse, and freeze 3. train the detection head on the fused feature vectors
for our model:
- use the same new feature encoders and CNN weights, but the detection head will need to be adjusted and trained based on the new fusion outputs specific to our approach
- quality estimators will be trained independent of detection outputs
- however, the loss function for the learned reliability gate will involve the results of the detection head --> after each forward pass, leverage the detection output to optimize the weights of the MLP
- *key detail* --> on the initial pass of the reliability gate training, we have to use the decoder-based reconstruction error. then, we'll train the detection head based on those outputs. finally, we'll optimize the reliability gate based on the detection results. 
- *in summary*: 1. use the pretrained encoders & transfuse components 2. train the quality estimators 3. train the learned reliability gate via decoder 4. train detection head 5. optimize learned gate based on detection outputs. 
