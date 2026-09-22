# Progress Notes
A document to centralize design decisions, to-do's, and relevant notes we come up with through the research process. 

Note on encoders:
Because our quality estimator model 

Next steps:
for baseline:
    - gpu access --> train feature encoders on dronevehicle dataset
    - gpu access --> train transfuse CNN for fusion on dronevehicle dataset
    - gpu access --> train the detection head on the outputted 
    - *we need to set up their architecture, trained on OUR dataset* 
for our model:
    - use the same new feature encoders and CNN weights, but the detection head will need to be adjusted
      and trained based on the new fusion outputs specific to our approach
