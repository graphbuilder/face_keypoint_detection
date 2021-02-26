# Face Keypoint Detection
Research & Slide:


NOTE: the most complete detector.py is in stage 3

## Stage 1
Simply runs SGD for 50 epochs
When running predicting phase, we can see the predicted points are almost the same for all inputs, which means it probably stuck at a local local minimal.

## Stage 2
Now change SGD to ADAM.

I also tried 
  1) flipping the images by 90, 180 and 270 degrees; 
  2) rotating them at random angles. 
In the rotaiton, I have tried carefully to implement the scaling factor, so after rotating, all points are still within limits.

NOTE: to see the effect of rotation, run python data.py; need to manually change the rotation_class in the main part for now.

BN layers are also implemented.

I ran for 100 epochs. 

One can compare the result in the 3 pdfs provided (no_rotation, with_all_angle, with_flipping). These are computed using the compare.py. We can see without BN and rotation, it will stuck for a long time. With BN, the loss for all 3 options decrease dramatically. It is somewhat strange that the augmentation does not improve the final fit, probably because only few validation images are rotated.

## Stage 3
I compare two options: one with both loss (pts and classes) added together, one uses the output from stage2 and only finetune the classification branch.  I only ran for 20 epochs.

One can compare the result in the 2 pdfs provided. I think I overfit the classification problem, but since the accuracy is over 90% already, I didn't implement the dropout layers.

