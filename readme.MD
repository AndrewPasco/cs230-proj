**CS230 Final Project - Andrew Pasco and Jorge Garcia**

Machine learning for vision-based tactile sensing.

Current approach: 

Implemented: 
- MiniUNet for segmentation
- Detection network (transfer learned from ImageNet1K-V2)
    - Layer 3 and 4 unfrozen
    - Original output layer replaced with binary classification head
        - 256-unit FC layer (dropout = 0.4) to single output with Sigmoid activation
- PCA-based direction prediction + image annotation

TODO:

- Object direction prediction
    - Learning-based direction prediction

- Integration of modularized pipeline for live inference on video

- Alternative approach with 3D reconstruction based on surface normal prediction    
    - Could use similar direction prediction approach given 3D mesh rather than segmented 2D contour