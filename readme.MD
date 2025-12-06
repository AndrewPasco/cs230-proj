**CS230 Final Project - Andrew Pasco and Jorge Garcia**

Machine learning for vision-based tactile sensing.

Current approach: 

Implemented: 
- MiniUNet for segmentation
- Detection network (transfer learned from MobileNetV3_Small)
    - Original output layer replaced with binary classification head
        - 256-unit FC layer (dropout = 0.2) to single output with Sigmoid activation
- PCA-based direction prediction + image annotation
- Inference pipeline with models in sequence
    - ~10fps execution rate
    - Currently creates annotated video as example of how live inference could look
- Alternative approach with 3D reconstruction based on surface normal prediction 
    - Can add this as final chain in inference (use segmentation mask to identify region for reconstruction)