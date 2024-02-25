# 3D Pose Extimation

![3DPose](../assets/output_wild.gif)

As we utilize 2D-to-3D lifting as the pretext task, we simply reuse the whole pretrained network. During finetuning, the input 2D skeletons are estimated from videos without extra masks or noises.

The 2D skeletons are provided by 2D
pose estimator trained on MPII and Human3.6M. For training from
scratch, we train for 60 epochs with learning rate 0.0002 and
batch size 32. For finetuning, we load the pretrained weights
and train for 30 epochs
