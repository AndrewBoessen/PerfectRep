# Finetune Action Classifier

Action classifier for powerlifting. Classify clip as squat, deadlift, or bench press.

## Pretrain Backbone

The action classifier head uses DSTFormer as a backbone for getting motion represenation. The pretrain backbone can be trained from scratch. See [Training Instruction](../README.md#training-instructions). Or download pretrain backbone parameters [here](https://drive.google.com/file/d/1Al49MhmvG3IG2ASWcb6Mx8mymArmb7Wz/view?usp=drive_link).

## Preprocess Data

1. Process Fit3D dataset. See [Data Preprocessing](../README.md#data-preprocessing)

2. Slice data into clips (len=243, stride=81)

```
python process_fit3d_action.py
```

3. Verify data files in `data/action/Fit3D`

## Train

Train the classifier using provided training script

```
python train_action.py \
--data_path /path/to/dataset \
--checkpoint /checkpoint/dir \
--pretrained /pretrain/model/dir \
--selection latest_epoch.bin \
--epochs 100 \
--batch_size 32
```
