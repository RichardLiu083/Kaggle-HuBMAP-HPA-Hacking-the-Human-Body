# Kaggle-HuBMAP-HPA-Hacking-the-Human-Body
2022/09/23 - silver medal - Top 4%
![image](https://github.com/RichardLiu083/Kaggle-HuBMAP-HPA-Hacking-the-Human-Body/blob/main/ranking.png)

## Solution

## Insights
- Split the main task into two subtask (predict lung type and the others), which makes the model more stable while training and validation. 
  Probably because lung type have weak signal and also noisy label.
- CNN model is better than transformer model when predicting lung type.
- Transformer model is better than CNN model when predicting the other type.
- Stain Normalization can reduce the difference between HPA and Hubmap data.
- Scale adjustment is crucial since HAP data have different pixel size compare to Hubmap data.
- Doing Mosaic、Cutmix augmentation on lung type data will lead to performence drop.
- Multi-classes model is worse than single class model.

## Models
- EfficientNet_b7 * 2 (Unet decoder)
- Coat_medium (Daformer decoder)
- Segformer_b5 * 2 (different image size)
- Beit_base

## Augmentation
- Stain Normalization (helps a lot)
- Mosaic
- Cutmix
- H、V Flip
- HueSaturationValue
- ShiftScaleRotate
- CoarseDropout
- Blur
- GaussNoise

## Training
- 100 epochs
- lr 3e-4 for CNN, 6e-5 for transformer
- bce + dice loss

## Validation
- 5 fold 
- only choose best model from validation

## Inference
- TTA * 8
- choosing lower threshold for Hubmap data
- 2 model(CNN) for lung type prediction, 4 model(Transformer) for the others. (total 6 model)
