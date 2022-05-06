# BlindNet: Scene Based Knowledge Distillation

## UNET: Pixel level prediction for each bounding box
Dataset structure:

-datasets/coco/annotations/

-datasets/coco/val2017/

-datasets/coco/train2017/

```
python src/train_unet.py
```

## Simple CNN: predict only the masked object
```
python src/train_single_maskcnn.py
```


## EXPERIMENT 3 (Pixel wise class id prediction)

First download dataset from [here](https://www.kaggle.com/datasets/alphadraco/coco-cat-id-masked-images).

to run the experiment, run the following command:
```
python driver.py
```


## EXPERIMENT 4 (Multilabel classification of classes in a scene)

First download dataset from [here](https://www.kaggle.com/datasets/alphadraco/coco-cat-id-masked-images).

to run the experiment, run the following command:
```
python driver_multilabel.py
```
