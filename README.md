# Forest-Point-MAE

## Self-supervised Learning on 3D Forest Data with Masked Autoencoders

## 1. Setup

```
bash scripts/setup.sh
```


## 2. Datasets

## 3. Point-MAE Pre-training
To pretrain Point-MAE on ShapeNet training set, run the following command.

```
CONFIGFILE="onlyshapenet_pretrain.yaml"
SAMPLING="fps"
DIR="savedirectory"
CUDA_VISIBLE_DEVICES=<GPU> python main.py  --config cfgs/pretraining/$CONFIGFILE\
                 --exp_name $DIR --sampling_method $SAMPLING --task pretrain
```
Pretraining on Forest data: 
```
# either
CONFIGFILE="32kpoints.yaml"
# or 
CONFIGFILE="treeset.yaml"
```
Add --ckpts $CKPT to main.py call to start from a pretrained model. For different patch generation strategies set --sampling method to 'rand', 'fps', 'slice_fps' or 'fpskmeans'.



## 4. Point-MAE Fine-tuning
Classification, run
```
CONFIGFILE="cls_treeset.yaml"
SAMPLING="fpskmeans"
DROP=0
CKPT="path/to/pretrained/model"
CUDA_VISIBLE_DEVICES=<GPU> python main.py  --config cfgs/pretraining/$CONFIGFILE --ckpts $CKPT\
                 --exp_name $DIR --sampling_method $SAMPLING --task cls --patch_dropout $drop
```

Few-shot classification, run:
```
CONFIGFILE="cls_treeset_fewshot.yaml"
CUDA_VISIBLE_DEVICES=<GPU> python main.py  --config cfgs/pretraining/$CONFIGFILE --ckpts $CKPT\
                 --exp_name $DIR --sampling_method $SAMPLING --task cls --patch_dropout $drop --shot 5 --nruns 10
```


Offset prediction, run
```
 python main.py --config cfgs/segmentation/offset.yaml --task offset --ckpts $CKPT\
         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
```
For larger chunks use  '--config cfgs/segmentation/offset32k.yaml'


## 5. Visualization

Visulization of Reconstruction/OFFSET prediction, run:

```
python main_vis.py --ckpts <path/to/pre-trained/model> --config cfgs/pretraining/treeset.yaml --task offset
python main_vis.py --ckpts <path/to/pre-trained/model> --config cfgs/segmentation/offset32k.yaml --task pretrain
```
and visualize the generated .npy files with custom/plotter.ipynb.

## Acknowledgements

The code is mainly based upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
