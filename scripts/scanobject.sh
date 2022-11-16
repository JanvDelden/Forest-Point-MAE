CUDA_VISIBLE_DEVICES=3
drop=0
SAMPLING="fps"
CKPT="experiments/pretraining/pretrain_official/pretrain.pth" 

DIR="SCNOBJ_drop0_"
python main.py  --config cfgs/classification/scanobject_hardest.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

drop=0.5
DIR="SCNOBJ_drop05_"
python main.py  --config cfgs/classification/scanobject_hardest.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

drop=0.9
DIR="SCNOBJ_drop09_"
python main.py  --config cfgs/classification/scanobject_hardest.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop

drop=0.9
DIR="SCNOBJ_drop09_resampling"
python main.py  --config cfgs/classification/scanobject_resampling.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop

CKPT="experiments/pretraining/only_shapenet_pretrain/baseline/ckpt-best.pth "
python main.py  --config cfgs/classification/scanobject_resampling.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop