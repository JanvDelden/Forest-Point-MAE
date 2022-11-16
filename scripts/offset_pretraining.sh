CUDA_VISIBLE_DEVICES=2
drop=0

# DIR="ONLY_FOREST"
# SAMPLING="fps"
# python main.py --config cfgs/pretraining/treeset.yaml\
#                  --exp_name $DIR --sampling_method $SAMPLING --task pretrain

DIR="DATA/forest."
CKPT="experiments/pretraining/treeset/ONLY_FOREST/ckpt-best.pth" # 1.7389 loss
SAMPLING="fps"



python main.py --config cfgs/segmentation/offset2.yaml --task offset\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s



