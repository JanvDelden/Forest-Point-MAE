CUDA_VISIBLE_DEVICES=0
SAMPLING="fps"
drop=0
seed=0

#None
DIR="DATA/none."

python main.py --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
 --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 5 --nruns 10 --num_workers 2 --seed $seed
sleep 5s

python main.py --config cfgs/segmentation/offset.yaml --task offset\
 --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8 --seed $seed
sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
 --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder --seed $seed
sleep 5s

# ShapeNet
DIR="DATA/shapenet."
CKPT="experiments/pretraining/pretrain_official/pretrain.pth"

python main.py --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 5 --nruns 10 --num_workers 2 --seed $seed
sleep 5s

python main.py --config cfgs/segmentation/offset.yaml --task offset\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8 --seed $seed
sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder --seed $seed
sleep 5s

# Forest
DIR="DATA/forest."
CKPT="experiments/pretraining/treeset/ONLY_FOREST/ckpt-best.pth" # 1.7389 loss

python main.py --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 5 --nruns 10 --num_workers 2 --seed $seed
sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder --seed $seed
sleep 5s

python main.py --config cfgs/segmentation/offset.yaml --task offset\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8 --seed $seed
sleep 5s

# BOTH
# covered by patch generation experiment "treeslicefps/kmeans"
