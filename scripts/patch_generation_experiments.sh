
#experiments/pretraining/treeset/sampling/.../ckpt-best.pth

CUDA_VISIBLE_DEVICES=1
drop=0

# rand
PRETRAINING="rand"
SAMPLING="rand"
DIR="tree${PRETRAINING}/${SAMPLING}"
CKPT="experiments/pretraining/treeset/sampling/$PRETRAINING/ckpt-best.pth" 
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s
# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s
python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder
sleep 5s

SAMPLING="fpskmeans"
DIR="tree${PRETRAINING}/${SAMPLING}"
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s

# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
# sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder

# fps
PRETRAINING="fps"
SAMPLING="fps"
DIR="tree${PRETRAINING}/${SAMPLING}"
CKPT="experiments/pretraining/treeset/sampling/$PRETRAINING/ckpt-best.pth" 
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s

# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
# sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder
sleep 5s

SAMPLING="fpskmeans"
DIR="tree${PRETRAINING}/${SAMPLING}"
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s

# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
# sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder

# slice_fps
PRETRAINING="slice_fps"
SAMPLING="slice_fps"
DIR="tree${PRETRAINING}/${SAMPLING}"
CKPT="experiments/pretraining/treeset/sampling/$PRETRAINING/ckpt-best.pth" 
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s

# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
# sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder
sleep 5s

SAMPLING="fpskmeans"
DIR="tree${PRETRAINING}/${SAMPLING}"
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s

# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
# sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder

#kmeans
PRETRAINING="kmeans"
SAMPLING="fpskmeans"
DIR="tree${PRETRAINING}/${SAMPLING}"
CKPT="experiments/pretraining/treeset/sampling/$PRETRAINING/ckpt-best.pth" 
# python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
#         --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --shot 5 --nruns 10
# sleep 5s

# python main.py --config cfgs/segmentation/offset.yaml --task offset\
#         --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
# sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder
