CUDA_VISIBLE_DEVICES=0
drop=0.9
PRETRAINING="fps"
SAMPLING="fps"
CKPT="experiments/pretraining/treeset/sampling/$PRETRAINING/ckpt-best.pth" 

DIR="clsexp_shot5"

python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 5 --nruns 10
sleep 5s

DIR="clsexp_shot10"

python main.py  --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 10 --nruns 10
sleep 5s

DIR="clsexp_all"

python main.py  --config cfgs/classification/cls_treeset.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

DIR="weiser_tree_pretrained"

python main.py  --config cfgs/classification/cls_treeset_weiser.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s

DIR="weiser_tree_randominit"

python main.py  --config cfgs/classification/cls_treeset_weiser.yaml --task cls\
                 --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop
sleep 5s