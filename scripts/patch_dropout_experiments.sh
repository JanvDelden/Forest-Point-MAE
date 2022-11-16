CUDA_VISIBLE_DEVICES=1
SAMPLING="fps"
CKPT="experiments/pretraining/treeset/sampling/fps/ckpt-best.pth" 
# drop =0 see patch generation experiments
# drop=0.5
drop=0.5
DIR="dropout/${drop}."

python main.py --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 5 --nruns 10 --num_workers 2
sleep 5s

python main.py --config cfgs/segmentation/offset.yaml --task offset\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder
sleep 5s

# drop=0.9
drop=0.9
DIR="dropout/${drop}."

python main.py --config cfgs/classification/cls_treeset_fewshot.yaml --task cls\
        --ckpts $CKPT --exp_name $DIR --fewshot --sampling_method $SAMPLING --patch_dropout $drop --shot 5 --nruns 10 --num_workers 2
sleep 5s
python main.py --config cfgs/segmentation/offset.yaml --task offset\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s

python main.py --config cfgs/regression/biomass_treeset.yaml --task regression\
        --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 2 --freeze_encoder
sleep 5s

