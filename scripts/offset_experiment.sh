CUDA_VISIBLE_DEVICES=5
SAMPLING="fps"
drop=0
CKPT="experiments/pretraining/32kpoints/jonathan_chunks/small_encoder/ckpt-best.pth"

#FULL TRAINSET
DATASET="full_trainset"
DIR="DATANEW/randominit/${DATASET}"

python main.py --config cfgs/segmentation/testset/${DATASET}.yaml --task offset\
 --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s

DIR="DATANEW/pretrained/${DATASET}"
python main.py --config cfgs/segmentation/testset/${DATASET}.yaml --task offset\
 --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s

#HALF TRAINSET
DATASET="half_trainset"
DIR="DATANEW/randominit/${DATASET}"

python main.py --config cfgs/segmentation/testset/${DATASET}.yaml --task offset\
 --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s

DIR="DATANEW/pretrained/${DATASET}"
python main.py --config cfgs/segmentation/testset/${DATASET}.yaml --task offset\
 --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
 sleep 5s

#QUARTER TRAINSET
DATASET="quarter_trainset"
DIR="DATANEW/randominit/${DATASET}"

python main.py --config cfgs/segmentation/testset/${DATASET}.yaml --task offset\
 --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
sleep 5s

DIR="DATANEW/pretrained/${DATASET}"
python main.py --config cfgs/segmentation/testset/${DATASET}.yaml --task offset\
 --ckpts $CKPT --exp_name $DIR --sampling_method $SAMPLING --patch_dropout $drop --num_workers 8
 sleep 5s
