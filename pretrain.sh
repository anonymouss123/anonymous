LOG_DIR=logger
mkdir -p ${LOG_DIR}
echo ${LOG_DIR}
NOW=$(date +"%Y%m%d_%H%M%S")

script -c 'python main_pretrain.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10501' --multiprocessing-distributed --world-size 1 --rank 0 \
  --logger pretrain_fedi --lr 0.5 --epochs 100 --pred-dim 8192 --batch-size 1024 --print-freq 100 \
  -data data/imagenet' ${LOG_DIR}/fedi-100ep-${NOW}.log 