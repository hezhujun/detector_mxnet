export CUDA_VISIBLE_DEVICES=0
python train_faster_rcnn.py \
 --dataset coco \
 --train_data_root /root/userfolder/datasets/PennFudanPed/ \
 --train_ann_file /root/userfolder/datasets/PennFudanPed/annotations/PennFudanPed.json \
 --val_data_root /root/userfolder/datasets/PennFudanPed/ \
 --val_ann_file /root/userfolder/datasets/PennFudanPed/annotations/PennFudanPed.json  \
 --gpus 0 \
 --use-fpn \
 --num-workers 4 \
 --lr 0.001 \
 --epochs 12 \
 --lr-decay-epoch 8,11 \
 --save-prefix tct/result \
 --batch-size 1 \
 --kv-store nccl