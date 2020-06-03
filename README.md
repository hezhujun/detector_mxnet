# Detector_mxnet

使用mxnet实现的Faster RCNN。

# Detail

## COCODetection
从COCODetection获取的元素item是一个元组(img, label)。img是mxnet ndarray，原图大小。label是np的ndarray，是一个n*5的矩阵，n表示这张图片objs的个数，5表示(x1, y1, x2, y2, class_id)，class_id是json文件类别id映射之后的类别id。


## FasterRCNNDefaultTrainTransform
输出:
- img: mxnet.NDArray, (3, h, w) 经过缩放，最小边为short，最大边不超过max_size，如果short是(a,b)，最小边在[a,b]范围内抖动。
- label: numpy.ndarray, (num_objs, 5) 每个objs label格式(x1, y1, x2, y2, class_id)
- cls_targets: list, 每个元素(h_i, w_i, num_anchors_per_position)表示每一层的anchors对应的class_id(包括背景类-1)。
- box_targets： list, 每个元素(h_i, w_i, num_anchors_per_position×4)，没有对应gt bbox的位置填充0。
- box_masks：list, 每个元素(h_i, w_i, num_anchors_per_position×4)，配合box_targets使用。

## FasterRCNNTrainBatchify
FasterRCNNTrainBatchify把加载的batch分成num_shared份，每份是一个小batch，batch size通常是1。后续处理会把每份数据加载到相应的gpu上。

输出

- img：如果只加载一张图片，mxnet.NDArray, (1, 3, h, w)。如果加载多张图片，list，每个元素(BS, 3, h, w)，

- label：list，每个元素(BS, num_objs, 5)，同一个元素内用-1填充对齐。

- cls_targets：tuple，每个元素(BS, num_anchors)。

- box_targets：tuple，每个元素(BS, num_anchors, 4)。

- box_masks：tuple，每个元素(BS, num_anchors, 4)。



## Issue

gluon.Trainer.step()会在gpu(0)申请内存？如果gpu(0)不能使用，记得添加系统变量export CUDA_VISIBLE_DEVICES= 指定程序使用的gpus。