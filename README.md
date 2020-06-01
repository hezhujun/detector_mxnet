# Detector_mxnet

使用mxnet实现的Faster RCNN。

# Detail

## COCODetection
从COCODetection获取的元素item是一个元组(img, label)。img是mxnet ndarray，原图大小。label是np的ndarray，是一个n*5的矩阵，n表示这张图片objs的个数，5表示(x1, y1, x2, y2, class_id)，class_id是json文件类别id映射之后的类别id。

## FasterRCNNTrainBatchify
数据处理流的输出

- sharded_img：list，个数等于gpus的个数，每个元素是图片batch (BS, 3, max_h, max_w)，max取BS张图片的最大值，图片是用0填充。训练时一般是一卡一图，所以BS是1。

- sharded_label：list，个数等于gpus的个数，每个元素(BS, max_objs, 5)，max_objs取决于BS张图片中最大的objs数，用-1填充。

- sharded_cls_targets：list，个数等于gpus的个数，每个元素(BS, num_anchors)，因为同一个item中BS张图片都padding成同样大小，所以BS张图片所对应的anchors是一样的，从这里开始到后面的输出每个item里面都不需要padding。

- sharded_box_targets：list，个数等于gpus的个数，每个元素(BS, num_anchors, 4)

- sharded_box_masks：list，个数等于gpus的个数，每个元素(BS, num_anchors, 4)