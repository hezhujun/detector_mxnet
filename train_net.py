import argparse
import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'

import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.contrib import amp   # 混合精度模块
import gluoncv as gcv

gcv.utils.check_version('0.7.0')
from gluoncv import data as gdata
from gluoncv import utils as gutils
from lib.model.model_zoo import get_model
from gluoncv.data.batchify import FasterRCNNTrainBatchify, Tuple, Append
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, \
    FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.parallel import Parallel
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric
from data_parallel import ForwardBackwardTask

import yaml
from PIL import Image
from lib.dataset.coco import COCODataset
from lib.util import draw_image
from gluoncv import utils as gutils
from utils import *


class Record(mx.metric.EvalMetric):

    def update(self, _, record):
        self.sum_metric += record
        self.global_sum_metric += record
        self.num_inst += 1
        self.global_num_inst += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg.yaml")
    args = parser.parse_args()
    return args


def validate(net, val_data, ctx, eval_metric, cfg):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not cfg["train"]["disable_hybridization"]:
        # input format is differnet than training, thus rehybridization is needed.
        net.hybridize(static_alloc=cfg["train"]["static_alloc"])
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        rpn_gt_recalls = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes, roi = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            gt_label = y[:, :, 4:5]
            gt_box = y[:, :, :4]
            for i in range(gt_label.shape[0]):
                _gt_label = nd.squeeze(gt_label[i])
                match_mask = nd.zeros_like(_gt_label)
                # 如果两个box面积都是0，iou是0
                iou = nd.contrib.box_iou(roi[i], gt_box[i], format='corner')
                num_raw = iou.shape[1]
                # 为每个gt box分配anchor
                # 参考http://zh.d2l.ai/chapter_computer-vision/anchor.html#%E6%A0%87%E6%B3%A8%E8%AE%AD%E7%BB%83%E9%9B%86%E7%9A%84%E9%94%9A%E6%A1%86
                for _ in range(_gt_label.shape[0]):
                    _iou = iou.reshape(-1)
                    max = nd.max(_iou, axis=0)
                    if max < 0.5:
                        break
                    pos = nd.argmax(_iou, axis=0)
                    raw = (pos / num_raw).astype(np.int64)
                    col = pos % num_raw
                    iou[raw, :] = 0
                    iou[:, col] = 0
                    match_mask[col] = 1
                match_mask = nd.contrib.boolean_mask(match_mask, _gt_label != -1)
                rpn_gt_recalls.append(nd.mean(match_mask).asscalar())

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                        det_scores, gt_bboxes,
                                                                        gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    rpn_gt_recall = np.mean(rpn_gt_recalls)
    print("RPN GT Recall", rpn_gt_recall)
    return eval_metric.get()


def train(net, train_loader, val_loader, eval_metric, ctx, cfg):
    kv = mx.kvstore.create(cfg["train"]["kv_store"])
    net.collect_params().setattr('grad_req', 'null')
    # 需要训练的参数的train_pattern在构造时输入到网络中
    net.collect_train_params().setattr('grad_req', 'write')
    optimizer_params = {
        "learning_rate": cfg["train"]["lr"],
        "wd": cfg["train"]["wd"],
        "momentum": cfg["train"]["momentum"],
    }
    trainer = gluon.Trainer(
        net.collect_train_params(),
        "sgd",
        optimizer_params,
        kvstore=kv
    )

    lr_decay = float(cfg["train"]["lr_decay"])
    lr_steps = sorted([float(ls) for ls in cfg["train"]["lr_decay_epoch"]])
    lr_warmup = float(cfg["train"]["lr_warmup_iteration"])

    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=cfg["train"]["rpn_smoothl1_rho"])
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss(rho=cfg["train"]["rcnn_smoothl1_rho"])

    metrics = [mx.metric.Loss("RPN_Conf"),
               mx.metric.Loss("RPN_SmoothL1"),
               mx.metric.Loss("RCNN_CrossEntropy"),
               mx.metric.Loss("RCNN_SmoothL1"),
               mx.metric.Loss('RPN_GT_Recall'),]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    data_prepare_time = Record("data_prepare_time")
    data_distributed_time = Record("data_distributed_time")
    net_forward_backward_time = Record("net_forward_backward_time")

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = cfg["train"]["save_prefix"] + "_train.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(cfg)

    if cfg["verbose"]:
        logger.info("Trainable parameters:")
        logger.info(net.collect_train_params().keys())
    logger.info("Start training from [Epoch {}]".format(cfg["train"]['start_epoch']))
    best_map = [0]
    for epoch in range(cfg["train"]["start_epoch"], cfg["train"]["epochs"]):
        rcnn_task = ForwardBackwardTask(net, trainer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss,
                                        rcnn_box_loss, mix_ratio=1.0, amp_enabled=None)
        # 多线程执行运算操作
        # 每个线程处理一部分数据，这部分数据可以在任意gpus上
        # 线程数量和gpus数量无关
        # 在多gpus训练中，每个设备都要调用一次forward-backward操作
        # 每次调用都需要等待执行结果
        # 使用Parallel，每次调用都不需要等待执行结果，
        # 实现并行运行
        # 也可以理解为后台线程进行运算（如果Parallel设置成主线程不运算）
        # cfg["executor_threads"]设置成1，
        # Parallel的代码：
        # 前cfg["executor_threads"]次put()的数据在主线程运算，
        # 后面再调用put()在后台线程运算
        # 目的是在第一次迭代中模型在主线程初始化
        # 调用cfg["executor_threads"]次是为了保证模型在不同设备上正常进行初始化？
        executor = Parallel(cfg["train"]["executor_threads"], rcnn_task)
        mix_ratio = 1.0
        if not cfg["train"]["disable_hybridization"]:
            net.hybridize(static_alloc=cfg["train"]["static_alloc"])

        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))

        for metric in metrics:
            metric.reset()
        for metric in metrics2:
            metric.reset()

        data_prepare_time.reset()
        data_distributed_time.reset()
        net_forward_backward_time.reset()

        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio

        before_data_prepare_point = time.time()
        for i, batch in enumerate(train_loader):
            data_prepare_time.update(None, time.time() - before_data_prepare_point)
            if epoch == 0 and i <= lr_warmup:
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup, cfg["train"]["lr_warmup_factor"])
                if new_lr != trainer.learning_rate:
                    if i % cfg["train"]["log_interval"] == 0:
                        logger.info("[Epoch 0 Iteration {}] Set learning rate to {}".format(i, new_lr))
                trainer.set_learning_rate(new_lr)
            before_data_distributed_point = time.time()
            # img, label, cls_targets, box_targets, box_masks = batch
            batch = split_and_load(batch, ctx_list=ctx)  # 分发数据
            data_distributed_time.update(None, time.time() - before_data_distributed_point)
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]

            before_net_forward_backward_point = time.time()
            for data in zip(*batch):
                executor.put(data)  #
            for j in range(len(ctx)):
                result = executor.get()
                for k in range(len(metric_losses)):
                    metric_losses[k].append(result[k])
                for k in range(len(add_losses)):
                    add_losses[k].append(result[len(metric_losses) + k])
            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)  # 把所有loss放到一起
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])
            trainer.step(cfg["dataset"]["batch_size_per_device"]*len(ctx))
            net_forward_backward_time.update(None, time.time() - before_net_forward_backward_point)

            if cfg["train"]["log_interval"] and not (i + 1) % cfg["train"]["log_interval"]:
                msg = ",".join([
                    "{}={:.3f}".format(*metric.get()) for metric in metrics + metrics2
                ])
                logger.info("[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}".format(
                    epoch, i, cfg["train"]["log_interval"] * cfg["dataset"]["batch_size_per_device"] * len(ctx) / (time.time() - btic), msg
                ))
                time_msg = ",".join([
                    "{}={}".format(*metric.get()) for metric in [data_prepare_time, data_distributed_time, net_forward_backward_time]
                ])
                logger.info("[Epoch {}][Batch {}], {}".format(epoch, i, time_msg))
                btic = time.time()

            before_data_prepare_point = time.time()

        msg = ",".join(["{}={:.3f}".format(*metric.get()) for metric in metrics])
        logger.info("[Epoch {}] Training cost: {:.3f}, {}".format(
            epoch, (time.time() - tic), msg
        ))
        if not (epoch + 1) % cfg["train"]["val_interval"]:
            map_name, mean_ap = validate(net, val_loader, ctx, eval_metric, cfg)
            val_msg = "\n".join(["{}={}".format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info("[Epoch {}] Validation: \n{}".format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, cfg["train"]["save_interval"], cfg["train"]["save_prefix"])


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, "r")as f:
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        cfg = yaml.load(f, Loader)

    cfg["train"]["wd"] = float(cfg["train"]["wd"])

    gutils.random.seed(cfg["seed"])
    gpus = cfg["train"]["gpus"]
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])
        print("CUDA_VISIBLE_DEVICES=".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    ctx = [mx.gpu(i) for i in range(len(gpus))]
    ctx = ctx if ctx else [mx.cpu()]

    if not os.path.exists(cfg["train"]["save_dir"]):
        os.makedirs(cfg["train"]["save_dir"])

    train_data = cfg["dataset"]["train_data"]
    train_dataset = COCODataset(root=train_data["root"], annFile=train_data["annFile"], use_crowd=False)

    val_data = cfg["dataset"]["val_data"]
    val_dataset = COCODataset(root=val_data["root"], annFile=val_data["annFile"], use_crowd=False)

    save_path = os.path.join(cfg["train"]["save_dir"], "eval")
    eval_metric = COCODetectionMetric(val_dataset, save_path, cleanup=True)

    # network
    kwargs = {}
    module_list = []
    model_cfg = cfg["model"].copy()
    if model_cfg.pop("use_fpn"):
        module_list.append("fpn")
    norm_layer = model_cfg.pop("norm_layer")
    if norm_layer:
        module_list.append(norm_layer)
        if norm_layer == "syncbn":
            kwargs["num_devices"] = len(ctx)

    num_gpus = len(ctx)
    net_name = "_".join(("faster_rcnn", *module_list, model_cfg.pop("network")))
    resume = model_cfg.pop("resume")
    net = get_model(
        net_name,
        dataset=train_dataset,
        pretrained_base=True,
        per_device_batch_size=cfg["dataset"]["batch_size_per_device"],
        **kwargs,
        **model_cfg
    )

    cfg["train"]["save_prefix"] = os.path.join(cfg["train"]["save_dir"], net_name)
    if resume and resume.strip():
        net.load_parameters(resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # dataloader
    train_bfn = FasterRCNNTrainBatchify(net, len(ctx))
    if hasattr(train_dataset, 'get_im_aspect_ratio'):
        im_aspect_ratio = train_dataset.get_im_aspect_ratio()
    else:
        im_aspect_ratio = [1.] * len(train_dataset)
    train_sampler = gcv.nn.sampler.SplitSortedBucketSampler(im_aspect_ratio,
                                                            batch_size=cfg["dataset"]["batch_size_per_device"] * len(ctx),
                                                            num_parts=1,
                                                            part_index=0,
                                                            shuffle=True)
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(
            FasterRCNNDefaultTrainTransform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=cfg["model"]["use_fpn"])
        ),
        batch_sampler=train_sampler,
        batchify_fn=train_bfn,
        num_workers=cfg["dataset"]["num_workers"],
    )

    val_bfn = Tuple(*[Append() for _ in range(3)])
    # short是列表或元组，是为了在训练过程中增加图片大小抖动这一数据增强方式
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(
            FasterRCNNDefaultValTransform(short, net.max_size)
        ),
        batch_size=len(ctx),  # 每张卡一张图片
        shuffle=False,
        batchify_fn=val_bfn,
        last_batch='keep',
        num_workers=cfg["dataset"]["num_workers"]
    )

    train(net, train_loader, val_loader, eval_metric, ctx, cfg)

