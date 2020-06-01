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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg.yaml")
    args = parser.parse_args()
    return args


def get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    # lr factor 线性增长函数
    # 最小为lr_warmup_factor
    # 最大为1
    return lr_warmup_factor * (1 - alpha) + alpha


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            # train_loader 返回的数据，数据的每一项都是list/tuple，每个device处理list/tuple中的一项
            # val_loader 同上，只是保证list/tuple中每项的batch size都是1
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_loader, ctx, eval_metric, cfg):
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not cfg["disable_hybridization"]:
        net.hybridize(static_alloc=cfg["static_alloc"])
    for batch in val_loader:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_bboxes.append(clipper(bboxes, x))
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in \
                zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)

    return eval_metric.get()


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info("[Epoch {}] mAP {} higher than current best {} saving to {}".format(
            epoch, current_map, best_map, "{:s}_best.params".format(prefix)
        ))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + "_best_map.log", "a") as f:
            f.write("{:04d}:\t{:.4f}\n".format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info("[Epoch {}] Saving parameter to {}".format(
            epoch, "{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map)
        ))
        net.save_parameters("{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map))


def train(net, train_loader, val_loader, eval_metric, ctx, cfg):
    kv = mx.kvstore.create(cfg["kv_store"])
    net.collect_params().setattr('grad_req', 'null')
    # 需要训练的参数的train_pattern在构造时输入到网络中
    net.collect_train_params().setattr('grad_req', 'write')
    optimizer_params = {
        "learning_rate": cfg["lr"],
        "wd": cfg["wd"],
        "momentum": cfg["momentum"],
    }
    trainer = gluon.Trainer(
        net.collect_train_params(),
        "sgd",
        optimizer_params,
        kvstore=kv
    )

    lr_decay = float(cfg["lr_decay"])
    lr_steps = sorted([float(ls) for ls in cfg["lr_decay_epoch"]])
    lr_warmup = float(cfg["lr_warmup"])

    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=cfg["rpn_smoothl1_rho"])
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss(rho=cfg["rcnn_smoothl1_rho"])

    metrics = [mx.metric.Loss("RPN_Conf"),
               mx.metric.Loss("RPN_SmoothL1"),
               mx.metric.Loss("RCNN_CrossEntropy"),
               mx.metric.Loss("RCNN_SmoothL1")]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = cfg["save_prefix"] + "_train.log"
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(cfg)

    if cfg["verbose"]:
        logger.info("Trainable parameters:")
        logger.info(net.collect_train_params().keys())
    logger.info("Start training from [Epoch {}]".format(cfg['start_epoch']))
    best_map = [0]
    for epoch in range(cfg["start_epoch"], cfg["epochs"]):
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
        executor = Parallel(cfg["executor_threads"], rcnn_task)
        mix_ratio = 1.0
        if not cfg["disable_hybridization"]:
            net.hybridize(static_allo=cfg["static_alloc"])

        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))

        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio

        for i, batch in enumerate(train_loader):
            if epoch == 0 and i <= lr_warmup:
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup, cfg["lr_warmup_factor"])
                if new_lr != trainer.learning_rate:
                    if i % cfg["log_interval"] == 0:
                        logger.info("[Epoch 0 Iteration {}] Set learning rate to {}".format(i, new_lr))
                trainer.set_learning_rate(new_lr)

            # img, label, cls_targets, box_targets, box_masks = batch
            batch = split_and_load(batch, ctx_list=ctx)  # 分发数据
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
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
            trainer.step(cfg["batch_size_per_device"]*len(ctx))

            if cfg["log_interval"] and not (i + 1) % cfg["log_interval"]:
                msg = ",".join([
                    "{}={:.3f}".format(*metric.get()) for metric in metrics + metrics2
                ])
                logger.info("[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}".format(
                    epoch, i, cfg["log_interval"] * cfg["batch_size_per_device"] * len(ctx) / (time.time() - btic), msg
                ))
                btic = time.time()

        msg = ",".join(["{}={:.3f}".format(*metric.get()) for metric in metrics])
        logger.info("[Epoch {}] Training cost: {:.3f}, {}".format(
            epoch, (time.time() - tic), msg
        ))
        if not (epoch + 1) % cfg["val_interval"]:
            map_name, mean_ap = validate(net, val_loader, ctx, eval_metric, cfg)
            val_msg = "\n".join(["{}={}".format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info("[Epoch {}] Validation: \n{}".format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, cfg["save_interval"], cfg["save_prefix"])


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, "r")as f:
        try:
            from yaml import CLoader as Loader, CDumper as Dumper
        except ImportError:
            from yaml import Loader, Dumper
        cfg = yaml.load(f, Loader)

    gutils.random.seed(cfg["seed"])
    ctx = [mx.gpu(i) for i in cfg["gpus"]]
    ctx = ctx if ctx else [mx.cpu()]

    train_data = cfg["dataset"]["train_data"]
    train_dataset = COCODataset(root=train_data["root"], annFile=train_data["annFile"], use_crowd=False)

    val_data = cfg["dataset"]["val_data"]
    val_dataset = COCODataset(root=val_data["root"], annFile=val_data["annFile"], use_crowd=False)

    save_path = cfg["save_prefix"] + "_eval"
    _dir = os.path.dirname(save_path)
    if _dir and not os.path.exists(_dir):
        os.makedirs(_dir)
    eval_metric = COCODetectionMetric(val_dataset, save_path, cleanup=True)

    # network
    kwargs = {}
    module_list = []
    if cfg["use_fpn"]:
        module_list.append("fpn")
    if cfg["norm_layer"]:
        module_list.append(cfg["norm_layer"])
        if cfg["norm_layer"] == "syncbn":
            kwargs["num_devices"] = len(ctx)

    num_gpus = len(ctx)
    net_name = "_".join(("faster_rcnn", *module_list, cfg["network"], "coco"))
    net = get_model(net_name, classes=train_dataset.classes, pretrained_base=True, per_device_batch_size=cfg["batch_size_per_device"], **kwargs)

    cfg["save_prefix"] += net_name
    if cfg["resume"] and cfg["resume"].strip():
        net.load_parameters(cfg["resume"].strip())
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
                                                            batch_size=cfg["batch_size_per_device"] * len(ctx),
                                                            num_parts=1,
                                                            part_index=0,
                                                            shuffle=True)
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(
            FasterRCNNDefaultTrainTransform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=cfg["use_fpn"])
        ),
        batch_sampler=train_sampler,
        batchify_fn=train_bfn,
        num_workers=cfg["num_workers"],
    )

    val_bfn = Tuple(*[Append() for _ in range(3)])
    # short是列表或元组是为了支持训练过程中增加图片大小抖动这一数据增强方式
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(
            FasterRCNNDefaultValTransform(short, net.max_size)
        ),
        batch_size=len(ctx),  # 每张卡一张图片
        shuffle=False,
        batchify_fn=val_bfn,
        last_batch='keep',
        num_workers=cfg["num_workers"]
    )

    train(net, train_loader, val_loader, eval_metric, ctx, cfg)

