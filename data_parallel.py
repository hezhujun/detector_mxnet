"""Data parallel task for Faster RCNN Model."""

from mxnet import autograd
from mxnet.contrib import amp
from mxnet import contrib, ndarray as nd

from gluoncv.utils.parallel import Parallelizable


class ForwardBackwardTask(Parallelizable):
    """ Faster R-CNN training task that can be scheduled concurrently using Parallel.
    Parameters
    ----------
    net : gluon.HybridBlock
        Faster R-CNN network.
    optimizer : gluon.Trainer
        Optimizer for the training.
    rpn_cls_loss : gluon.loss
        RPN box classification loss.
    rpn_box_loss : gluon.loss
        RPN box regression loss.
    rcnn_cls_loss : gluon.loss
        R-CNN box head classification loss.
    rcnn_box_loss : gluon.loss
        R-CNN box head regression loss.
    mix_ratio : int
        Object detection mixup ratio.
    amp_enabled : bool
        Whether to enable Automatic Mixed Precision.
    """

    def __init__(self, net, optimizer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss,
                 mix_ratio, amp_enabled):
        super(ForwardBackwardTask, self).__init__()
        self.net = net
        self._optimizer = optimizer
        self.rpn_cls_loss = rpn_cls_loss
        self.rpn_box_loss = rpn_box_loss
        self.rcnn_cls_loss = rcnn_cls_loss
        self.rcnn_box_loss = rcnn_box_loss
        self.mix_ratio = mix_ratio
        self.amp_enabled = amp_enabled

    def forward_backward(self, x):
        data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks = x
        with autograd.record():
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            # cls_pred (BS, num_samples, C+1)
            # box_pred (BS, num_pos, C, 4)
            # rpn_box  (BS, num_samples, 4)
            # samples  (BS, num_samples)   gt_class_id
            # matches  (BS, num_samples)   gt_indices
            # raw_rpn_score (BS, num_anchors, 1)
            # raw_rpn_box   (BS, num_anchors, 4)
            # anchors       (BS, num_anchors, 4)
            # cls_targets   (BS, num_samples)
            # box_targets   (BS, num_pos, C, 4)
            # box_masks     (BS, num_pos, C, 4)
            # indices       (BS, num_pos)       相对于rpn_box的序号
            # roi           (BS, rpn_post_nms, 4) rpn返回的roi
            # cls_pred, box_pred, rpn_box, samples, matches, raw_rpn_score, raw_rpn_box, anchors, cls_targets, \
            #     box_targets, box_masks, indices = self.net(data, gt_box, gt_label)
            cls_pred, box_pred, _, _, _Z, rpn_score, rpn_box, _, cls_targets, \
                box_targets, box_masks, _, roi = self.net(data, gt_box, gt_label)
            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            # rpn_cls_targets: 1: pos 0: neg -1: ignore
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = self.rpn_cls_loss(rpn_score, rpn_cls_targets,
                                          rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = self.rpn_box_loss(rpn_box, rpn_box_targets,
                                          rpn_box_masks) * rpn_box.size / num_rpn_pos
            # rpn overall loss, use sum rather than average
            rpn_loss = rpn_loss1 + rpn_loss2
            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = self.rcnn_cls_loss(
                cls_pred, cls_targets, cls_targets.expand_dims(-1) >= 0) * cls_targets.size / \
                         num_rcnn_pos
            rcnn_loss2 = self.rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / \
                         num_rcnn_pos
            rcnn_loss = rcnn_loss1 + rcnn_loss2
            # overall losses
            total_loss = rpn_loss.sum() * self.mix_ratio + rcnn_loss.sum() * self.mix_ratio

            rpn_loss1_metric = rpn_loss1.mean() * self.mix_ratio
            rpn_loss2_metric = rpn_loss2.mean() * self.mix_ratio
            rcnn_loss1_metric = rcnn_loss1.mean() * self.mix_ratio
            rcnn_loss2_metric = rcnn_loss2.mean() * self.mix_ratio
            rpn_acc_metric = [[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]]
            rpn_l1_loss_metric = [[rpn_box_targets, rpn_box_masks], [rpn_box]]
            rcnn_acc_metric = [[cls_targets], [cls_pred]]
            rcnn_l1_loss_metric = [[box_targets, box_masks], [box_pred]]

            if self.amp_enabled:
                with amp.scale_loss(total_loss, self._optimizer) as scaled_losses:
                    autograd.backward(scaled_losses)
            else:
                total_loss.backward()

            # rpn_gt_recalls = []
            # for i in range(gt_label.shape[0]):
            #     # 如果两个box面积都是0，iou是0
            #     iou = contrib.ndarray.box_iou(roi[i], gt_box[i], format='corner')
            #     iou_gt_max = nd.max(iou, axis=0)
            #     _gt_label = nd.squeeze(gt_label[i])
            #     iou_gt_max = contrib.nd.boolean_mask(iou_gt_max, _gt_label != -1)
            #     rpn_gt_recall = nd.mean(iou_gt_max >= 0.5)
            #     rpn_gt_recalls.append(rpn_gt_recall)
            # rpn_gt_recall = sum(rpn_gt_recalls) / len(rpn_gt_recalls)
            rpn_gt_recall = nd.zeros((1,), ctx=roi.context)

        return rpn_loss1_metric, rpn_loss2_metric, rcnn_loss1_metric, rcnn_loss2_metric, rpn_gt_recall, \
               rpn_acc_metric, rpn_l1_loss_metric, rcnn_acc_metric, rcnn_l1_loss_metric