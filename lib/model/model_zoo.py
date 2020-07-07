from __future__ import absolute_import

import warnings

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SyncBatchNorm

from .faster_rcnn import get_faster_rcnn
from gluoncv.nn.feature import FPNFeatureExpander


def faster_rcnn_resnet50_v1b(dataset, pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    dataset: VisionDataset
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = faster_rcnn_resnet50_v1b(dataset, pretrained=True)
    >>> print(model)
    """
    assert kwargs["roi_mode"] != "bilinear", "not support"
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b
    classes = dataset.classes
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        strides=16, clip=4.14,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)


def faster_rcnn_fpn_resnet50_v1b(dataset, pretrained=False, pretrained_base=True, **kwargs):
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b
    classes = dataset.classes
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    top_features = None
    if kwargs["roi_mode"] == "bilinear":
        features = FPNFeatureExpander(
            network=base_network,
            outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                     'layers4_relu8_fwd'], num_filters=[100, 100, 100, 100], use_1x1=True,
            use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
        box_features = nn.HybridSequential()
        for _ in range(2):
            box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
            box_features.add(nn.Activation('relu'))
    else:
        features = FPNFeatureExpander(
            network=base_network,
            outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                     'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
            use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
        # 2 FC layer before RCNN cls and reg
        box_features = nn.HybridSequential()
        for _ in range(2):
            box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
            box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, box_features=box_features, classes=classes,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_resnet50_v1b_coco(dataset, pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    dataset: VisionDataset
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = faster_rcnn_resnet50_v1b_coco(dataset, pretrained=True)
    >>> print(model)
    """
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b
    classes = dataset.classes
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.14,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)


def faster_rcnn_fpn_resnet50_v1b_coco(dataset, pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    dataset : VisionDataset
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = faster_rcnn_fpn_resnet50_v1b_coco(dataset, pretrained=True)
    >>> print(model)
    """
    from gluoncv.model_zoo.resnetv1b import resnet50_v1b
    classes = dataset.classes
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, box_features=box_features, classes=classes,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7,7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


_models = {
    'faster_rcnn_resnet50_v1b': faster_rcnn_resnet50_v1b,
    'faster_rcnn_fpn_resnet50_v1b': faster_rcnn_fpn_resnet50_v1b,
    'faster_rcnn_resnet50_v1b_coco': faster_rcnn_resnet50_v1b_coco,
    'faster_rcnn_fpn_resnet50_v1b_coco': faster_rcnn_fpn_resnet50_v1b_coco,
}


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.

    Returns
    -------
    list of str
        Entire list of model names in model_zoo.

    """
    return _models.keys()