from mxnet.ndarray.gen_contrib import _Null
import mxnet.ndarray as nd
from mxnet import gluon


def BilinearROIPooling(F, data=None, rois=None, pooled_size=_Null, spatial_scale=_Null, sample_ratio=_Null, type="max", position_sensitive=_Null, out=None, name=None, **kwargs):
    # (BS, C, pooled_size[0], pooled_size[1])
    pooled_feat = F.contrib.ROIAlign(data, rois, pooled_size, spatial_scale, sample_ratio,
                                      position_sensitive)
    # (C, BS, pooled_size[0]*pooled_size[1])
    pooled_feat = pooled_feat.reshape((0, 0, -1)).transpose((1, 0, 2))
    # (C, BS*pooled_size[0]*pooled_size[1]) - >(BS*pooled_size[0]*pooled_size[1], C)
    pooled_feat = pooled_feat.reshape((0, -1)).transpose((1, 0))
    # (BS*pooled_size[0]*pooled_size[1], C, 1)
    pooled_feat = F.expand_dims(pooled_feat, axis=2)
    # (BS*pooled_size[0]*pooled_size[1], C, C)
    out_product = F.batch_dot(pooled_feat, pooled_feat.transpose((0, 2, 1)))
    # (C*C, BS*pooled_size[0]*pooled_size[1])
    out_product = out_product.reshape((0, -1)).transpose((1, 0))
    # (C*C, BS, pooled_size[0]*pooled_size[1])
    out_product = out_product.reshape((0, -1, pooled_size[0] * pooled_size[1]))
    if type == "max":
        # (C*C, BS)
        out_product = F.max(out_product, axis=2)
    elif type == "mean":
        # (C*C, BS)
        out_product = F.mean(out_product, axis=2)
    else:
        raise NotImplementedError()
    # (BS, C*C)
    out_product = out_product.transpose((1, 0))
    return out_product


class BilinearPooling(gluon.HybridBlock):
    def __init__(self, in_channels, roi_size, type="max", prefix=None, params=None):
        super().__init__(prefix, params)
        self.in_channels = in_channels
        assert isinstance(roi_size, (int, tuple, list))
        if isinstance(roi_size, (tuple, list)):
            assert len(roi_size) == 2
        else:
            roi_size = (roi_size, roi_size)
        self.roi_size = roi_size
        self.type = type

    def hybrid_forward(self, F, rois):
        """

        :param F:
        :param rois: (num_rois, in_channels, roi_size[0], roi_size[1])
        :return:
        """
        # (num_rois, roi_size[0], roi_size[1], in_channels)
        rois = rois.transpose((0, 2, 3, 1))
        # (num_rois*roi_size[0]*roi_size[1], in_channels)
        rois = rois.reshape((-1, self.in_channels))
        # (num_rois*roi_size[0]*roi_size[1], in_channels, 1)
        rois = F.expand_dims(rois, axis=2)
        # (num_rois*roi_size[0]*roi_size[1], in_channels, in_channels)
        out_product = F.batch_dot(rois, rois.transpose((0, 2, 1)))
        # (num_rois*roi_size[0]*roi_size[1], in_channels*in_channels)
        out_product = out_product.reshape((0, -1))
        # (num_rois, roi_size[0]*roi_size[1], in_channels*in_channels)
        out_product = out_product.reshape((-1, self.roi_size[0]*self.roi_size[1], self.in_channels**2))
        if self.type == "max":
            # (num_rois, in_channels*in_channels)
            out_product = F.max(out_product, axis=1)
        elif self.type == "mean":
            # (num_rois, in_channels*in_channels)
            out_product = F.mean(out_product, axis=1)
        else:
            raise NotImplementedError()

        return out_product