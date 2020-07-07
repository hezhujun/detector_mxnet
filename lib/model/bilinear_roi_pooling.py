import mxnet.ndarray as nd
from mxnet import gluon, autograd
import numpy as np
import mxnet as mx


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


def bilinear_roi_pooling(data, rois, spatial_scale, type="max"):
    """

    :param data: (BS, C, H, W)
    :param rois: (N, 5)
    :param spatial_scale: float
    :param type:
    :return:
    """
    assert isinstance(spatial_scale, float)
    BS, C, H, W = data.shape
    N = rois.shape[0]
    out_data = []
    rois = rois.asnumpy()
    for i in range(N):
        roi = rois[i]
        batch_id = roi[0].astype(np.int64)
        x1, y1, x2, y2 = roi[1:] * spatial_scale
        x1, y1, x2, y2 = np.floor(x1), np.floor(y1), np.ceil(x2), np.ceil(y2)
        x1, y1, x2, y2 = np.clip(x1, 0, W), np.clip(y1, 0, H), np.clip(x2, 0, W), np.clip(y2, 0, H)
        x1, y1, x2, y2 = x1.astype(np.int64), y1.astype(np.int64), x2.astype(np.int64), y2.astype(np.int64)
        if x1 >= x2 or y1 >= y2:
            out_data.append(nd.zeros((C, C), ctx=data.context, dtype=data.dtype))
            continue
        # (C, h, w)
        roi_data = data[batch_id, :, y1:y2, x1:x2]
        # (h*w, C)
        roi_data = roi_data.reshape((C, -1)).transpose((1, 0))
        # (h*w, C, 1)
        roi_data = roi_data.reshape((0, 0, 1))
        # (h*w, C, C)
        out_product = nd.batch_dot(roi_data, roi_data.transpose((0, 2, 1)))
        # (C, C)
        if type == "max":
            reduce_product = nd.max(out_product, axis=0)
        elif type == "mean":
            reduce_product = nd.mean(out_product, axis=0)
        else:
            raise NotImplementedError()
        out_data.append(reduce_product)
    out_data = nd.stack(*out_data)
    return out_data


class BilinearROIPooling(mx.operator.CustomOp):

    def __init__(self, spatial_scale, type="max"):
        self.spatial_scale = float(spatial_scale)
        self.type = type
        assert type == "max" or type == "mean", "not support bilinear roi pooling type: {}".format(type)

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        rois = in_data[1]
        BS, C, H, W = data.shape
        N = rois.shape[0]
        out = []
        rois = rois.asnumpy()
        for i in range(N):
            roi = rois[i]
            batch_id = roi[0].astype(np.int64)
            x1, y1, x2, y2 = roi[1:] * self.spatial_scale
            x1, y1, x2, y2 = np.floor(x1), np.floor(y1), np.ceil(x2), np.ceil(y2)
            x1, y1, x2, y2 = np.clip(x1, 0, W), np.clip(y1, 0, H), np.clip(x2, 0, W), np.clip(y2, 0, H)
            x1, y1, x2, y2 = x1.astype(np.int64), y1.astype(np.int64), x2.astype(np.int64), y2.astype(np.int64)
            if x1 >= x2 or y1 >= y2:
                out.append(nd.zeros((C, C), ctx=data.context, dtype=data.dtype))
                continue
            # (C, h, w)
            roi_data = data[batch_id, :, y1:y2, x1:x2]
            # (h*w, C)
            roi_data = roi_data.reshape((C, -1)).transpose((1, 0))
            # (h*w, C, 1)
            roi_data = roi_data.reshape((0, 0, 1))
            # (h*w, C, C)
            out_product = nd.batch_dot(roi_data, roi_data.transpose((0, 2, 1)))
            if self.type == "max":
                reduce_product = nd.max(out_product, axis=0)
            elif self.type == "mean":
                reduce_product = nd.mean(out_product, axis=0)
            else:
                raise NotImplementedError()
            out.append(reduce_product)
        out = nd.stack(*out)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        rois = in_data[1]
        BS, C, H, W = data.shape
        N = rois.shape[0]
        dout = out_grad[0]
        ddata = nd.zeros_like(data)

        rois = rois.asnumpy()
        for i in range(N):
            roi = rois[i]
            batch_id = roi[0].astype(np.int64)
            x1, y1, x2, y2 = roi[1:] * self.spatial_scale
            x1, y1, x2, y2 = np.floor(x1), np.floor(y1), np.ceil(x2), np.ceil(y2)
            x1, y1, x2, y2 = np.clip(x1, 0, W), np.clip(y1, 0, H), np.clip(x2, 0, W), np.clip(y2, 0, H)
            x1, y1, x2, y2 = x1.astype(np.int64), y1.astype(np.int64), x2.astype(np.int64), y2.astype(np.int64)
            if x1 >= x2 or y1 >= y2:
                continue
            h = y2 - y1
            w = x2 - x1
            # (C, h, w)
            roi_data = data[batch_id, :, y1:y2, x1:x2]
            # (h*w, C)
            roi_data = roi_data.reshape((C, -1)).transpose((1, 0))
            # (h*w, C, 1)
            roi_data = roi_data.reshape((0, 0, 1))
            # (h*w, C, C)
            out_product = nd.batch_dot(roi_data, roi_data.transpose((0, 2, 1)))
            # (C, C)
            if self.type == "max":
                reduce_product = nd.max(out_product, axis=0)
                max_mask = out_product == reduce_product
                # max_index = nd.argmax(out_product, axis=0)
                # max_index = max_index.reshape((C * C))
                # d_max = nd.eye(h*w)[max_index].transpose((1, 0)).reshape((h*w, C, C))
                dout_product = nd.stack(*[dout[i] for _ in range(h*w)]) * max_mask
            elif self.type == "mean":
                dout_product = nd.stack(*[dout[i] for _ in range(h*w)]) / (h*w)
            else:
                raise NotImplementedError()

            droi_data = []
            for j in range(C):
                droi_data.append(nd.sum(dout_product[:, j, :] * roi_data[:, :, 0], axis=1) +
                                 nd.sum(dout_product[:, :, j] * roi_data[:, :, 0], axis=1))
            droi_data = nd.stack(*droi_data, axis=1)  # (hw, C)
            droi_data = droi_data.transpose((1, 0)).reshape((C, h, w))
            ddata[batch_id, :, y1:y2, x1:x2] = droi_data
        self.assign(in_grad[0], req[0], ddata)
        self.assign(in_grad[1], req[1], nd.zeros_like(in_data[1]))


@mx.operator.register("BilinearROIPooling")
class BilinearROIPoolingProp(mx.operator.CustomOpProp):
    def __init__(self, spatial_scale, type="max"):
        super().__init__(True)
        self.spatial_scale = spatial_scale
        self.type = type

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        rois_shape = in_shapes[1]
        output_shape = (rois_shape[0], data_shape[1], data_shape[1])
        return (data_shape, rois_shape), (output_shape,), ()

    def list_outputs(self):
        return ["output"]

    def list_arguments(self):
        return ['data', 'rois']

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return BilinearROIPooling(self.spatial_scale, self.type)

