import gluoncv as gcv
import numpy as np
from mxnet import ndarray as nd


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


def get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    # lr factor 线性增长函数
    # 最小为lr_warmup_factor
    # 最大为1
    return lr_warmup_factor * (1 - alpha) + alpha


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