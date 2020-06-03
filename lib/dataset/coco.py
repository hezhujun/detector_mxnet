from gluoncv import data as gdata
import os


class COCODataset(gdata.COCODetection):
    """
    gluoncv提供的COCODetection只支持coco数据集，而且遵循coco数据集的目录结构，不方便扩展到其他数据集
    该类扩展COCODetection，支持任意的coco类型的数据集
    因为CLASSES是绑定类，而不是对象，当新建一个COCODataset数据集对象时，
    COCODataset的类属性CLASSES会修改成载入annFile中的classes
    所以同一个程序中不要创建classes不同的COCODataset数据集对象
    gluoncv的代码有时会调用dataset.CLASSES，有时会调用dataset.classes
    因此，COCODataset和COCODetection一样，把数据集的classes存放到类属性CLASSES中
    """

    def __init__(self, root, annFile, transfom=None, mini_object_area=0, skip_empty=True, use_crowd=True):
        self._annFile = annFile
        self._classes = None
        gdata.mscoco.utils.try_import_pycocotools()
        from pycocotools.coco import COCO
        _coco = COCO(self._annFile)
        self._coco_ = _coco
        classes = [c["name"] for c in _coco.loadCats(_coco.getCatIds())]
        type(self).CLASSES = classes
        super(COCODataset, self).__init__(root, ("",), transfom, mini_object_area, skip_empty, use_crowd)
        self._coco.append(_coco)

    def _load_jsons(self):
        items = []
        labels = []
        im_aspect_ratios = []
        _coco = self.coco
        json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())
        }
        self.json_id_to_contiguous = json_id_to_contiguous
        self.contiguous_id_to_json = {
            v: k for k, v in self.json_id_to_contiguous.items()
        }

        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            filename = entry["file_name"]
            abs_path = os.path.join(self._root, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_bbox(_coco, entry)
            if not label:
                continue
            im_aspect_ratios.append(float(entry["width"]) / entry["height"])
            items.append(abs_path)
            labels.append(label)
        return items, labels, im_aspect_ratios

    @property
    def coco(self):
        return self._coco_

