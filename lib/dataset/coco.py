from gluoncv import data as gdata
import os


class COCODataset(gdata.COCODetection):

    def __init__(self, root, annFile, transfom=None, mini_object_area=0, skip_empty=True, use_crowd=True):
        self._annFile = annFile
        self._classes = None
        gdata.mscoco.utils.try_import_pycocotools()
        from pycocotools.coco import COCO
        _coco = COCO(self._annFile)
        self._coco_ = _coco
        classes = [c["name"] for c in _coco.loadCats(_coco.getCatIds())]
        self.CLASSES = classes
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

