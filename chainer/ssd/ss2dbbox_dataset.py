# VOCBboxDatasetをカスタムしたもの

ss2d_bbox_label_names = ('2dface')

import argparse
import copy
import glob
import numpy as np
import pickle

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation
from chainercv.chainer_experimental.datasets.sliceable import SliceableDataset
import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
class SS2DBboxDataset(GetterDataset):

    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox` [#voc_bbox_1]_, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label` [#voc_bbox_1]_, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`difficult` (optional [#voc_bbox_2]_), ":math:`(R,)`", \
        :obj:`bool`, --

    .. [#voc_bbox_1] If :obj:`use_difficult = True`, \
        :obj:`bbox` and :obj:`label` contain difficult instances.
    .. [#voc_bbox_2] :obj:`difficult` is available \
        if :obj:`return_difficult = True`.
    """

    def __init__(self, data_dir='auto', split='train',
                 use_difficult=False, return_difficult=False):
        super(SS2DBboxDataset, self).__init__()
        
        id_list_file = os.path.join(
            data_dir, 'Ids/{0}.txt'.format(split))
        
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        id_ = self.ids[i]
        img_path = os.path.join(self.data_dir, 'Images', id_ )
       # print(img_path)
        img = read_image(img_path, color=True)
        return img

    def _get_annotations(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(0)
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(0)
        if len(label)>0:
            bbox = np.stack(bbox).astype(np.float32)
            label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)
        return bbox, label, difficult

def read_image(path, dtype=np.float32, color=True):
    with open(path,mode='rb') as f:
        img = pickle.load(f)
        f.close
        return img.astype(np.float32)