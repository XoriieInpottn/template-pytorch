#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2021-10-11
"""

from docset import DocSet
from imgaug import augmenters as iaa, SegmentationMapsOnImage
from torch.utils.data import Dataset


class SegmentationTransform(object):

    def __init__(self, iaa_fn):
        self._iaa_fn = iaa_fn

    def __call__(self, image, mask=None):
        if mask is None:
            image = self._iaa_fn(image=image)
        else:
            seg_maps = SegmentationMapsOnImage(mask, shape=mask.shape)
            image, seg_maps = self._iaa_fn(image=image, segmentation_maps=seg_maps)
            mask = seg_maps.arr
        return image, mask


class TestTransform(SegmentationTransform):

    def __init__(self, height, width):
        super(TestTransform, self).__init__(iaa.Sequential([
            # todo: example code
            iaa.Resize({'height': height, 'width': width}),
        ]))


class TrainTransform(SegmentationTransform):

    def __init__(self, height, width):
        super(TrainTransform, self).__init__(iaa.Sequential([
            # todo: example code
            iaa.Resize({'height': height, 'width': width}),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ]))


class MyDataset(Dataset):

    def __init__(self, ds_path, transform=None):
        # todo: example code
        self._ds = DocSet(ds_path, 'r')
        self._transform = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        doc = self._ds[i]
        if callable(self._transform):
            # todo: example code
            doc = self._transform(doc['image'], doc['label'])
        return doc
