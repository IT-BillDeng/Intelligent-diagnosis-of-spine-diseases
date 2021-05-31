import os
import xml.etree.ElementTree as ET
# from data.Load_data import get_list, read_txt
from data.Load_data import get_list, read_txt
import numpy as np

from .util import read_image


class SDLBboxDataset:

    def __init__(self, data_dir, split='train',
                 use_difficult=False, return_difficult=False,
                 ):
        
        PATH = os.path.join(data_dir, '{0}/data/'.format(split))
        # print(PATH)
        # self.img_list, self.txt_list = get_list(PATH)
        self.ids = get_list(PATH)
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = PATH
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = SDL_BBOX_LABEL_NAMES


    def __len__(self):
        return len(self.ids)
        # return self.N

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        txt_file = self.data_dir + id_ + '.txt'
        bbox, label = read_txt(txt_file)
        # difficult = np.zeros(len(bbox))

        # Load a image
        img_file = self.data_dir + id_ + '.jpg'
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        # print(img.shape)
        # print(bbox.shape)
        # print(label.shape)
        return img, bbox, label

    __getitem__ = get_example



SDL_BBOX_LABEL_NAMES = (
    'L1',
    'L2',
    'L3',
    'L4',
    'L5',
    'T12-L1',
    'L1-L2',
    'L2-L3',
    'L3-L4',
    'L4-L5',
    'L5-S1')