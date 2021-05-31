from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from train import train, eval
import numpy as np

matplotlib.use('agg')

def eval_(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels = list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        # gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    MSE_sum = 0
    AC_cnt = 0
    pred_labels_lost = 0
    gt_labels_lost = 0
    for i in range(test_num):
        MSE_sum_, AC_cnt_, pred_labels_lost_, gt_labels_lost_ = result_makesure(pred_bboxes[i], pred_labels[i], pred_scores[i], gt_bboxes[i], gt_labels[i])
        MSE_sum += MSE_sum_
        AC_cnt += AC_cnt_
        pred_labels_lost += pred_labels_lost_
        gt_labels_lost += gt_labels_lost_
    MES = MSE_sum / (opt.num_class * test_num - gt_labels_lost - pred_labels_lost)
    AC = AC_cnt / (opt.num_class * test_num - gt_labels_lost)

    # print(MSE_sum, AC_cnt, gt_labels_lost, pred_labels_lost)

    return MES, AC


def get_midpoint(bbox):
    x = (bbox[2] + bbox[0]) / 2
    y = (bbox[3] + bbox[1]) / 2
    return  x, y

def get_dist(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def get_dist_lim(gt_bboxes):
    x = list()
    y = list()
    for bbox in gt_bboxes:
        x_, y_ = get_midpoint(bbox)
        x.append(x_)
    return ((max(x) - min(x)) / 15) ** 2

def result_makesure(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels):
    MSE_sum = 0
    AC_cnt = 0
    pred_labels_lost = 0
    gt_labels_lost = 0
    dist_lim = get_dist_lim(gt_bboxes)
    for i in range(opt.num_class):
        pred_bbox = np.zeros((4))
        score_max = 0
        dist_min = opt.min_size * opt.min_size
        if not np.where(gt_labels == i):
            # pred_labels_lost = pred_labels_lost + 1
            gt_labels_lost = gt_labels_lost + 1
            continue
        if not np.where(pred_labels == i):
            pred_labels_lost = pred_labels_lost + 1
            continue
        # jj = np.where(pred_labels == i)[0]
        for j in np.where(pred_labels == i)[0]:
            if pred_scores[j] > score_max:
                score_max = pred_scores[j]
                pred_bbox = pred_bboxes[j]
        if score_max < 0.3:
            pred_labels_lost = pred_labels_lost + 1
            continue
        pred_x, pred_y = get_midpoint(pred_bbox)
        # jj = np.where(gt_labels == i)
        for j in np.where(gt_labels == i)[0]:
            gt_x, gt_y = get_midpoint(gt_bboxes[j])
            if get_dist(pred_x, pred_y, gt_x, gt_y) < dist_min:
                dist_min = get_dist(pred_x, pred_y, gt_x, gt_y)
        MSE_sum += dist_min
        if dist_min <= dist_lim:
            AC_cnt += 1
        
    return MSE_sum, AC_cnt, pred_labels_lost, gt_labels_lost


def test() :
    print("Begin testing")
    print('load data')
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    tester = FasterRCNNTrainer(faster_rcnn).cuda()
    tester.load(opt.load_path)
    if opt.load_path:
        tester.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    MSE, AC = eval_(test_dataloader, faster_rcnn, test_num=opt.test_num)
    print("Average MSE =", MSE)
    print("AC =", AC)
    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
    print(eval_result)


if __name__ == '__main__':
    test()
    
    # import fire

    # fire.Fire(test())