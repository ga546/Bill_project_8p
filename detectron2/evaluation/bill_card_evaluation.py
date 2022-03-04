# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import json
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator


class BillCardDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # # Too many tiny files, download all to local for speed.
        # annotation_dir_local = PathManager.get_local_path(
        #     os.path.join(meta.dirname, "Annotations/")
        # )
        self._anno_file_template = os.path.join(os.path.join(meta.dirname, "val/jsons/"), "{}.json") # 读取标注文件json类型
        self._image_set_path = os.path.join(meta.dirname, "val", meta.split + ".txt")  # 数据集补充val图片名称 val.txt
        self._class_names = meta.thing_classes
        # assert meta.year in [2007, 2012], meta.year
        # self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):   #输出检测结果的而处理
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device) #需要修改吗？
            boxes = instances.pred_points.numpy()  #修改
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):  #修改
                x1, y1, x2, y2, x3, y3, x4, y4 = box
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)     #处理后的检测结果，输出一下
        del all_predictions

        self._logger.info(
            "Evaluating {}".format(
                self._dataset_name
            )
        )
        with tempfile.TemporaryDirectory(prefix="bill_card_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            for cls_id, cls_name in enumerate(self._class_names):
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))  # 将检测结果存入到临时文件中去:按照类别分开，即类别数=文件数。

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                    )
                    aps[thresh].append(ap * 100)
        # 各个类别的AP:
        print("train_ticket:")
        print("AP50:", aps[50][0], "AP75:", aps[75][0], "AP:")
        print("bank_card:")
        print("AP50:", aps[50][1], "AP75:", aps[75][1], "AP:")
        print("bill_card:")
        print("AP50:", aps[50][2], "AP75:", aps[75][2], "AP:")
        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}
        print("ret:",ret)
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
# def parse_rec(filename):
#     """Parse a PASCAL VOC xml file."""
#     with PathManager.open(filename) as f:
#         tree = ET.parse(f)
#     objects = []
#     for obj in tree.findall("object"):
#         obj_struct = {}
#         obj_struct["name"] = obj.find("name").text
#         obj_struct["pose"] = obj.find("pose").text
#         obj_struct["truncated"] = int(obj.find("truncated").text)
#         obj_struct["difficult"] = int(obj.find("difficult").text)
#         bbox = obj.find("bndbox")
#         obj_struct["bbox"] = [
#             int(bbox.find("xmin").text),
#             int(bbox.find("ymin").text),
#             int(bbox.find("xmax").text),
#             int(bbox.find("ymax").text),
#         ]
#         objects.append(obj_struct)
#
#     return objects
def parse_rec(filename):
    """Parse a json file."""
    data2 = []
    dict2 = {}
    data1 = json.load(open(filename))
    for dict1 in data1["shapes"]:
        dict2["name"] = dict1["label"]
        dict2["difficult"] = 0
        dict2["bbox"] = np.array(dict1["points"]).reshape(-1).tolist()
        data2.append(dict2)
    return data2


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]    # 所有的图像名称列表
    #print("imagenames",imagenames)
    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename)) # 对输入进行处理
    #print("recs",recs)
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname] # R:list(dict())
        bbox = np.array([x["bbox"] for x in R])   # 检测框参数，应为8
        # difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read dets                                                                     #从此处开始修改2022-3-02
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    #print(splitlines)
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 8)  # 目标
    # sort by confidence
    sorted_ind = np.argsort(-confidence)  # 将list中的元素从小到大排列，提取其对应的index(索引)并输出
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]  # 按照confidence排序好的BB和image_ids
    #print("class_recs",class_recs)
    #print("image_ids",image_ids)
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    #print(type(class_recs))
    #print()
    for d in range(nd):
        R = class_recs[imagenames[int(image_ids[d])]]
        bb = BB[d, :].astype(float)          #bb:array数组类型
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)       #2维array类型
        overlaps = []
        if BBGT.size > 0:
            for idx in range(BBGT.shape[0]):
                overlaps.append(compute_iou_test(bb,BBGT[idx, :]))
            overlaps = np.array(overlaps)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)  # 返回最大值得索引值

        # if BBGT.size > 0:
        #     # compute overlaps
        #     # intersection
        #     ixmin = np.maximum(BBGT[:, 0], bb[0])
        #     iymin = np.maximum(BBGT[:, 1], bb[1])
        #     ixmax = np.minimum(BBGT[:, 2], bb[2])
        #     iymax = np.minimum(BBGT[:, 3], bb[3])
        #     iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        #     ih = np.maximum(iymax - iymin + 1.0, 0.0)
        #     inters = iw * ih
        #
        #     # union
        #     uni = (
        #         (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
        #         + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
        #         - inters
        #     )
        #
        #     overlaps = inters / uni
        #     ovmax = np.max(overlaps)
        #     jmax = np.argmax(overlaps)  # 返回最大值得索引值

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def compute_iou_test(array1, array2):
    a = np.array(array1).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull
    b = np.array(array2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            #print(inter_area)
            union_area = poly1.area + poly2.area - inter_area
            #print(union_area)
            if union_area == 0:
                iou = 0
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            #print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou