#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import os.path
from os import path
import time
from loguru import logger

import importlib
import sys

import cv2
import numpy as np

import torch
from tabulate import tabulate

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import boxes, fuse_model, get_model_info, postprocess, vis
from map.script.map import map_score

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

DSLAB20 = [31.78,98.08,88.51,98.26,97.46,99.61,97.26,98.26,95.94,94.08,99.93,97.50]

DATASETS = ["Hempbox",
            "Chueried_Hive01",
            "ClemensRed",
            "Echolinde",
            "Erlen_diago",
            "Erlen_front",
            "Erlen_smart",
            "Erlen_Hive11",
            "Froh14",
            "Froh23",
            "UnitedQueens",
            "Doettingen_Hive1"]

def get_exp_by_file(exp_file,data_dir):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp(data_dir)
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_by_name(exp_name,data_dir):
    import yolox

    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    filedict = {
        "yolox-s": "yolox_s.py",
        "yolox-m": "yolox_m.py",
        "yolox-l": "yolox_l.py",
        "yolox-x": "yolox_x.py",
        "yolox-tiny": "yolox_tiny.py",
        "yolox-nano": "nano.py",
        "yolov3": "yolov3.py",
    }
    filename = filedict[exp_name]
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name,data_dir):
    """
    get Exp object by file or name. If exp_file and exp_name
    are both provided, get Exp by exp_file.

    Args:
        exp_file (str): file path of experiment.
        exp_name (str): name of experiment. "yolo-s",
    """
    assert (
        exp_file is not None or exp_name is not None
    ), "plz provide exp file or exp name."
    if exp_file is not None:
        return get_exp_by_file(exp_file,data_dir)
    else:
        return get_exp_by_name(exp_name,data_dir)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.exp = exp
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        #print(self.test_size)
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, dataset,cls_conf=0.35,):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            file_path = str("map/input/" + str(dataset) + "/detection-results/" + str(img_info["file_name"])[:-4] + ".txt")
            f= open(file_path,"w+")
            f.close()
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        #print(bboxes)
        bboxes /= ratio
        #print(bboxes)

        pred_path = str("map/input/" + str(dataset) + "/detection-results/")
        path_dont_exist = not(os.path.isdir(pred_path))
        if path_dont_exist:
            os.makedirs(pred_path, exist_ok=True)
        file_path = str("map/input/" + str(dataset) + "/detection-results/" + str(img_info["file_name"])[:-4] + ".txt")

        f= open(file_path,"w+")
        
        #print(annotations/img_info["ratio"])
        bboxes1 = []
        for i,obj in enumerate(bboxes):
            x1 = np.max((0, obj[0]))
            y1 = np.max((0, obj[1]))
            x2 = obj[2]
            y2 = obj[3]
            #x2 = img_info["width"] - np.max((0, obj[0]))
            
            #y2 = img_info["height"] - np.max((0, obj[1])) 
            #x1 = img_info["width"] - obj[2]
            
            #y1 = img_info["height"] - obj[3]
            if x2 >= x1 and y2 >= y1:
                bboxes1.append([x1, y1, x2, y2])
            
            f.write(str(int(output[i,6].item())) + " " + str(output[i,5].item()) + " " + str(x1.item()) + " " + str(y1.item()) + " " + str(x2.item()) + " " + str(y2.item()) + "\n")
        
        f.close()

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes1, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(exp,path, current_time, save_result,dataset,test_size):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    val_loader = exp.get_eval_loader(
            batch_size=4,
            is_distributed=False,
        )
    ground_truth_path = "map/input/" + str(dataset) 
    ground_truth_path = os.path.join(ground_truth_path,"ground-truth/")
    path_dont_exist = not(os.path.isdir(ground_truth_path))
    if path_dont_exist:
        os.makedirs(ground_truth_path, exist_ok=True)
    for i,image_name in enumerate(files):
        file_name = os.path.basename(image_name)
        img = cv2.imread(image_name)
        ratio = min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
        if path_dont_exist:
            file_path = str(ground_truth_path + str(file_name)[:-4] + ".txt")
            f1= open(file_path,"w+")
            annotations = exp.valdataset.annotations[i][0]
            #annotations = exp.dataset._dataset.annotations[i][0]
            annotations /= ratio
            for i,obj in enumerate(annotations):    
                f1.write(str(int(obj[4].item())) + " "  + str(obj[0].item()) + " " + str(obj[1].item()) + " " + str(obj[2].item()) + " " + str(obj[3].item()) + "\n")
       


def main(args):
    table = []
    exp = get_exp(args.exp_file, args.name, str("datasets/" + str(DATASETS[0]) + "/"))
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    current_time = time.localtime()
    work_dir = os.getcwd()
    for i,dataset in enumerate(DATASETS):
        os.chdir(work_dir)
        exp = get_exp(args.exp_file, args.name, str("datasets/" + str(dataset) + "/"))
        exp.test_size = (args.tsize, args.tsize)
        if args.demo == "image":
            path = str("datasets/" + str(dataset) + "/validate")
            image_demo(exp,path, current_time, args.save_result,dataset,exp.test_size)
        
        score = map_score(dataset,args,work_dir)
        score.append(DSLAB20[i])
        if(float(score[1][:-1]) > score[2]):
            score.append("BETTER")
        else:
            score.append("WORSE")
        table.append(score)
    os.chdir(work_dir)
    print(tabulate(table[:-1], headers=["Dataset","mAP Score","DSLab20","Comparison"]))
    file_path = str("map/output/mAP_results.txt")
    f= open(file_path,"w+")
    f.write(tabulate(table[:-1], headers=["Dataset","mAP Score","DSLab20","Comparison"]))
    f.close()



if __name__ == "__main__":
    args = make_parser().parse_args()
    #exp = get_exp(args.exp_file, args.name)

    main(args)
