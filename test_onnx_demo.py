import argparse
import os
import sys
import os.path as osp
import cv2
import torch
import numpy as np
import onnxruntime as ort
from math import exp
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
import random
import os
import cv2
import argparse
import glob
import multiprocessing
import copy
import pdb

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
FLOAT = False

PALETTE = np.array([[255, 0, 0],
                    [36, 255, 0], [118, 89, 0],
                    [156, 0, 255], [255, 0, 0],
                    [255, 120, 0], [255, 255, 0],
                    [0, 108, 71], [111, 39, 0],
                    [18, 0, 255], [0, 0, 0],
                    [255, 255, 255]])


def np_softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), -1)


def inference(fn):
    img = cv2.imread(fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     rawimg=copy.deepcopy(img)
    # img = cv2.cvtColor(rawimg, cv2.COLOR_BGR2YUV)
    if FLOAT:
        img[:, :, 0] = (img[:, :, 0] - 123.675) * 0.01712
        img[:, :, 1] = (img[:, :, 1] - 116.28) * 0.0175
        img[:, :, 2] = (img[:, :, 2] - 103.53) * 0.01743
        img_tensor = np.expand_dims(np.transpose(cv2.resize(img, (960, 512)), [2, 0, 1]), axis=0)
        img_tensor = (img_tensor / 255).astype(np.float32)
        offset = 0
    else:
        offset = 128
        # img_tensor = np.expand_dims(np.asarray(cv2.resize(img, (960, 512))), axis=0)
        img_tensor = np.expand_dims(np.transpose(cv2.resize(img, (960, 512)), [2, 0, 1]), axis=0).astype(np.float32)
        img_tensor = (img_tensor / 255)

    ort_session = ort.InferenceSession(args.model)
    result = ort_session.run(None, {'input': img_tensor})
    logits = np_softmax(np.array(result[0][0]))
    mask_map = np.argmax(logits, axis=-1)
    mask_map_upsample = cv2.resize(mask_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(args.save_dir + "/" + os.path.basename(fn), mask_map_upsample)
    mask_colored = PALETTE[mask_map_upsample]
    # print(mask_colored.shape)
    mask_colored = np.array(mask_colored, dtype=np.uint8)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    #     colored_img=np.array((rawimg+mask_colored)//2,dtype=np.uint8)
    colored_img = cv2.addWeighted(img, 0.5, mask_colored, 0.5, 0.0)
    cv2.imwrite(args.save_dir + "_color/" + os.path.basename(fn), colored_img)
    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp",
                        default="/home/youfeng/project/CLionProjects/rockchip/01-api/v1.5.2/rknpu2/examples/rknn_yolov5_demo/convert_rknn_demo/unet/obstacle_test")
    parser.add_argument("--model",
                        default="/home/youfeng/project/CLionProjects/rockchip/01-api/v1.5.2/rknpu2/examples/rknn_yolov5_demo/convert_rknn_demo/unet/onnx_models/bisenetv2_fcn_960x540_250k_semantic_segmentation_focal_class6_20231117_skip_postprocess.onnx")
    parser.add_argument("--save-dir", default="./obstacle_result")
    args = parser.parse_args()
    MAX_PROC = 12
    fns = glob.glob(args.fp + "/*.jpg")
    inference(fns[0])
    # ort_session = ort.InferenceSession(model_file=args.model)
    # ort_session = ort.InferenceSession("./onnx_models/bisenetv2_fcn_960x540_250k_semantic_segmentation_focal_class6_20231117_skip_postprocess.onnx")
    # result = (ort_session.run(None, {'images': fns[0]}))
    # result = inference(ort_session, fns[0])
    # with multiprocessing.Pool(processes=MAX_PROC) as pool:
    #     pool.map(inference,fns)
