
import cv2
import numpy as np

from rknn.api import RKNN
import os

if __name__ == '__main__':

    platform = 'rk3588'
    exp = 'bisenetv2'
    Width = 960
    Height = 512
    # Model from https://github.com/airockchip/rknn_model_zoo
    # MODEL_PATH = './onnx_models/unet_mobilenet_1024x2048_nv12_original_float_model.onnx'
    # MODEL_PATH = './onnx_models/tf_unet_trained.onnx'
    MODEL_PATH = './onnx_models/bisenetv2_fcn_960x540_250k_semantic_segmentation_focal_class6_20231117_skip_postprocess.onnx'
    # MODEL_PATH = '/home/youfeng/project/CLionProjects/04_horizon/12_horizon/ModelZoo-master/DeeplabV3Plus/deeplabv3plus_efficientnetb0.onnx'
    NEED_BUILD_MODEL = True
    # NEED_BUILD_MODEL = False
    im_file = './frankfurt_000000_000294_leftImg8bit.png'

    # Create RKNN object
    rknn = RKNN()

    OUT_DIR = "rknn_models"
    RKNN_MODEL_PATH = './{}/{}_{}.rknn'.format(
        OUT_DIR, exp+'-'+str(Width)+'-'+str(Height), platform)
    if NEED_BUILD_MODEL:
        DATASET = './dataset.txt'
        rknn.config(mean_values=[[0, 0, 0]], std_values=[
                    [255, 255, 255]], target_platform=platform)
        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(MODEL_PATH)
        if ret != 0:
            print('load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=DATASET)
        if ret != 0:
            print('build model failed.')
            exit(ret)
        print('done')

        # Export rknn model
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
        print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export rknn model failed.')
            exit(ret)
        print('done')
    else:
        ret = rknn.load_rknn(RKNN_MODEL_PATH)
        data = rknn.rknn_base.inputs_meta["attrs"]
        for key in data:
            print(key, "-->", data[key])

        print("test load rknn model api")

    rknn.release()
