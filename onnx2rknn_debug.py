import cv2
import numpy as np

from rknn.api import RKNN
import os

ONNX_MODEL = './onnx_models/bisenetv2_fcn_960x540_250k_semantic_segmentation_focal_class6_20231117_skip_postprocess.onnx'
RKNN_MODEL = './onnx_models/bisenetv2_fcn_960x540_250k_semantic_segmentation_focal_class6_20231117_skip_postprocess.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = False
input_imgW = 960
input_imgH = 512

PALETTE = np.array([[255, 0, 0],
                    [36, 255, 0], [118, 89, 0],
                    [156, 0, 255], [255, 0, 0],
                    [255, 120, 0], [255, 255, 0],
                    [0, 108, 71], [111, 39, 0],
                    [18, 0, 255], [0, 0, 0],
                    [255, 255, 255]])


def np_softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), -1)


def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal',
                quantized_method='channel', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    # ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['output0'])
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # # Export RKNN model
    # print('--> Export rknn model')
    # ret = rknn.export_rknn(RKNN_MODEL)
    # if ret != 0:
    #     print('Export rknn model failed!')
    #     exit(ret)
    # print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


if __name__ == '__main__':
    print('This is main ...')
    img_path = './test3.jpg'
    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[:2]

    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)

    img = np.expand_dims(origimg, 0)
    # img = img/255

    result = export_rknn_inference(img)     #[1,64,120,12]

    logits = np_softmax(np.array(result[0][0]))
    # logits[:, :, 0] = logits[:, :, 0] * 0.6
    # logits[:, :, 1] = logits[:, :, 1] * 1.2
    # logits[:, :, 2] = logits[:, :, 2] * 1.2
    mask_map = np.argmax(logits, axis=-1)
    # mask_map = np.argmax(result[0][0], axis=-1)
    mask_map_upsample = cv2.resize(mask_map, (input_imgW, input_imgH), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(args.save_dir + "/" + os.path.basename(fn), mask_map_upsample)
    cv2.imwrite("result.jpg", mask_map_upsample)
    mask_colored = PALETTE[mask_map_upsample]
    # print(mask_colored.shape)
    mask_colored = np.array(mask_colored, dtype=np.uint8)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    #     colored_img=np.array((rawimg+mask_colored)//2,dtype=np.uint8)
    resize_mask_colored = cv2.resize(orig_img, (img_w, img_h))
    colored_img = cv2.addWeighted(orig_img, 0.5, resize_mask_colored, 0.5, 0.0)
    cv2.imwrite("reslut_color.jpg", colored_img)
