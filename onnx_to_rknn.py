import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

""""
将onnx模型转换为rknn模型
"""

if __name__ == '__main__':
    ONNX_MODEL = 'yolov5m_640x640.onnx'
    RKNN_MODEL = 'yolov5m_640x640.rknn'

    # Create RKNN object
    rknn = RKNN()
    print('--> config model')
    # rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.82, 58.82, 58.82]], reorder_channel='0 1 2')
    # rknn.config(batch_size=1,target_platform=["rk1806", "rk1808", "rk3399pro"], mean_values='0 0 0 255')
    rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2', batch_size=1)
    # rknn.config(channel_mean_value='0 0 0 1', reorder_channel='0 1 2', batch_size=1)
    # rknn.config(mean_values=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], std_values=[[255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]], reorder_channel='0 1 2', batch_size=1)
    print('done')
 
    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load resnet50v2 failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')  # pre_compile=True
    # ret = rknn.build(do_quantization=True)  # pre_compile=True
    if ret != 0:
        print('Build resnet50 failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export resnet50v2.rknn failed!')
        exit(ret)
    print('done')
    rknn.release()

