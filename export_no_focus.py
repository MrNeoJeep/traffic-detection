"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import numpy as np
import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging, check_img_size


class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = models.common.Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(x)


models.common.Focus = Focus

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov5s.pt',
                        help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    labels = model.names
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples
    img = torch.zeros(opt.batch_size, 3, *opt.img_size[::-1])  # image size(1,3,320,192) iDetection
    img = torch.cat([img[..., ::2, ::2], img[..., 1::2, ::2], img[..., ::2, 1::2], img[..., 1::2, 1::2]], 1)
    # np.save(opt.weights.replace('.pt', f'_{opt.img_size[1]}x{opt.img_size[0]}.npy'), np.array(img, "uint8"))
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = True  # set Detect() layer export=True
    y = model(img)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
