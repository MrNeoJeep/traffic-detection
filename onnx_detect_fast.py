import cv2
import time
import torch
import random
import numpy as np
import onnxruntime
import torchvision


def auto_resize(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    new_size = tuple(map(int, np.array(img.shape[:2][::-1]) * scale))
    return cv2.resize(img, new_size), scale


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    nc = prediction.shape[2] - 5  # number of classes [1, 6552, 48]
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    new_img, scale = auto_resize(img, *new_wh)
    shape = new_img.shape
    new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - shape[0], 0, new_wh[0] - shape[1], cv2.BORDER_CONSTANT,
                                 value=color)
    return new_img, scale


class Detect:
    def __init__(self, model, size, names, anchors=()):  # detection layer
        self.size = size
        nc = len(names)
        self.names = names
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.anchors = a  # shape(nl,na,2)
        self.anchor_grid = a.clone().view(self.nl, 1, -1, 1, 1, 2)  # shape(nl,1,na,1,1,2)
        self.sess = onnxruntime.InferenceSession(model)
        self.input_names = list(map(lambda x: x.name, self.sess.get_inputs()))
        self.output_names = list(map(lambda x: x.name, self.sess.get_outputs()))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(nc)]

    def predict(self, img_src, conf_thres=.4, iou_thres=0.5, draw_box=False):
        img_shape = img_src.shape
        img, scale = letterbox(img_src, self.size)
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR to RGB, to 3x416x416
        img = img[None]
        t0 = time.time()
        print("input img:\t:", img.shape)
        x = self.sess.run(self.output_names, {self.input_names[0]: img})
        print("inference time", time.time() - t0)
        z = []  # inference output
        w = self.size[0]
        for i in range(len(x)):
            x[i] = torch.from_numpy(x[i])
            batch_size, channel_n, ny, nx, predict_n = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            y = x[i].sigmoid()
            # 开始复原xy 和 wh 在输入图中的大小
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            self.grid[i] = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * (w / nx)  # == (h / ny)  xy 计算出预测结果在输入图的xy坐标
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh  计算出对应的wh
            # 此时的结果已经是输入图（416*416）的box了
            z.append(y.view(batch_size, -1, self.no))  # 收集所有预测结果
        out = torch.cat(z, 1)  # 将所有预测结果合并在一起
        pred_res = non_max_suppression(out, 0.4)[0]
        pred_res[:, :4] /= scale
        boxes = pred_res[:, :4]
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
        if draw_box:
            for *xyxy, conf, cls in pred_res:
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box(xyxy, img_src, label=label, color=self.colors[int(cls)], line_thickness=3)
        return pred_res


def test_video(det, video_path):
    reader = cv2.VideoCapture()
    reader.open(video_path)
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        det.predict(frame, draw_box=True)
        cv2.imshow("res", auto_resize(frame, 1200, 600)[0])
        cv2.waitKey(1)


if __name__ == '__main__':
    NC = 43
    SIZE = (416, 256)
    CLASSES = [str(i) for i in range(NC)]
    # anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchors = [[5, 6, 11, 10, 8, 22], [19, 20, 16, 41, 33, 39], [32, 97, 74, 147, 166, 96]]
    d = Detect(r"weights/best_416x256.onnx", SIZE, CLASSES, anchors)
    test_video(d, r"videos/danrentaiti_01.mp4")
    # img = cv2.imread(r"D:\Workspace\test_space_01\yolov5\onnx_test\images\0.jpg")
    # d.predict(img, True)
    # cv2.imshow("src", AutoScale(img, 1200, 600).new_img)
    # cv2.waitKey()
    cv2.destroyAllWindows()
