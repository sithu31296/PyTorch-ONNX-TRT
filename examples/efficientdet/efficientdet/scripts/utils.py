import itertools
import torch 
import torch.nn as nn
import numpy as np 
from torchvision.ops import nms
import os 
import pycuda.driver as cuda 
import pycuda.autoinit
import tensorrt as trt
from threading import Thread
import time
import cv2
import random

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(obj_list))]


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def anchors_def(anchor_scale, image_shape=(512, 512), dtype=torch.float32):
    pyramid_levels = [3, 4, 5, 6, 7]
    strides = [2 ** x for x in pyramid_levels]
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    boxes_all = []
    for stride in strides:
        boxes_level = []
        for scale, ratio in itertools.product(scales, ratios):
            base_anchor_size = anchor_scale * stride * scale
            anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
            anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

            x = torch.arange(stride / 2, image_shape[1], stride)
            y = torch.arange(stride / 2, image_shape[0], stride)
            xv, yv = torch.meshgrid(x, y)
            xv, yv = xv.t().reshape(-1), yv.t().reshape(-1)

            # y1,x1,y2,x2
            boxes = torch.stack((yv - anchor_size_y_2, xv - anchor_size_x_2, yv + anchor_size_y_2, xv + anchor_size_x_2))
            
            boxes_level.append(boxes.transpose(0, 1).unsqueeze(1))
            
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = torch.cat(boxes_level, dim=1)
        boxes_all.append(boxes_level.reshape(-1, 4))
    
    anchor_boxes = torch.cat(boxes_all, dim=0).type(dtype).unsqueeze(0)

    return anchor_boxes


def aspectaware_resize_padding(image, width, height):
    old_h, old_w, c = image.shape

    if old_w > old_h:
        new_w, new_h = width, int(width / old_w * old_h)
    else:
        new_w, new_h = int(height / old_h * old_w), height

    canvas = np.zeros((height, height, c), np.float32)

    if new_w != old_w or new_h != old_h:
        image = cv2.resize(image, (new_w, new_h))
        
    padding_h = height - new_h
    padding_w = width - new_w

    canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def scale_coords(metas, preds):
    if len(preds['rois']) == 0:
        return preds
    new_w, new_h, old_w, old_h, padding_w, padding_h = metas
    preds['rois'][:, [0, 2]] = preds['rois'][:, [0, 2]] / (new_w / old_w)
    preds['rois'][:, [1, 3]] = preds['rois'][:, [1, 3]] / (new_h / old_h)

    return preds


def plot_bbox(preds, img):
    if len(preds['rois']) == 0:
        return img

    for j in range(len(preds['rois'])):
        (x1, y1, x2, y2) = preds['rois'][j].astype(np.int)
        color = colors[int(preds['class_ids'][j])]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
        obj = obj_list[preds['class_ids'][j]]
        score = float(preds['scores'][j])
        label = f'{obj}, {score:.3f}'
        t_size = cv2.getTextSize(label, 0, 2/3, 1)[0]
        cv2.rectangle(img, (x1, y1), (x1+t_size[0], y1-t_size[1]-3), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1-2), 0, 2/3, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img


class WebcamStream:
    def __init__(self, src=1):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        cap.set(3, 640)
        cap.set(4, 480)
        assert cap.isOpened(), f"Failed to open {src}"
        _, self.frame = cap.read()
        
        Thread(target=self.update, args=([cap]), daemon=True).start()

    def update(self, cap):
        while cap.isOpened():
            cap.grab()
            _, self.frame = cap.retrieve()

    def read(self):
        return self.frame.copy()

    def stop(self):
        cv2.destroyAllWindows()
        raise StopIteration
         

class FPS:
    def __init__(self):
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"

    def start(self):
        self.prev_time = time.time()

    def stop(self):
        self.curr_time = time.time()
        exec_time = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        self.accum_time += exec_time

    def get_fps(self):
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        return self.fps


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
