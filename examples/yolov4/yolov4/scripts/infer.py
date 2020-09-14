import tensorrt as trt 
import numpy as np 
import os 
import cv2
import torch
from yolov4.scripts.utils import *
#from utils import *
import re

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def get_engine(model_path: str):
    if os.path.exists(model_path) and model_path.endswith('trt'):
        print(f"Reading engine from file {model_path}")
        with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print(f"FILE: {model_path} not found or extension not supported.")


def preprocess(img, img_size):
    image = letterbox(img, new_shape=img_size[-1])[0]
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = image[np.newaxis, ...]
    image /= 255.0
    return np.ascontiguousarray(image)


def postprocess(pred, img_size, original_img):
    pred = pred.reshape(1, -1, 85)
    output = non_max_suppression(torch.from_numpy(pred), conf_thres=0.2, iou_thres=0.2)[0]

    #for det in output:
    if output is not None and len(output):
        output[:, :4] = scale_coords(img_size, output[:, :4], original_img.shape[:2]).round()

        for *xyxy, conf, cls in output:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, original_img, label=label, color=colors[int(cls)], line_thickness=2)

    return original_img


class YOLOV4:
    def __init__(self, model_path='cfg/yolov4_512_640.trt'):
        self.img_size = (352, 416)
        engine = get_engine(model_path)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine)

    def predict(self, frame):
        image = preprocess(frame, self.img_size)
        self.inputs[0].host = image
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)[-1]
        vis = postprocess(trt_outputs, self.img_size, frame)

        return vis


def main():
    model_path = 'cfg/yolov4_512_640.trt'
    img_size = (512, 640)

    webcam = WebcamStream()
    fps = FPS()

    engine = get_engine(model_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    while True:
        fps.start()
        frame = webcam.read()

        image = preprocess(frame[..., ::-1], img_size)
        inputs[0].host = image

        trt_outputs = do_inference_v2(context, bindings, inputs, outputs, stream)[-1]
        vis = postprocess(trt_outputs, img_size, frame)

        fps.stop()
        print(fps.get_fps())

        cv2.imshow('frame', vis)

        if cv2.waitKey(1) == ord("q"):
            webcam.stop()

if __name__ == '__main__':
    main()
