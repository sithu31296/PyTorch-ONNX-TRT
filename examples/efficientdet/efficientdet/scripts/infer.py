import tensorrt as trt 
import numpy as np 
import os 
import cv2
import torch
from efficientdet.scripts.utils import *
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


def preprocess(img, img_size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    normalized_img = (img / 255 - mean) / std 
    framed_img, *framed_meta = aspectaware_resize_padding(normalized_img, img_size, img_size)
    framed_img = framed_img.transpose(2, 0, 1)

    return np.ascontiguousarray(framed_img[np.newaxis, ...]), framed_meta


def postprocess_outputs(pred, anchors, img_size, image, original_img, regressBoxes, clipBoxes, threshold, iou_threshold, framed_meta):
    regression = torch.from_numpy(pred[0].reshape(1, -1, 4))
    classification = torch.from_numpy(pred[1].reshape(1, -1, 90))

    out = postprocess(image, anchors, regression, classification,
                    regressBoxes, clipBoxes, 
                    threshold, iou_threshold)[0]
    
    out = scale_coords(framed_meta, out)
    vis = plot_bbox(out, original_img)

    return vis


class EFFICIENTDET:
    def __init__(self, model_path='cfg/efficientdet-d0.trt'):
        model_type = int(re.search(r'\d+', model_path).group())
        self.img_size = 512
        self.threshold = 0.2
        self.iou_threshold = 0.2
        anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.anchors = anchors_def(anchor_scale=anchor_scale[model_type])

        engine = get_engine(model_path)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine)

    def predict(self, frame):
        #frame = cv2.flip(frame, 0)
        image, framed_meta = preprocess(frame, self.img_size)
        self.inputs[0].host = image
        trt_outputs = do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)
        vis = postprocess_outputs(trt_outputs, self.anchors, self.img_size, image, frame, self.regressBoxes, self.clipBoxes, self.threshold, self.iou_threshold, framed_meta)

        return vis


def main():
    model_type = 0
    model_path = f'cfg/efficientdet-d{model_type}.trt'
    img_size = 512
    threshold = 0.2
    iou_threshold = 0.2
    anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.]

    webcam = WebcamStream()
    fps = FPS()
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    anchors = anchors_def(anchor_scale=anchor_scale[model_type])

    with get_engine(model_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)

        while True:
            fps.start()
            frame = webcam.read()

            image, framed_meta = preprocess(frame, img_size)
            inputs[0].host = image

            trt_outputs = do_inference_v2(context, bindings, inputs, outputs, stream)

            vis = postprocess_outputs(trt_outputs, anchors, img_size, image, frame, regressBoxes, clipBoxes, threshold, iou_threshold, framed_meta)

            fps.stop()
            print(fps.get_fps())

            cv2.imshow('frame', vis)

            if cv2.waitKey(1) == ord("q"):
                webcam.stop()

    
if __name__ == '__main__':
    main()
