import tensorrt as trt 
import numpy as np 
import pycuda.driver as cuda 
import pycuda.autoinit
import common 
import os 
import cv2
from PIL import Image
import time
from threading import Thread

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def preprocess(img, input_resolution):
    image = cv2.resize(img[..., ::-1], input_resolution).transpose(2, 0, 1).astype(np.float32)
    
    image /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = mean[:, np.newaxis, np.newaxis]
    std = std[:, np.newaxis, np.newaxis]
    image = (image - mean) / std
    
    image = np.expand_dims(image, axis=0)
    return np.array(image, dtype=np.float32, order='C')


def postprocess(pred, input_resolution):
    depth = pred.reshape(input_resolution)
    depth = normalize_depth(depth)
    return depth

def normalize_depth(depth):
    depth *= 1000.0
    depth = depth - depth.min()
    depth = (depth / depth.max()) * 255
    #depth = ((depth - depth.min()) / (depth.max() - depth.min())) * 255
    return depth.astype(np.uint8)

class WebcamVideoStream:
    """From PyImageSearch
    Webcam reading with multi-threading
    """
    def __init__(self, src=0, name='WebcamVideoStream'):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.name = name
        self.stopped = False 

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def build_engine(onnx_file_path):
    """
        Takes an ONNX file and creates a TensorRT engine to run inference with.
    """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 256MB
        builder.max_batch_size = 1

        # Parser model file 
        print(f"Loading ONNX file from path {onnx_file_path} ...")
        with open(onnx_file_path, 'rb') as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print(f"ERROR: Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None 
        print(f"Completed parsing of ONNX file.")
        print(f"Building an engine form file {onnx_file_path}; this may take a while ...")
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")

        with open(onnx_file_path.replace('.onnx', '.trt'), 'wb') as f:
            f.write(engine.serialize())
        
        return engine


def get_engine(model_path: str):
    """
        Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it.
    """
    if os.path.exists(model_path):
        if model_path.endswith('trt'):
            print(f"Reading engine from file {model_path}")
            with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())

        elif model_path.endswith('onnx'):
            build_engine(model_path)

        else:
            print("Invalid File: Only .onnx and .trt are supported.")
    else:
        print(f"FILE: {model_path} not found.")


def main():
    model_path = 'weights/bts_nyu_320_mem.trt'
    input_image_path = 'images/NYU0937.jpg'
    input_resolution = (320, 320)

    vs = WebcamVideoStream().start()
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"

    with get_engine(model_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        while True:
            prev_time = time.time()

            frame = vs.read()
            image = preprocess(frame, input_resolution)

            inputs[0].host = image

            trt_outputs = common.do_inference_v2(context, bindings, inputs, outputs, stream)[-1]

            vis = postprocess(trt_outputs, input_resolution)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                print(fps)
                curr_fps = 0

            cv2.imshow('frame', vis)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()
        vs.stop()
    
    #cv2.imwrite('images/trt_output.jpg', depth_image)

if __name__ == '__main__':
    main()