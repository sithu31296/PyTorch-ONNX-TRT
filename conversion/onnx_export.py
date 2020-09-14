import argparse
import onnx
import os 
from onnxsim import simplify
import torch
from models.common import *
from models.yolo import Model


def load_yolo_model(args):
    # although yolov5 weights contain model codes, some operations didn't support in TensorRT
    # so we need to reload the model with modified model file `yolo.py`
    model = Model('models/yolov5l.yaml')
    ckpt = torch.load(args.model_path, map_location=torch.device('cpu'))
    ckpt = {k: v for k, v in ckpt.state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.fuse()

    return model


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=352, help='inference size (pixels)')
    parser.add_argument('--width', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--model-path', type=str, default='weights/yolov5l.pt', help='PyTorch Model Path')
    return parser.parse_args()


if __name__ == '__main__':
    args = argument_parser()

    # load the model (you can export the model to cuda or not)
    model = load_yolo_model(args)
   
    # output filename
    f = args.model_path.replace('.pt', '.onnx')  

    # dummy input (if your model is in cuda, send to cuda if not, leave it as original)
    img = torch.zeros((1, 3, args.height, args.width))

    #out = model(img)[0]  # dry run
    # Export to onnx 
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['image'], output_names=['output'])
    
    # simplify it
    onnx_model = onnx.load(f)  # load onnx model

    # dummy input shape
    input_shapes = {None: [1, 3, args.height, args.width]}

    # simplify it using onnx simplifier
    simplified_model, check = simplify(onnx_model, skip_fuse_bn=True, input_shapes=input_shapes)
    assert check, "Simplified ONNX model could not be validated."
    onnx.save(simplified_model, os.path.splitext(f)[0]+f'_{args.height}_{args.width}.onnx')

    # Check onnx model
    onnx.checker.check_model(simplified_model)  # check onnx model
    #print(onnx.helper.printable_graph(simplified_model.graph))  # print a human readable representation of the graph
    print('Export complete. ONNX model saved to %s\nView with https://github.com/lutzroeder/netron' % f)
