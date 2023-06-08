import cv2
import numpy as np
import glob
import os
import argparse
import time
import torch
import imutils

import onnx
import onnxruntime as rt

from midas.transforms import Resize, PrepareForNet
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

def init_pytorch_model(model_type):
    
    model_path = os.path.join("weights", model_type + ".pt")

    keep_aspect_ratio = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "dpt_hybrid_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "midas_v21_384":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    elif model_type == "dpt_swin2_tiny_256":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2t16_256",
            non_negative=True,
        )
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    elif model_type == "dpt_levit_224":
        model = DPTDepthModel(
            path=model_path,
            backbone="levit_384",
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )
        net_w, net_h = 224, 224
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.to(device)

    return  model, transform, device


def init_onnx_model(model_type):

    model_path = os.path.join("weights", model_type + ".onnx")

    print(model_path)
    
    if model_type == "dpt_hybrid_384" or model_type == "midas_v21_384": 
        net_w, net_h = 384, 384
    elif model_type == "dpt_swin2_tiny":
        net_w, net_h = 256, 256
    elif model_type == "dpt_levit_224":
        net_w, net_h = 224, 224
    else:
        print(f"Model type '{model_type}' not implemented")
        assert False

    model =  rt.InferenceSession(model_path)

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    resize_image = Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            )

    def compose2(f1, f2):
        return lambda x: f2(f1(x))
    
    transform = compose2(resize_image, PrepareForNet())
    
    return model, net_w, net_h, transform, input_name, output_name

def init_model(backbone, model_type):
    if backbone == "pytorch":
        return init_pytorch_model(model_type)
    return init_onnx_model(model_type)

def estimate_depth_by_pytorch(img, model_params):
    model, transform, device = model_params

    if img.ndim == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    resized_image = transform({"image": img})["image"]

    original_image = original_image.shape[1::-1]
    with torch.no_grad():

        sample = torch.from_numpy(resized_image).to(device).unsqueeze(0)

        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original_image[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    depth_image = postprocess_depth(prediction)

    return depth_image

def estimate_depth_by_onnx(img, model_params):
    model, net_w, net_h, transform, input_name, output_name = model_params

    if img.ndim == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]

    output = model.run([output_name], {input_name: img_input.reshape(1, 3, net_h, net_w).astype(np.float32)})[0]
    prediction = np.array(output).reshape(net_h, net_w)
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    depth_image = postprocess_depth(prediction)
        
    return depth_image

def estimate_depth(panorama, backbone, model_params):
    if backbone == "pytorch":
        return estimate_depth_by_pytorch(panorama, model_params)
    return estimate_depth_by_onnx(panorama, model_params)
    
def postprocess_depth(depth):
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    depth_min_value = depth.min()
    depth_max_value = depth.max()

    if depth_max_value - depth_min_value > np.finfo("float").eps:
        out = 255 * (depth - depth_min_value) / (depth_max_value - depth_min_value)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    return out

def postprocess_stitching(stitched_img):
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    mask = np.zeros(thresh_img.shape, dtype="uint8")
    x, y, w, h = cv2.boundingRect(areaOI)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    minRectangle = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        minRectangle = cv2.erode(minRectangle, None)
        sub = cv2.subtract(minRectangle, thresh_img)

    contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = imutils.grab_contours(contours)
    areaOI = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(areaOI)

    stitched_img = stitched_img[y:y + h, x:x + w]

    return stitched_img

def read_images(input_path):
    images = []
    image_paths = glob.glob(os.path.join(input_path, "*.jpg"))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        images.append(image)
    return images


def run(input_path, output_path, backend, model, threshold=0.94):

    start = time.time()

    model_params = init_model(backend, model)
   
    images = read_images(input_path)

    stitcher = cv2.Stitcher_create()
    error, panorama = stitcher.stitch(images)

    if not error:
        panorama = postprocess_stitching(panorama)
   
        depth_image = estimate_depth(panorama, backend, model_params)
        
        cv2.imwrite(os.path.join(output_path, "panorama.png") , depth_image)

    print(time.time() - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', 
        default='input',
        help='Input images folder'
    )

    parser.add_argument('-o', '--output', 
        default='output',
        help='Output images folder'
    )

    parser.add_argument('-b', '--backend',
        default='onnx',
        help='Backend for depth estimation'
    )

    parser.add_argument('-m', '--model', 
        default='dpt_hybrid_384',
        help='Model type'
    )

    parser.add_argument('-th', '--threshold',
        default='0.94',
        help='Confidence threshold for stitching')

    args = parser.parse_args()

    run(args.input, args.output, args.backend, args.model, args.threshold)