# -*- coding:utf-8 -*-
# @Author: Li Hui, Jiangnan University
# @Email: lihui.cv@jiangnan.edu.cn
# @Project: CrossFuse
# @File: test_color_image
# @Time: 2023/3/9 15:27


import os
import cv2
import torch
import numpy as np
from torch.autograd import Variable
from network.net_autoencoder import Auto_Encoder_single
from network.net_conv_trans import Trans_FuseNet
from ultralytics.utils.nms import non_max_suppression
from tools import utils
from args_trans import Args as args

k = 10

CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "van",
    "freight_car"
]

DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
# imagenet_labels = dict(enumerate(open("classes.txt")))


def load_model(custom_config_auto, custom_config_trans, model_path_auto_ir, model_path_auto_vi, model_path_trans):
    model_auto_ir = Auto_Encoder_single(**custom_config_auto)
    model_auto_ir.load_state_dict(torch.load(model_path_auto_ir))
    model_auto_ir.to(DEVICE)
    model_auto_ir.eval()
    
    model_auto_vi = Auto_Encoder_single(**custom_config_auto)
    model_auto_vi.load_state_dict(torch.load(model_path_auto_vi))
    model_auto_vi.to(DEVICE)
    model_auto_vi.eval()
    # ---------------------------------------------------------
    model_trans = Trans_FuseNet(**custom_config_trans)
    model_trans.load_state_dict(torch.load(model_path_trans))
    model_trans.to(DEVICE)
    model_trans.eval()
    return model_auto_ir, model_auto_vi, model_trans

"""
Function to decode the detections from the model output into bounding boxes and class labels
"""

def decode_detections(detections, image_width, image_height,
                      threshold=0.25, iou_threshold=0.45):
    # TransFuse returns (prediction_tensor, auxiliary_dict).
    if isinstance(detections, (tuple, list)):
        detections = detections[0]

    # Expected shape: [batch, 4 + num_classes, num_predictions].
    if detections.ndim != 3:
        raise ValueError(
            f"Expected [B, C, N] predictions, got {detections.shape}"
        )

    # Ultralytics NMS expects this raw prediction format and performs
    # xywh-to-xyxy conversion, confidence filtering, and NMS.
    results = non_max_suppression(
        detections,
        conf_thres=threshold,
        iou_thres=iou_threshold,
        nc=len(CLASS_NAMES),
    )

    boxes = []
    for det in results:
        for x1, y1, x2, y2, confidence, class_id in det.tolist():
            class_id = int(class_id)

            # Scale boxes if the detector's input size differs from
            # the image on which the boxes will be drawn.
            x1 = max(0, min(int(x1), image_width - 1))
            y1 = max(0, min(int(y1), image_height - 1))
            x2 = max(0, min(int(x2), image_width - 1))
            y2 = max(0, min(int(y2), image_height - 1))

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                continue

            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            })

    return boxes

def test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path,
         output_path_fea):

    ir_img = utils.get_train_images(ir_path, None, None, flag=False)
    vi_img, vi_cb, vi_cr = utils.get_test_images_color(vi_path, None, None, flag=img_flag)
    ir_img = Variable(ir_img, requires_grad=False)
    vi_img = Variable(vi_img, requires_grad=False)
    # if args.cuda:
    ir_img = ir_img.to(DEVICE)
    vi_img = vi_img.to(DEVICE)

    # ---------------------------------------------
    # outputs = model.reconsturce(ir_img, vi_img)
    ir_sh, ir_de = model_auto_ir(ir_img)
    vi_sh, vi_de = model_auto_vi(vi_img)
    outputs = model_trans(ir_de, ir_sh, vi_de, vi_sh, shift_flag)

    print("Fused feature shape:", outputs["fused_features"].shape)
    detections = outputs["detections"]
    print("Detection output type:", type(detections))
    if isinstance(detections, (tuple, list)):
        for i, item in enumerate(detections):
            if torch.is_tensor(item):
                print(f"Detection output [{i}]: shape={item.shape}")
            else:
                print(f"Detection output [{i}]: type={type(item)}")
    elif torch.is_tensor(detections):
        print("Detection output shape:", detections.shape)
    
    # Decode and display bounding boxes on the original visible image
    fused = outputs["out"].detach().float().cpu()[0, 0]

    # Already in the 0–255 range.
    img_out = fused.clamp(0, 255).numpy().astype(np.uint8)

    # Convert grayscale to 3-channel BGR for drawing boxes.
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)

    image_height, image_width = img_out.shape[:2]

    boxes = decode_detections(
        outputs['detections'],
        image_width,
        image_height,
        threshold=0.5
    )
    print("Number of detections:", len(boxes))    
    for box in boxes:
        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]
        confidence = box["confidence"]
        class_name = box["class_name"]

        cv2.rectangle(
            img_out,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        label = "{} {:.2f}".format(class_name, confidence)
        cv2.putText(
            img_out,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    cv2.imwrite(os.path.join(output_path, "det_" + ir_name), img_out)
    # ir_self = outputs['ir_self']
    # vi_self = outputs['vi_self']
    # fuse_cross = outputs['fuse_cross']
    # # ---------------------------------------------
    
    # ---------------------------------------------
    # path_out = output_path + '/results_crossfuse_'
    # path_out_fea = output_path_fea + '/result_crossfuse_'
    # utils.save_image_color(img_out, vi_cb, vi_cr, path_out + ir_name)
    # utils.save_image(ir_self, path_out_fea + 'irself_' + ir_name)
    # utils.save_image(vi_self, path_out_fea + 'viself_' + ir_name)
    # utils.save_image(fuse_cross, path_out_fea + 'cross_' + ir_name)
    
    print('Done. ', ir_name)


if __name__ == "__main__":
    # Auto-Encoder
    custom_config_auto = {
        "in_channels": 1,
        "out_channels": 1,
        "en_out_channels1": 32,
        "en_out_channels": 64,
        "num_layers": 3,
        "dense_out": 128,
        "part_out": 128,
        "train_flag": False,
    }
    # Trans module
    custom_config_trans = {
        "en_out_channels1": 32,
        "out_channels": 1,
        "part_out": 128,
        "train_flag": False,
        
        "img_size": 32,
        "patch_size": 2,
        "depth_self": 1,
        "depth_cross": 1,
        "n_heads": 16,
        "qkv_bias": True,
        "mlp_ratio": 4,
        "p": 0.,
        "attn_p": 0.,
    }
    
    resume_model_auto_ir = "./models/autoencoder/auto_encoder_epoch_4_ir.model"
    resume_model_auto_vi = "./models/autoencoder/auto_encoder_epoch_4_rgb.model"
    
    # model_path_auto = "./models/autoencoder/auto_encoder_epoch_3.model"
    # model_path_trans = "./models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model"
    model_path_trans = "./models/transfuse/fusetrans_epoch_10.model"
    # ----------------------------------------------------
    img_flag = True

    test_path_ir = '../../processed_dataset/test/ir'
    test_path_vi = '../../processed_dataset/test/rgb'
    
    ir_pathes, ir_names = utils.list_images_test(test_path_ir)
    # ---------------------------------------------------
    output_path = './output/crossfuse_test'
    if os.path.exists(output_path) is False:
        os.mkdir(output_path)
    output_path_fea = output_path + '/feature'
    if os.path.exists(output_path_fea) is False:
        os.mkdir(output_path_fea)
    # ---------------------------------------------------
    count = 0
    shift_flag = True
    with torch.no_grad():
        model_auto_ir, model_auto_vi, model_trans = load_model(custom_config_auto, custom_config_trans,
                                                               resume_model_auto_ir, resume_model_auto_vi,
                                                               model_path_trans)
        for ir_name in ir_names:
            # vi_name = ir_name.replace('IR', 'VIS')
            vi_name = ir_name
            ir_path = os.path.join(test_path_ir, ir_name)
            vi_path = os.path.join(test_path_vi, vi_name)
            # ---------------------------------------------------
            # if vi_name.__contains__('11'):
            test(model_auto_ir, model_auto_vi, model_trans, shift_flag, ir_path, vi_path, ir_name, output_path,
                 output_path_fea)