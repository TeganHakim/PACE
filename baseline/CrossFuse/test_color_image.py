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

DEVICE =torch.device("cpu" if not torch.cuda.is_available() else "cuda")
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
    model_trans.load_state_dict(torch.load(model_path_trans), strict=False) # Take out strict=False after testing!!
    model_trans.to(DEVICE)
    model_trans.eval()
    return model_auto_ir, model_auto_vi, model_trans

"""
Function to decode the detections from the model output into bounding boxes and class labels
"""
def decode_detections(detections, image_width, image_height, threshold=0.5):
    detections = detections[0]
    objectness = torch.sigmoid(detections[0])
    x = torch.sigmoid(detections[1])
    y = torch.sigmoid(detections[2])
    w = torch.sigmoid(detections[3])
    h = torch.sigmoid(detections[4])

    # Class predictions
    class_scores = torch.softmax(detections[5:10], dim=0)
    feature_height = detections.shape[1]
    feature_width = detections.shape[2]

    boxes = []
    for row in range(feature_height):
        for col in range(feature_width):
            confidence = objectness[row, col].item()
            if confidence < threshold:
                continue
            # Find most likely class
            class_score, class_id = torch.max(class_scores[:, row, col], dim=0)
            class_score = class_score.item()
            class_id = class_id.item()

            # Combine objectness and class confidence
            confidence = confidence * class_score
            if confidence < threshold:
                continue

            center_x = x[row, col].item() * image_width
            center_y = y[row, col].item() * image_height
            box_width = w[row, col].item() * image_width
            box_height = h[row, col].item() * image_height

            x1 = int(center_x - box_width / 2)
            y1 = int(center_y - box_height / 2)
            x2 = int(center_x + box_width / 2)
            y2 = int(center_y + box_height / 2)
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id]
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

    print("Fused feature shape:", outputs['fused_features'].shape)
    print("Detection output:", outputs['detections'].shape)
    
    # Decode and display bounding boxes on the original visible image
    vi_original = cv2.imread(vi_path)
    image_height, image_width = vi_original.shape[:2]
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
            vi_original,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )
        label = "{} {:.2f}".format(class_name, confidence)
        cv2.putText(
            vi_original,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    cv2.imwrite(os.path.join(output_path, "det_" + ir_name), vi_original)
    # img_out = outputs['out']
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
    resume_model_auto_vi = "./models/autoencoder/auto_encoder_epoch_4_vi.model"
    
    # model_path_auto = "./models/autoencoder/auto_encoder_epoch_3.model"
    model_path_trans = "./models/transfuse/fusetrans_epoch_32_bs_8_num_20k_lr_0.1_s1_c1.model"
    # model_path_trans = "./models/transfuse/fusetrans_epoch_8_nosh.model"
    # ----------------------------------------------------
    img_flag = True

    test_path_ir = './images/M3FD_Fusion/ir'
    test_path_vi = './images/M3FD_Fusion/vis'
    data_type = '/M3FD_Fusion_nosh'
 
    # test_path_ir = './images/vot/ir'
    # test_path_vi = './images/vot/vis'
    # data_type = '/vot_transfuse'
    
    ir_pathes, ir_names = utils.list_images_test(test_path_ir)
    # ---------------------------------------------------
    output_path1 = './output/crossfuse_test'
    if os.path.exists(output_path1) is False:
        os.mkdir(output_path1)
    output_path = output_path1 + data_type
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