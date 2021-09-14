import os
import cv2
# import time
import argparse

import torch
# import model.detector
from model import detector
import utils.utils

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='', 
                        help='The path of test image')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model = detector.Detector(cfg["classes"], cfg["anchor_num"], True)  # no need to move to gpu
    model.load_state_dict(torch.load(opt.weights))

    #sets the module in eval node
    model.eval()
    
    #数据预处理
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0,3, 1, 2))
    # img = img.to(device).float() / 255.0
    img = img.float() / 255.0

    # trace and save
    traced_net = torch.jit.trace(model, img)
    torch.jit.save(traced_net, './traced_net.pt')
