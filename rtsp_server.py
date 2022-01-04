#!/usr/bin/python3
import os
import cv2
import time
import argparse

import torch
import model.detector

import sys
import utils.utils

import json
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=59,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def cam_pipline(settings: dict):
    return "rtsp://{}:{}@{}:{}/Streaming/Channels/{}".format(
        settings['user'], 
        settings['password'], 
        settings['ip'], 
        settings['port'],
        settings['channel'])

# see: https://stackoverflow.com/questions/47396372/write-opencv-frames-into-gstreamer-rtsp-server-pipeline?rq=1
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, cap: str, model_weights, config, device, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.cam_url = cap
        self.cap = cv2.VideoCapture(cap)
        self.config = config
        self.model = model_loader(self.config, model_weights, device)
        self.number_frames = 0
        self.fps = 30
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=640,height=360,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96'.format(self.fps)
        
        ret, frame = self.cap.read()
        if ret:
            self.out = img_process(
                ori_img=frame, 
                mod=self.model, 
                config=self.config, 
                device=torch.device('cuda')
            )

    def on_need_data(self, src, lenght):
        # ret, frame = self.cap.read()
        ret = self.cap.grab()
        # self.number_frames += 1
        if not ret:
            self.cap.release()
            self.cap.open(self.cam_url)
            return None
        
        if (self.number_frames % 3) == 0:
            ret, frame = self.cap.retrieve()
            self.out = img_process(
                ori_img=frame, 
                mod=self.model, 
                config=self.config, 
                device=torch.device('cuda')
            )

        data = self.out.tostring()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
        # print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
        #                                                                         self.duration,
        #                                                                         self.duration / Gst.SECOND))
        if retval != Gst.FlowReturn.OK:
            print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, cap: str, model_weights, device, config, **properties):
        super(GstServer, self).__init__(**properties)
        self.factory = SensorFactory(cap=cap, model_weights=model_weights, device=device, config=config)
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)

def model_loader(config: dict, weights: str, device: torch.device):
    #模型加载
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = model.detector.Detector(config["classes"], config["anchor_num"], True).to(device)
    m.load_state_dict(torch.load(weights,map_location='cpu'))
    #sets the module in eval node
    m.eval()
    return m

def img_process(ori_img, mod, config, device):
    res_img = cv2.resize(ori_img, (config["width"], config["height"]), interpolation= cv2.INTER_LINEAR)
    img = res_img.reshape(1, config["height"], config["width"], 3)
    img = torch.from_numpy(img.transpose(0,3, 1, 2))
    img = img.to(device).float() / 255.0

    #模型推理
    # print('forward')
    preds = mod(img)
    # print('end forward')
    # end = time.perf_counter()
    # _time = (end - start) * 1000.
    # print("forward time:%fms"%_time)

    #特征图后处理
    output = utils.utils.handel_preds(preds, config, device)
    output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

    #加载label names
    LABEL_NAMES = []
    with open(config["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())
    
    h, w, _ = ori_img.shape
    scale_h, scale_w = h / config["height"], w / config["width"]

    #绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()
    
        obj_score = box[4]
        if obj_score >= 0.6:
            category = LABEL_NAMES[int(box[5])]

            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        
    return ori_img

def skip_image(capture, count):
    for i in range(count):
        _, __ = capture.read()

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='', 
                        help='The path of the .pth model to be transformed')

    with open('cam.json', 'r') as load_file:
        cam_settings = json.load(load_file)

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"

    # #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # my_model = model_loader(cfg, opt.weights, device)

    # load camera
    # cam = cv2.VideoCapture(cam_pipline(cam_settings))
    # assert cam.isOpened() is True, '[{}]: Camera open failed. '.format(
    #     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # start server
    GObject.threads_init()
    Gst.init(None)
    server = GstServer(cap=cam_pipline(cam_settings), model_weights=opt.weights, device=device, config=cfg)
    loop = GObject.MainLoop()
    print('[{}]: Loop begins. '.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    loop.run()

    """

    while True:
        start = time.perf_counter()
        # read image
        for i in range(12):
            '''
            After the cctv camera is opened, all the frames which captured is stored in 
            the buffer in the camera. To catch up with the camera, dump a few frames here. 
            '''
            _, __ = cam.read()
        retVal, ori_img = cam.read()
        assert ori_img is not None, '[{}]: Camera read failed. '.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        ori_img = img_process(ori_img=ori_img, mod=my_model, config=cfg, device=device)

        end = time.perf_counter()
        _time = (end - start) * 1000.
        cv2.putText(ori_img, "{: >7.2f}ms".format(_time), (0, 16), 0, 0.5, (255, 0, 0), 1)
        cv2.putText(ori_img, "{: >7.1f}fps".format(1000/_time), (0, 32), 0, 0.5, (255, 0, 0), 1)

    cv2.destroyAllWindows()
    cam.release()
    """