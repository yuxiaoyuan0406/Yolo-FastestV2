import os
import argparse

import torch
import model.detector

import sys
from ctypes import *
import cv2
import time
import pyttsx3
sys.path.append('..')
import utils.utils


class HkAdapter:
    dll_path = r"F:\codes\pycharm\dll"

    def load_hkdll(self):
        # 载入HCCore.dll
        hccode = WinDLL(__class__.dll_path + "\\HCCore.dll")
        # 载入HCNetSDK.dll
        hcnetsdk = cdll.LoadLibrary(__class__.dll_path + "\\HCNetSDK.dll")
        return hcnetsdk


class NET_DVR_DEVICEINFO_V30(Structure):
    _fields_ = [
        ("sSerialNumber", c_byte * 48),  # 序列号
        ("byAlarmInPortNum", c_byte),  # 模拟报警输入个数
        ("byAlarmOutPortNum", c_byte),  # 模拟报警输出个数
        ("byDiskNum", c_byte),  # 硬盘个数
        ("byDVRType", c_byte),  # 设备类型，详见下文列表
        ("byChanNum", c_byte),
        ("byStartChan", c_byte),
        ("byAudioChanNum", c_byte),  # 设备语音对讲通道数
        ("byIPChanNum", c_byte),
        ("byZeroChanNum", c_byte),  # 零通道编码个数
        ("byMainProto", c_byte),  # 主码流传输协议类型：
        ("bySubProto", c_byte),  # 字码流传输协议类型：
        ("bySupport", c_byte),
        ("bySupport1", c_byte),
        ("bySupport2", c_byte),
        ("wDevType", c_uint16),  # 设备型号，详见下文列表
        ("bySupport3", c_byte),
        ("byMultiStreamProto", c_byte),
        ("byStartDChan", c_byte),  # 起始数字通道号，0表示无数字通道，比如DVR或IPC
        ("byStartDTalkChan", c_byte),
        ("byHighDChanNum", c_byte),  # 数字通道个数，高8位
        ("bySupport4", c_byte),
        ("byLanguageType", c_byte),
        ("byVoiceInChanNum", c_byte),  # 音频输入通道数
        ("byStartVoiceInChanNo", c_byte),  # 音频输入起始通道号，0表示无效
        ("bySupport5", c_byte),
        ("bySupport6", c_byte),
        ("byMirrorChanNum", c_byte),  # 镜像通道个数，录播主机中用于表示导播通道
        ("wStartMirrorChanNo", c_uint16),
        ("bySupport7", c_byte),
        ("byRes2", c_byte)]  # 保留，置为0


class NET_DVR_DEVICEINFO_V40(Structure):
    _fields_ = [
        # struDeviceV30结构体中包括接口体，我们只需要额外定义一下该子结构体即可
        ("struDeviceV30", NET_DVR_DEVICEINFO_V30),
        ("bySupportLock", c_byte),  # 设备是否支持锁定功能，bySupportLock为1时，dwSurplusLockTime和byRetryLoginTime有效
        ("byRetryLoginTime", c_byte),
        ("byPasswordLevel", c_byte),
        ("byProxyType", c_byte),
        ("dwSurplusLockTime", c_ulong),
        ("byCharEncodeType", c_byte),
        ("bySupportDev5", c_byte),
        ("bySupport", c_byte),
        ("byLoginMode", c_byte),
        ("dwOEMCode", c_ulong),
        ("iResidualValidity", c_int),
        ("byResidualValidity", c_byte),
        ("bySingleStartDTalkChan", c_byte),
        ("bySingleDTalkChanNums", c_byte),
        ("byPassWordResetLevel", c_byte),
        ("bySupportStreamEncrypt", c_byte),
        ("byMarketType", c_byte),
        ("byRes2", c_byte * 253),
    ]


class NET_DVR_USER_LOGIN_INFO(Structure):
    _fields_ = [
        ("sDeviceAddress", c_char * 129),
        ("byUseTransport", c_byte),
        ("wPort", c_uint16),
        ("sUserName", c_char * 64),
        ("sPassword", c_char * 64),
        ("bUseAsynLogin", c_int),
        ("byProxyType", c_byte),
        ("byUseUTCTime", c_byte),
        ("byLoginMode", c_byte),
        ("byHttps", c_byte),
        ("iProxyID", c_long),
        ("byVerifyMode", c_byte),
        ("byRes3", c_byte * 120)
    ]


class NET_DVR_DEVSERVER_CFG(Structure):
    _fields_ = [
        ("dwSize", c_byte),
        ("byIrLampServer", c_byte),
        ("byTelnetServer", c_byte),
        ("byABFServer", c_byte),
        ("byEnableLEDStatus", c_byte),
        ("byEnableAutoDefog", c_byte),
        ("byEnableSupplementLight", c_byte),
        ("byEnableDeicing", c_byte),
        ("byEnableVisibleMovementPower", c_byte),
        ("byEnableThermalMovementPower", c_byte),
        ("byEnablePtzPower", c_byte),
        ("byPowerSavingControl", c_byte),
        ("byRes[245];", c_byte),
    ]


def init(hksdk):  # 初始化函数
    if not hksdk.NET_DVR_Init():
        print("初始失败")
        return False
    if not hksdk.NET_DVR_SetConnectTime():
        print("设置连接时间失败")
        return False
    print("初始化成功")
    return True


def login(hksdk, url, usename, password, port=8000):  # 登录函数
    # python的String向ctype里的c_char传递的数据需要进行bytes编码
    burl = bytes(url, "ascii")
    busename = bytes(usename, "ascii")
    bpassword = bytes(password, "ascii")

    # 初始化两个结构体
    login_info = NET_DVR_USER_LOGIN_INFO()
    device_info = NET_DVR_DEVICEINFO_V40()
    # 设置登录信息
    login_info.wPort = port
    login_info.bUseAsynLogin = 0
    login_info.sUserName = busename
    login_info.sPassword = bpassword
    login_info.sDeviceAddress = burl

    # 获得引用，byref基本等于c++的&符号
    param_login = byref(login_info)  # 传递的为指针则使用该种方式
    param_device = byref(device_info)

    # 执行NET_DVR_Login_V40函数，获取userid
    useid = hksdk.NET_DVR_Login_V40(param_login, param_device)
    # 登录成功时，useid的值为0、1.....，失败时为-1
    # 可以调用NET_DVR_GetLastError查看错误码
    if useid == -1:
        print("登录失败，错误码为{}".format(hksdk.NET_DVR_GetLastError()))
    else:
        print("登录成功，用户id为{}".format(useid))

    return useid


def uinit(hksdk, useid):  # 反初始化函数
    isOK = hksdk.NET_DVR_Logout(useid)
    if isOK == -1:
        print("登出失败错误码为{}".format(hksdk.NET_DVR_GetLastError()))
    else:
        print("登出成功")
    hksdk.NET_DVR_Cleanup()


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


if __name__ == '__main__':
    #登录摄像头
    print("-----------初始化与登录---------")
    hkadapter = HkAdapter()
    hksdk_m = hkadapter.load_hkdll()
    init(hksdk_m)
    userid = login(hksdk_m, "169.254.129.254", "admin", "ligonglou616")
    print("----------初始化与登录完成---------")
    hksdk_m.NET_DVR_SetAlarmOut(userid, 0x00ff, 0)

    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='',
                        help='The path of the .pth model to be transformed')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location='cpu'))

    # sets the module in eval node
    model.eval()

    # load camera
    # print(gstreamer_pipeline(flip_method=0))
    cam = cv2.VideoCapture("rtsp://admin:ligonglou616@169.254.129.254:554/Streaming/Channels/2")
    assert cam.isOpened() is True, 'Camera open failed. '

    while True:
        # 数据预处理
        retVal, ori_img = cam.read()
        res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(device).float() / 255.0

        # 模型推理
        start = time.perf_counter()
        preds = model(img)
        # end = time.perf_counter()
        # _time = (end - start) * 1000.
        # print("forward time:%fms"%_time)

        # 特征图后处理
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres=0.3, iou_thres=0.4)

        # 加载label names
        LABEL_NAMES = []
        with open(cfg["names"], 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())

        h, w, _ = ori_img.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]

        # 绘制预测框
        for box in output_boxes[0]:
            box = box.tolist()

            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]
            image_name = '%s%s.jpg' % ('F:/HIK/', time.ctime())
            if category == 'person':
                hksdk_m.NET_DVR_SetAlarmOut(userid, 0x00ff, 0)
                # engine = pyttsx3.init()
                # engine.say('请将电动车移出电梯外')
                cv2.imwrite(image_name, ori_img)
            else:
                hksdk_m.NET_DVR_SetAlarmOut(userid, 0x00ff, 0)

            x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
            x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        end = time.perf_counter()
        _time = (end - start) * 1000.
        cv2.putText(ori_img, "{: >7.2f}ms".format(_time), (0, 16), 0, 0.5, (255, 0, 0), 1)
        cv2.putText(ori_img, "{: >7.1f}fps".format(1000 / _time), (0, 32), 0, 0.5, (255, 0, 0), 1)

        cv2.imshow("result", ori_img)
        hksdk_m.NET_DVR_SetAlarmOut(userid, 0x00ff, 0)
        keyCode = cv2.waitKey(30) & 0xFF
        # Stop the program on the ESC key
        if keyCode == 27:
            break

    cv2.destroyAllWindows()
    cam.release()


