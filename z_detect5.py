# -*- coding = utf-8 -*-
# @Time : 2022/10/28 17:56
# @Author : cxk
# @File : z_detect5.py
# @Software : PyCharm


import sys
import ctypes
import signal

import argparse
import time
import win32con
import win32api

from mss import mss
import cv2
from pynput import mouse
from pathlib import Path

# 可选更底层键盘捕获库（需管理员权限）。若不可用则自动回退到 GetAsyncKeyState。
try:
    import keyboard  # type: ignore
    _KEYBOARD_AVAILABLE = True
except Exception:
    _KEYBOARD_AVAILABLE = False

from z_captureScreen import capScreen

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.utils import *

from z_ctypes import SendInput, mouse_input, mouse_event

# 自动查找权重文件：优先你的 head_body 权重，其次所有 runs/**/weights 下的最新 .pt
def _find_latest_best():
    import glob, os
    from pathlib import Path
    # 回退：以源码目录为基准
    script_dir = Path(__file__).resolve().parent
    # 1) 明确优先路径（如果存在则直接使用）
    preferred = [
        script_dir / 'runs' / 'exp10_exp_head_body' / 'weights' / 'best_exp_head_body.pt',
        script_dir / 'runs' / 'exp10_exp_head_body' / 'weights' / 'last_exp_head_body.pt',
    ]
    for p in preferred:
        try:
            if p.exists():
                return str(p)
        except Exception:
            pass
    # 2) 广泛搜索，优先 best* 与 *head_body*，最后所有 *.pt
    patterns = [
        'runs/**/weights/best*.pt',
        'runs/**/weights/*head_body*.pt',
        'runs/**/weights/*.pt',
    ]
    candidates = []
    for pat in patterns:
        try:
            candidates += glob.glob(str(script_dir / pat), recursive=True)
        except Exception:
            pass
    if candidates:
        try:
            return max(candidates, key=os.path.getctime)
        except Exception:
            pass
    # 3) 回退到项目默认
    return str(script_dir / 'weights' / 'best_200.pt')

PROCESS_PER_MONITOR_DPI_AWARE = 2
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)


def pre_process(img0, img_sz, half, device):
    """
    img0: from capScreen(), format: HWC, BGR
    """
    # padding resize
    img = letterbox(img0, new_shape=img_sz)[0]
    # convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)

    # preprocess
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0-255 to 0.0-1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def inference_img(img, model, augment, conf_thres, iou_thres, classes, agnostic):
    """
    推理，模型参数，...
    """
    # inference
    pred = model(img, augment=augment)[0]
    # apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic)
    return pred


def calculate_position(xyxy):
    """
    计算中心坐标
    """
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    # print('\n左上点坐标:(' + str(c1[0]) + ',' + str(c1[1]) + '), 右上点坐标:(' + str(c2[0]) + ',' + str(
    #     c1[1]) + ')')
    # print('左下点坐标:(' + str(c1[0]) + ',' + str(c2[1]) + '), 右下点坐标:(' + str(c2[0]) + ',' + str(
    #     c2[1]) + ')')
    # print("中心点的坐标为：(" + str((c2[0] - c1[0]) / 2 + c1[0]) + "," + str(
    #     (c2[1] - c1[1]) / 2 + c1[1]) + ")")
    center_x = int((c2[0] - c1[0]) / 2 + c1[0])
    center_y = int((c2[1] - c1[1]) / 2 + c1[1])
    return center_x, center_y


def view_imgs(img0):
    """
    弹窗展示结果，press q to quit
    """
    import cv2
    img0 = cv2.resize(img0, (480, 540))
    # img0 = cv2.resize(img0, (960, 540))
    cv2.imshow('ws demo', img0)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


def move_mouse(mouse_pynput, aim_persons_center, gain=1.2, step_div=40):
    """
    将鼠标移动改为“相对从屏幕中心移动”，适配大多数FPS的 Raw Input。
    选择距离屏幕中心最近的目标，按相对位移分步移动。
    """
    if aim_persons_center:
        sw, sh = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        center_x, center_y = sw // 2, sh // 2

        # 距离屏幕中心最近的目标中心点及距离
        best_position = None
        for aim_person in aim_persons_center:
            dist = ((aim_person[0] - center_x) ** 2 + (aim_person[1] - center_y) ** 2) ** 0.5
            if not best_position:
                best_position = (aim_person, dist)
            else:
                _, old_dist = best_position
                if dist < old_dist:
                    best_position = (aim_person, dist)

        dx = int(best_position[0][0] - center_x)
        dy = int(best_position[0][1] - center_y)
        # 增益调节：根据游戏灵敏度适配
        dx = int(dx * gain)
        dy = int(dy * gain)

        # 分步平滑（避免一次移动过大被游戏忽略），步长可根据效果调节
        steps = max(abs(dx), abs(dy)) // max(1, step_div) + 1
        step_dx = int(dx / steps) if steps else 0
        step_dy = int(dy / steps) if steps else 0
        for _ in range(steps):
            # 优先 SendInput，失败则回退 mouse_event
            ret = SendInput(mouse_input(win32con.MOUSEEVENTF_MOVE, step_dx, step_dy))
            if ret == 0:
                mouse_event(win32con.MOUSEEVENTF_MOVE, step_dx, step_dy, 0, 0)
            time.sleep(0.001)


class AimYolo:

    def __init__(self, opt):
        self.weights = opt.weights
        self.image_path = getattr(opt, 'image_path', None)
        self.img_size = opt.img_size
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.view_img = opt.view_img
        # 取消默认类别过滤；若未显式传入 --classes，则检测所有类别
        self.classes = opt.classes if opt.classes is not None else None
        self.agnostic_nms = opt.agnostic_nms
        self.augment = opt.augment
        self.no_move = getattr(opt, 'no_move', False)
        # 鼠标移动参数，可根据游戏灵敏度调整
        self.mouse_gain = getattr(opt, 'mouse_gain', 1.2)
        self.mouse_step_div = getattr(opt, 'mouse_step_div', 40)
        # 控制与模式
        self.scanning_enabled = False  # 中键点击开启/关闭
        self.aim_target = 'head'       # F1=head, F2=body
        self.should_exit = False       # P 退出
        # 锁定模式（一次获取并锁定）
        self.lock_active = False
        self.locked_target_center = None
        self.acquire_requested = False
        self.track_radius = 120  # 跟踪匹配阈值（像素半径）
        # 采集策略：限制锁定在屏幕中心附近并要求最低置信度
        self.min_acquire_conf = getattr(opt, 'min_acquire_conf', 0.35)
        # 严格锁定：仅锁定当前目标类别（head 或 body），不回退到另一类
        self.acquire_strict = getattr(opt, 'acquire_strict', False)

        # 根据当前屏幕自动设置截图区域
        sw, sh = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        self.bounding_box = {'left': 0, 'top': 0, 'width': sw, 'height': sh}
        self.acquire_max_radius = getattr(opt, 'acquire_max_radius', min(sw, sh) // 6)  # 默认≈画面对角的1/6

        # 回退：以源码目录解析权重路径
        script_dir = Path(__file__).resolve().parent

        def _resolve_weight_path(w):
            p = Path(w)
            if p.is_absolute():
                return str(p)
            # 尝试相对脚本目录
            cand1 = script_dir / p
            if cand1.exists():
                return str(cand1)
            # 尝试放在脚本目录下的 weights 目录中，仅按文件名匹配
            cand2 = script_dir / 'weights' / p.name
            if cand2.exists():
                return str(cand2)
            # 若都不存在，原样返回，让 attempt_load 自行处理（可能触发下载）
            return str(p)

        if isinstance(self.weights, (list, tuple)):
            self.weights = [_resolve_weight_path(w) for w in self.weights]
        else:
            self.weights = _resolve_weight_path(self.weights)
        print(f"Using weights: {self.weights}")

        # load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.model = self.model.to(self.device)
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # name and color
        self.names = self.model.modules.names if hasattr(self.model, 'module') else self.model.names
        print(f"Model classes: {self.names}")
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    @torch.no_grad()
    def run(self):

        img_sz = self.img_size

        # warm up
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

        # 离线图片模式：如果提供了 --image-path，则仅对该图片进行一次检测并退出
        if self.image_path is not None:
            import os
            if not os.path.exists(self.image_path):
                print(f"错误：图片路径不存在 -> {self.image_path}")
                return
            # 兼容含中文路径的读取
            import numpy as np
            img0 = cv2.imread(self.image_path)
            if img0 is None:
                try:
                    data = np.fromfile(self.image_path, dtype=np.uint8)
                    img0 = cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception:
                    img0 = None
            if img0 is None:
                print(f"错误：无法读取图片 -> {self.image_path}")
                return

            img = pre_process(img0=img0, img_sz=img_sz, half=self.half, device=self.device)
            t1 = torch_utils.time_synchronized()
            pred = inference_img(img=img, model=self.model, augment=self.augment, conf_thres=self.conf_thres,
                                 iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = torch_utils.time_synchronized()
            det = pred[0]
            s = ''
            s += '%gx%g ' % img.shape[2:]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += '%g %ss, ' % (n, self.names[int(c)])
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            if self.view_img:
                view_imgs(img0=img0)
            return

        # mss and capture screen
        sct = mss()
        print("The mss object created.")
        # mouse control
        mouse_control = mouse.Controller()
        print("The mouse controller created.")

        # 键鼠控制改为全局轮询，移除局部监听器，避免重复触发

        # 若只在循环中的if前定义空列表，在检测不到目标时，for循环没有对其进行初始化，会报错
        aim_persons_center = []
        aim_persons_center_head = []
        aim_persons_center_body = []

        # 全局按键轮询的前一次状态（用于边沿检测，避免连发）
        prev_mbutton = False
        prev_f1 = False
        prev_f2 = False
        prev_p = False
        while True:
            if self.should_exit:
                break
            # 兼容游戏输入：优先使用 keyboard 库进行更底层捕获；不可用则回退到 GetAsyncKeyState
            mbutton_state = bool(win32api.GetAsyncKeyState(0x04) & 0x8000)  # VK_MBUTTON（鼠标中键）
            if _KEYBOARD_AVAILABLE:
                f1_state = keyboard.is_pressed('f1')
                f2_state = keyboard.is_pressed('f2')
                # P 大小写都尝试
                p_state = keyboard.is_pressed('p') or keyboard.is_pressed('P')
            else:
                f1_state = bool(win32api.GetAsyncKeyState(0x70) & 0x8000)   # VK_F1
                f2_state = bool(win32api.GetAsyncKeyState(0x71) & 0x8000)   # VK_F2
                p_state = bool(win32api.GetAsyncKeyState(ord('P')) & 0x8000)  # 'P'

            if mbutton_state and not prev_mbutton:
                if not self.lock_active:
                    self.acquire_requested = True
                    print("请求获取并锁定目标（单次识别）")
                else:
                    self.lock_active = False
                    self.locked_target_center = None
                    print("已取消锁定，等待下一次中键以重新识别")
            if f1_state and not prev_f1:
                self.aim_target = 'head'
                print('瞄准模式：头部(F1)')
            if f2_state and not prev_f2:
                self.aim_target = 'body'
                print('瞄准模式：身体(F2)')
            if p_state and not prev_p:
                self.should_exit = True
                print('收到退出指令(P)，程序即将结束...')

            prev_mbutton = mbutton_state
            prev_f1 = f1_state
            prev_f2 = f2_state
            prev_p = p_state
            img0 = capScreen(sct, self.bounding_box)  # HWC and BGR

            img = pre_process(img0=img0, img_sz=img_sz, half=self.half, device=self.device)

            t1 = torch_utils.time_synchronized()
            pred = inference_img(img=img, model=self.model, augment=self.augment, conf_thres=self.conf_thres,
                                 iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # process detections
            det = pred[0]

            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # write results
                aim_persons_center = []
                aim_persons_center_head = []
                aim_persons_center_body = []
                for *xyxy, conf, cls in det:

                    # Add bbox to image
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)
                    center_x, center_y = calculate_position(xyxy)
                    cconf = float(conf)
                    aim_persons_center.append([center_x, center_y, cconf])
                    if int(cls) == 0:
                        aim_persons_center_head.append([center_x, center_y, cconf])
                    elif int(cls) == 1:
                        aim_persons_center_body.append([center_x, center_y, cconf])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # 获取请求：在当前检测结果基础上选择并锁定一个目标（距离屏幕中心最近）
            if self.acquire_requested:
                sw, sh = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
                cx, cy = sw // 2, sh // 2
                max_r2 = self.acquire_max_radius ** 2

                def select_candidate(c_list):
                    if not c_list:
                        return None
                    # 先筛选：在中心半径内且置信度达标
                    filtered = [p for p in c_list if p[2] >= self.min_acquire_conf and (p[0]-cx)**2 + (p[1]-cy)**2 <= max_r2]
                    if filtered:
                        # 在筛选集里选最高置信度
                        return max(filtered, key=lambda p: p[2])
                    # 否则退化为在原集合中选择距离中心最近的
                    return min(c_list, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)

                candidate = None
                if self.aim_target == 'head':
                    if aim_persons_center_head:
                        candidate = select_candidate(aim_persons_center_head)
                    elif not self.acquire_strict:
                        other = aim_persons_center_body
                        candidate = select_candidate(other) if other else select_candidate(aim_persons_center)
                else:  # body
                    if aim_persons_center_body:
                        candidate = select_candidate(aim_persons_center_body)
                    elif not self.acquire_strict:
                        other = aim_persons_center_head
                        candidate = select_candidate(other) if other else select_candidate(aim_persons_center)

                if candidate is not None:
                    self.lock_active = True
                    self.locked_target_center = (candidate[0], candidate[1])
                    dist = ((candidate[0]-cx)**2 + (candidate[1]-cy)**2) ** 0.5
                    print(f"已锁定目标：[{candidate[0]}, {candidate[1]}]（conf={candidate[2]:.2f}, dist={dist:.1f}, mode={self.aim_target}）")
                    if dist > self.acquire_max_radius:
                        print(f"提示：锁定点距中心偏远（>{self.acquire_max_radius}px），请将准星靠近目标后再中键锁定效果更好。")
                else:
                    print("未检测到可锁定目标或置信度不足，请靠近目标后再试。")
                self.acquire_requested = False

            # 跟踪与移动：仅在锁定状态下移动鼠标；尝试用本帧检测更新锁定中心
            if self.lock_active and self.locked_target_center is not None:
                # 根据当前模式选择候选集；严格模式下不回退到另一类
                if self.aim_target == 'head':
                    candidates = aim_persons_center_head if self.acquire_strict else (aim_persons_center_head or aim_persons_center_body or aim_persons_center)
                else:
                    candidates = aim_persons_center_body if self.acquire_strict else (aim_persons_center_body or aim_persons_center_head or aim_persons_center)
                if candidates:
                    lx, ly = self.locked_target_center
                    def d2(p):
                        return (p[0]-lx)**2 + (p[1]-ly)**2
                    best = min(candidates, key=d2)
                    if d2(best) <= self.track_radius**2:
                        self.locked_target_center = (best[0], best[1])
                if not self.no_move:
                    move_mouse(mouse_control, [self.locked_target_center], gain=self.mouse_gain, step_div=self.mouse_step_div)
            aim_persons_center = []
            aim_persons_center_head = []
            aim_persons_center_body = []

            # view img
            if self.view_img:
                # 在视图上标注锁定点
                if self.lock_active and self.locked_target_center is not None:
                    lx, ly = int(self.locked_target_center[0]), int(self.locked_target_center[1])
                    cv2.circle(img0, (lx, ly), 8, (0, 255, 255), -1)
                    cv2.circle(img0, (lx, ly), self.track_radius, (0, 255, 255), 2)
                view_imgs(img0=img0)

        # 全局轮询无需停止监听器

        # End ------------------------------------------


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=_find_latest_best(), help='model.pt path(s)')
    parser.add_argument('--image-path', type=str, default=None, help='offline image path to run single detection')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-move', action='store_true', help='dry-run for mouse, do not move pointer')
    parser.add_argument('--mouse-gain', type=float, default=1.2, help='relative move gain (higher -> faster)')
    parser.add_argument('--mouse-step-div', type=int, default=40, help='bigger value -> fewer but larger steps')
    parser.add_argument('--min-acquire-conf', type=float, default=0.35, help='min conf to acquire lock')
    parser.add_argument('--acquire-max-radius', type=int, default=0, help='max center radius to acquire; 0 -> auto')
    parser.add_argument('--acquire-strict', action='store_true', help='only lock the chosen target class; disable fallback to other classes')
    opt = parser.parse_args()
    # 将 0 的 acquire-max-radius 转换为根据屏幕尺寸的默认值
    if getattr(opt, 'acquire_max_radius', 0) == 0:
        sw, sh = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
        setattr(opt, 'acquire_max_radius', min(sw, sh) // 6)
    return opt


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    opt = parseArgs()
    print(opt)

    aim_yolo = AimYolo(opt)
    print('The AimYolo Object Created.')

    aim_yolo.run()

