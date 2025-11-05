# -*- coding = utf-8 -*-
# @Time : 2022/10/24 10:22
# @Author : cxk
# @File : z_ctypes.py
# @Software : PyCharm

import ctypes
from ctypes import wintypes

# 使用 Windows 官方 wintypes，保证 32/64 位类型正确
LONG = wintypes.LONG
DWORD = wintypes.DWORD
# ctypes.wintypes 不提供 ULONG_PTR，按当前进程指针宽度定义：32位->c_ulong，64位->c_ulonglong
_PTR_SIZE = ctypes.sizeof(ctypes.c_void_p)
ULONG_PTR = ctypes.c_ulonglong if _PTR_SIZE == 8 else ctypes.c_ulong
WORD = wintypes.WORD

INPUT_MOUSE = 0


class MouseInput(ctypes.Structure):
    _fields_ = [
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]


class InputUnion(ctypes.Union):
    _fields_ = [
        ('mi', MouseInput)
    ]


class Input(ctypes.Structure):
    _fields_ = [
        ('type', DWORD),          # INPUT.type
        ('iu', InputUnion)
    ]


def mouse_input_set(flags, x, y, data):
    # dwExtraInfo 设为 0；类型必须是 ULONG_PTR（指针宽度整型）
    return MouseInput(LONG(x), LONG(y), DWORD(data), DWORD(flags), DWORD(0), ULONG_PTR(0))


def input_do(structure):
    if isinstance(structure, MouseInput):
        return Input(DWORD(INPUT_MOUSE), InputUnion(mi=structure))
    raise TypeError('Cannot create Input structure!')


def mouse_input(flags, x=0, y=0, data=0):
    return input_do(mouse_input_set(flags, x, y, data))


def SendInput(*inputs):
    # 绑定函数原型，避免类型不匹配导致返回值异常
    _SendInput = ctypes.windll.user32.SendInput
    _SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int)
    _SendInput.restype = wintypes.UINT

    n_inputs = len(inputs)
    lp_input = Input * n_inputs
    p_inputs = lp_input(*inputs)
    cb_size = ctypes.c_int(ctypes.sizeof(Input))
    return _SendInput(n_inputs, p_inputs, cb_size)


# 旧API兜底：部分游戏对 SendInput 不友好时，可使用 mouse_event
def mouse_event(dwFlags, dx=0, dy=0, dwData=0, dwExtraInfo=0):
    return ctypes.windll.user32.mouse_event(dwFlags, dx, dy, dwData, dwExtraInfo)


if __name__ == '__main__':
    SendInput(mouse_input(1, -100, -200))


