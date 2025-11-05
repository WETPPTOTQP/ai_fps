# AimYolo（crow_changed）

基于 YOLOv5 的射击类游戏瞄准辅助工具。支持屏幕实时检测、人头/身体目标识别、单次锁定与自动鼠标跟随。该版本为 crow_changed 分支（或目录）的源码运行版本，适用于 Windows。

---

## 主要特性

- 实时屏幕检测与弹窗显示（OpenCV）。
- 两类目标：`head`（类别0）与 `body`（类别1）。
- 中键单次识别并锁定，F1/F2 快速切换锁定目标类型（头/身体）。
- 自动鼠标跟随移动，可调速度与跟随半径；支持观测模式（不移动鼠标）。
- 离线图片检测模式（一次检测后退出）。
- 权重自动解析：优先你的训练输出，再回退至项目默认权重。

---

## 运行环境

- 操作系统：Windows 10/11（必须，依赖 `pywin32`、`mss`）。
- Python：推荐 3.8–3.11（更高版本请注意依赖兼容性）。
- 显卡：有 CUDA 则优先用 GPU，否则自动回退 CPU。
- 权限：普通权限即可；管理员权限可提升键盘捕获兼容性（非必须）。

---

## 依赖安装

如仓库无 `requirements.txt`，可直接安装核心依赖：

```bash
pip install numpy opencv_python mss pywin32 pynput keyboard Pillow tqdm PyYAML scipy thop onnx
# 选择与你环境匹配的 torch/torchvision（CPU 或 CUDA）
# 国内源示例：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy opencv_python mss pywin32 pynput keyboard Pillow tqdm PyYAML scipy thop onnx
```

- 建议先升级 pip：`pip install -U pip`
- 关于 `torch`：请根据你的 Python/CUDA 版本选择合适的轮子（CPU版或对应 CUDA 版）。

---

## 目录结构（简要）

```
crow_changed/
├── data/           # 数据与配置
├── models/         # 模型代码（YOLOv5）
├── runs/           # 训练输出目录（建议将权重放在 runs/**/weights/）
├── utils/          # 通用工具
├── weights/        # 默认或备用权重（如 best_200.pt）
├── detect.py       # YOLOv5 检测脚本（标准）
├── train.py        # YOLOv5 训练脚本（标准）
├── test.py         # YOLOv5 测试脚本（标准）
└── z_detect5.py    # 本项目主入口：实时检测与鼠标控制
```

---

## 权重解析逻辑

`z_detect5.py` 中的 `_find_latest_best()` 按以下顺序查找权重：

1. 首选（可按需修改）：
   - `runs/exp10_head_body/weights/best_head_body.pt`
   - `runs/exp10_head_body/weights/last_head_body.pt`
2. 广泛搜索（按创建时间取最新）：
   - `runs/**/weights/best*.pt`
   - `runs/**/weights/*head_body*.pt`
   - `runs/**/weights/*.pt`
3. 若仍未找到，回退到：`weights/best_200.pt`

- 你可以将自己的权重放在 `runs/<你的实验>/weights/`，或显式通过 `--weights` 指定。

---

## 快速开始

- 观测模式（不移动鼠标，推荐先验证）：

```bash
python z_detect5.py --view-img --no-move
```

- 正常模式（弹窗与鼠标移动）：

```bash
python z_detect5.py --view-img
```

- 指定权重运行：

```bash
python z_detect5.py --view-img --weights runs/exp10_head_body/weights/best_head_body.pt
```

- 离线图片检测（一次检测后退出）：

```bash
python z_detect5.py --image-path <你的图片路径> --view-img
```

---

## 交互与控制

- 中键点击：发起一次识别并锁定（再次点击取消锁定）
- F1：切换锁定目标为“头部”
- F2：切换锁定目标为“身体”
- P：退出程序

锁定策略：
- 优先在屏幕中心附近识别，默认半径为屏幕较短边的约 1/6（可配置）。
- 可启用严格模式，仅锁定当前目标类型（不回退到另一类）。

---

## 命令行参数（常用）

- `--weights` 模型权重路径（可多个），默认自动查找
- `--image-path` 离线图片路径（一次检测后退出）
- `--img-size` 推理分辨率，默认 `640`
- `--conf-thres` 置信度阈值，默认 `0.4`
- `--iou-thres` NMS IOU 阈值，默认 `0.5`
- `--view-img` 打开实时弹窗
- `--classes` 过滤类别，例如 `--classes 0` 或 `--classes 0 1`
- `--agnostic-nms` 类别无关 NMS
- `--augment` 增强推理
- `--no-move` 仅显示，不移动鼠标（观测模式）
- `--mouse-gain` 鼠标移动增益，默认 `1.2`（增大更快）
- `--mouse-step-div` 鼠标分步移动粒度，默认 `40`（增大步数更少但更大）
- `--min-acquire-conf` 最低锁定置信度，默认 `0.35`
- `--acquire-max-radius` 锁定中心最大半径，默认 `0`（自动按屏幕尺寸计算）
- `--acquire-strict` 严格锁定当前目标类型（不回退到另一类）

---

## 常见问题与排查

- 弹窗不显示或报错：
  - 确保安装了 `opencv_python`，并使用 `--view-img` 启动。

- 权重加载失败：
  - 检查 `runs/**/weights/*.pt` 是否存在；或替换 `weights/best_200.pt`。
  - 可用 `--weights <权重路径>` 显式指定。

- 鼠标移动过慢/过快：
  - 调整 `--mouse-gain` 与 `--mouse-step-div`。

- 键盘/鼠标捕获不稳定：
  - 可尝试以管理员身份运行；或先用 `--no-move` 验证模型与弹窗，再开启移动。

- `cv2` 未绑定或导入问题：
  - 使用模块级导入 `import cv2`，不要在函数内部再次 `import cv2`，避免作用域导致的 `UnboundLocalError`。

---

## 训练与评估（可选）

- 训练：`train.py`（按你的数据与标签进行训练后，权重会输出到 `runs/<exp>/weights/`）
- 评估：`test.py`（对数据集进行验证）
- YOLOv5 标准脚本参考使用方法，请根据你的数据路径与配置进行调整。

---

## 合规与免责声明

- 请遵守游戏及平台的使用条款与法律法规，仅用于学习研究。
- 在公共或竞争环境中使用可能导致账号风险，请谨慎评估。

---

## 许可协议

- 详见仓库内 `LICENSE` 文件。

---

## 致谢

- 本项目参考并基于 YOLOv5 相关实现，感谢社区的开源贡献。
- 感谢 `mss`、`pywin32`、`pynput`、`keyboard`、`opencv-python` 等优秀开源库。

