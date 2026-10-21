# ArUco 视觉定位项目（ArUco_ZMP）

基于 OpenCV ArUco 的视觉定位与位姿检测项目。通过 USB 摄像头识别 ArUco 标签，实时计算标签相对相机的 **3D 位置（XYZ）、欧拉角（Roll/Pitch/Yaw）和直线距离**，可用于机器人导航、助行车跟随、跌倒检测等场景。

- **目标平台**：树莓派（UART 发布）+ STM32 下位机；PC 上同样可运行调试
- **默认字典**：`DICT_6X6_250`（6×6 网格，250 个 ID）
- **默认标签尺寸**：100 mm（可在各脚本配置区修改）
- **坐标系**：相机为原点，X 右 / Y 下 / Z 前，单位随 `MARKER_SIZE` 设定

## 生成 ArUco 标签

在线生成并打印：<https://chev.me/arucogen/>

- Dictionary 选择 `6X6 (250)`（与代码中 `DICT_6X6_250` 一致），输入所需 ID
- 打印尺寸须与代码中 `MARKER_SIZE` 一致（默认 100 mm），否则测距结果不准确

## 文件说明

| 文件 | 说明 |
| :--- | :--- |
| `camera_calibration.py` | **相机标定**。用棋盘格（9×6 内角点、25mm 格子，分辨率 1280×720）采集照片并调用 `calibrateCamera` 计算内参矩阵 `mtx` 与畸变系数 `dist`，结果直接打印可复制 |
| `aruco_no_distortion.py` | ArUco 位姿检测，**已写入标定内参、畸变系数置 0**（适用无畸变镜头/OV9281），分辨率 1280×720 |
| `aruco_distortion.py` | 同上，但**写入真实畸变系数**，用于有畸变镜头 |
| `aruco_demo.py` | 入门演示：ArUco 检测 + 位姿估计（内参为 640×480 估算值），画面显示 XYZ、欧拉角、距离并绘制坐标轴，注释详尽 |
| `fall_detection.py` | **跌倒检测演示**：标签俯仰角（Pitch）绝对值超过 45° 判定为摔倒，画面红框报警（助行车使用者姿态监测） |
| `aruco_uart.py` | **主程序（树莓派）**：ArUco 检测 + 中值/低通滤波 + 有效性校验，按固定 30 Hz 通过 UART 以文本协议发送位姿给 STM32 |
| `aruco_parameters.md` | ArUco 可获取参数（tvec/rvec/欧拉角/角点等）的详细说明文档 |
| `requirements.txt` | 依赖清单 |

## 环境搭建

```bash
# 1. 创建虚拟环境（Python 3.10+）
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / 树莓派

# 2. 安装依赖
pip install -r requirements.txt
```

依赖说明：

- `opencv-contrib-python`：ArUco 模块所在包
- `numpy`：矩阵与滤波计算
- `pyserial`：**仅** `aruco_uart.py` 启用 UART 时需要；树莓派上也可 `sudo apt install python3-serial`

## 使用流程

0. **生成并打印标签**：用 [chev.me/arucogen](https://chev.me/arucogen/) 生成 `DICT_6X6_250` 标签并打印（尺寸须与 `MARKER_SIZE` 一致）。
1. **标定相机**（可选但推荐）：运行 `camera_calibration.py`，棋盘格稳定检测时按 `s` 保存（建议 15 张以上、覆盖画面各区域），按 `c` 计算，按 `q` 退出。**标定分辨率必须与识别时一致**（默认 1280×720）。
2. **写入内参**：把标定输出的 `mtx`、`dist` 复制到检测脚本的 `camera_matrix`、`dist_coeffs`；无畸变镜头直接用 `aruco_no_distortion.py`。
3. **运行检测**：按场景选择脚本
   - PC 调试：`python aruco_demo.py`（或 `aruco_no_distortion.py` / `aruco_distortion.py`），按 `q` 退出
   - 跌倒检测：`python fall_detection.py`
   - 树莓派 + STM32：修改 `aruco_uart.py` 配置区后 `python aruco_uart.py`

## UART 协议（`aruco_uart.py`）

固定 30 Hz 发送文本协议，格式类似 NMEA：

```
$ARUCO,seq,status,id,x_mm,y_mm,z_mm,roll_x10,pitch_x10,yaw_x10,dist_mm*CS#
```

- `status`：`0 = LOST`（目标丢失）/ `1 = VALID`（本轮有效）/ `2 = STALE`（短时丢帧，沿用滤波值）
- `x_mm/y_mm/z_mm`：标签中心相对相机坐标（mm）；`dist_mm`：直线距离（mm）
- `roll_x10/pitch_x10/yaw_x10`：欧拉角 ×10（0.1 度）
- `CS`：`$` 与 `*` 之间内容的 XOR 校验和（2 位十六进制），STM32 端可据此校验

## 常用配置项（各脚本顶部配置区）

| 配置 | 说明 |
| :--- | :--- |
| `ARUCO_DICT_TYPE` | ArUco 字典，需与打印的标签一致 |
| `MARKER_SIZE` | 标签实际边长（米），直接影响 XYZ 测距精度 |
| `camera_matrix` / `dist_coeffs` | 相机内参与畸变，标定后填入 |
| `ENABLE_UART` | `True` 才发送串口；`False` 仅终端打印 |
| `SERIAL_PORT` / `BAUD_RATE` | 串口设备与波特率（树莓派默认 `/dev/ttyS0`、115200） |
| `TARGET_IDS` | 标签 ID 白名单，`None` 接受全部，`{3}` 只认 ID=3 |
| `FALL_ANGLE_THRESHOLD` | 跌倒判定角度阈值（度），见 `fall_detection.py` |
| `SHOW_WINDOW` | 树莓派无显示器运行（SSH/自启动）时设为 `False` |

## 注意事项

- 3D 坐标的单位取决于 `MARKER_SIZE` 的单位：填 `0.1`（米）则 XYZ 输出为米
- 标定、识别时的**相机分辨率必须一致**，否则内参失效、测距不准
- `aruco_uart.py` 中 `FRAME_WIDTH/HEIGHT` 设为 640×480，若替换为标定内参需保证分辨率匹配
- 树莓派使用 `/dev/ttyS0` 前需释放串口（`cmdline.txt` 中移除 `console=ttyS0,115200`）
