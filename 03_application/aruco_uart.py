import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time
import sys
from collections import deque


# ============================================================
#  树莓派 ArUco 视觉定位 + UART 发布程序
# ============================================================
#
#  适用场景：
#    树莓派通过 USB 摄像头识别 ArUco 标签，计算标签相对摄像头的位姿，
#    再通过 UART 发送给 STM32。
#
#  设计目标：
#    1. 适合机器人/助行车项目，不是简单打印测试。
#    2. 固定频率发送数据，避免 STM32 收到数据忽快忽慢。
#    3. 增加滤波，减少 ArUco 位姿抖动。
#    4. 增加状态位，处理短时间丢帧、目标丢失。
#    5. 增加序号和校验和，为 STM32 后续写接收状态机做准备。
#
#  当前协议是“文本调试协议”，方便用串口助手直接看。
#  后续如果需要，我再帮你改成正式二进制协议。
#
# ============================================================


# ============================================================
# 一、串口配置
# ============================================================

# 当前你的树莓派 Python 已经可以打开 /dev/ttyS0。
# 注意：
#   如果 cmdline.txt 里仍保留 console=ttyS0,115200，
#   那么这个串口仍然可能有 Linux 登录信息干扰。
#   正式接 STM32 前，应该释放该串口。
SERIAL_PORT = "/dev/ttyS0"

# 树莓派与 STM32 串口波特率
BAUD_RATE = 115200

# 是否启用串口发送
#   True：通过 UART 发给 STM32/串口助手
#   False：只在终端打印，不发串口
ENABLE_UART = True


# ============================================================
# 二、发布频率配置
# ============================================================

# 固定发送频率，单位 Hz。
# 对助行车/机器人场景，1Hz 太低。
# 这里先用 30Hz，即约每 33ms 发送一包。
SEND_HZ = 30.0
SEND_INTERVAL = 1.0 / SEND_HZ

# 终端打印频率分频。
# 串口仍然 30Hz 发送，但终端不必每包都打印，避免刷屏影响性能。
# 例如 PRINT_DECIMATION=5，表示每 5 包打印一次，即约 6Hz。
PRINT_TO_CONSOLE = True
PRINT_DECIMATION = 5


# ============================================================
# 三、摄像头配置
# ============================================================

# 你的扫描结果显示 USB 摄像头是 /dev/video0。
# 用路径比用数字 0 更稳定。
CAMERA_INDEX = "/dev/video0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

# 开机自启动/SSH运行时必须 False。
# 如果接显示器调试，可以改 True。
SHOW_WINDOW = False


# ============================================================
# 四、ArUco 配置
# ============================================================

ARUCO_DICT_TYPE = aruco.DICT_6X6_250

# 标签实际物理边长，单位：米。
# 这里 0.1 表示 100mm。
MARKER_SIZE = 0.1

# 目标标签 ID 白名单。
#   None：接受所有 ID。
#   {3}：只接受 ID=3 的标签。
#   {1, 2, 3}：只接受这些 ID。
#
# 后续用于助行车时，建议固定一个人体标签 ID，例如 TARGET_IDS = {3}
TARGET_IDS = None

# 相机内参。当前还是估算值。
# 后续如果要提升距离精度，建议重新做相机标定。
camera_matrix = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


# ============================================================
# 五、数据有效性与滤波配置
# ============================================================

# 距离有效范围，单位 mm。
# 太近/太远的数据直接认为无效，防止误识别。
MIN_DIST_MM = 100
MAX_DIST_MM = 4000

# 相邻有效帧之间允许的位置最大跳变，单位 mm。
# 如果一帧突然跳太多，通常是误识别或姿态估计异常。
MAX_JUMP_MM = 600

# 相邻有效帧之间允许的角度最大跳变，单位 0.1度。
# 例如 600 表示 60.0度。
MAX_ANGLE_JUMP_X10 = 600

# 中值滤波窗口长度。
# 3 或 5 比较合适。越大越稳，但延迟越大。
MEDIAN_WINDOW = 3

# 一阶低通滤波系数。
# alpha 越大响应越快，抖动也越明显。
# alpha 越小越平滑，但延迟更大。
FILTER_ALPHA_POS = 0.45
FILTER_ALPHA_ANGLE = 0.35

# 短时丢帧时间。
# 在这个时间内没识别到新标签，继续发送上一帧滤波值，但状态为 STALE。
STALE_TIMEOUT_S = 0.20

# 目标丢失时间。
# 超过这个时间还没识别到，就发送 LOST，STM32 不应继续使用旧位姿。
LOST_TIMEOUT_S = 0.50


# ============================================================
# 六、状态定义
# ============================================================
#
# 发送给 STM32 的 status 字段：
#
#   0 = LOST
#       长时间未识别到目标，数据无效。
#
#   1 = VALID
#       本轮识别到了目标，并通过了过滤，数据有效。
#
#   2 = STALE
#       短时间没有识别到新目标，暂时沿用上一帧滤波值。
#       STM32 可以短时间保持，但不能长时间依赖。
#
# ============================================================

STATUS_LOST = 0
STATUS_VALID = 1
STATUS_STALE = 2

INVALID_ID = 65535


# ============================================================
# 七、工具函数
# ============================================================

def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角。

    返回：
        roll, pitch, yaw，单位：度

    注意：
        这里的欧拉角定义主要用于工程调试。
        如果后续对姿态定义有严格要求，可以再统一坐标系约定。
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([
        math.degrees(x),
        math.degrees(y),
        math.degrees(z)
    ])


def clamp_int16(value):
    """
    限制到 int16 范围，防止极端异常数据导致 STM32 解析溢出。
    """
    value = int(round(value))
    if value > 32767:
        return 32767
    if value < -32768:
        return -32768
    return value


def angle_diff_x10(a, b):
    """
    计算两个角度的差值，单位 0.1 度。

    这里处理 -180~180 度附近的跳变问题。
    例如 179.9° 和 -179.9° 实际只差 0.2°，不是 359.8°。
    """
    diff = a - b
    while diff > 1800:
        diff -= 3600
    while diff < -1800:
        diff += 3600
    return diff


def lowpass_angle_x10(old_value, new_value, alpha):
    """
    角度低通滤波，单位 0.1 度。
    处理跨越 ±180° 的情况。
    """
    diff = angle_diff_x10(new_value, old_value)
    filtered = old_value + alpha * diff

    while filtered > 1800:
        filtered -= 3600
    while filtered < -1800:
        filtered += 3600

    return filtered


def xor_checksum_ascii(text):
    """
    计算文本协议校验和。

    规则：
        对 $ 和 * 之间的内容逐字节异或。
        类似 NMEA 协议的 XOR 校验。

    例如：
        body = "ARUCO,12,1,3,100,0,800,0,0,0,806"
        checksum = xor_checksum_ascii(body)
        packet = "$" + body + "*" + checksum + "#"

    STM32 端：
        1. 找到 '$'
        2. 找到 '#'
        3. 取中间内容
        4. 按 '*' 分成 body 和 checksum
        5. 对 body 做 XOR
        6. 和 checksum 比较
    """
    cs = 0
    for ch in text.encode("ascii", errors="ignore"):
        cs ^= ch
    return f"{cs:02X}"


# ============================================================
# 八、串口相关函数
# ============================================================

def init_serial():
    """
    初始化串口。

    如果串口打开失败，程序不会崩溃。
    它会继续做 ArUco 识别，只是不向 UART 发送。
    """
    if not ENABLE_UART:
        print("[UART] 串口发送未启用")
        return None

    try:
        import serial
    except ImportError:
        print("[UART] 未安装 pyserial，串口不可用")
        print("[UART] 可执行：sudo apt install -y python3-serial")
        return None

    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=0.01,
            write_timeout=0.01
        )
        print(f"[UART] 串口打开成功：{SERIAL_PORT}, {BAUD_RATE}")
        return ser
    except Exception as e:
        print(f"[UART] 串口打开失败：{e}")
        print("[UART] 程序将继续运行，但不会发送串口数据")
        return None


def send_uart(ser, packet):
    """
    发送一包串口数据。

    当前发送的是文本协议：
        $ARUCO,...*CS#\\r\\n

    加 \\r\\n 是为了方便串口助手观察；
    STM32 端正式解析时建议以 '$' 和 '#' 为准，不要只依赖换行。
    """
    if ser is None:
        return

    try:
        ser.write((packet + "\r\n").encode("ascii"))
    except Exception as e:
        print(f"[UART] 发送失败：{e}")


# ============================================================
# 九、ArUco 检测兼容函数
# ============================================================

def create_aruco_detector():
    """
    创建 ArUco 检测器。
    兼容不同 OpenCV 版本。
    """
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)

    try:
        parameters = aruco.DetectorParameters()
    except AttributeError:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        return detector, aruco_dict, parameters

    return None, aruco_dict, parameters


def detect_markers(gray, detector, aruco_dict, parameters):
    """
    检测 ArUco 标签。
    """
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(
            gray,
            aruco_dict,
            parameters=parameters
        )

    return corners, ids, rejected


# ============================================================
# 十、滤波器类
# ============================================================

class PoseFilter:
    """
    ArUco 位姿滤波器。

    功能：
        1. 保存最近几帧原始数据，做中值滤波。
        2. 对中值结果做一阶低通滤波。
        3. 记录最后一次有效识别时间。
        4. 提供 VALID / STALE / LOST 状态。

    数据单位：
        x/y/z/dist：mm
        roll/pitch/yaw：0.1 度
    """

    def __init__(self):
        self.buffers = {
            "x_mm": deque(maxlen=MEDIAN_WINDOW),
            "y_mm": deque(maxlen=MEDIAN_WINDOW),
            "z_mm": deque(maxlen=MEDIAN_WINDOW),
            "dist_mm": deque(maxlen=MEDIAN_WINDOW),
            "roll_x10": deque(maxlen=MEDIAN_WINDOW),
            "pitch_x10": deque(maxlen=MEDIAN_WINDOW),
            "yaw_x10": deque(maxlen=MEDIAN_WINDOW),
        }

        self.filtered = None
        self.last_valid_time = 0.0
        self.last_marker_id = INVALID_ID
        self.last_raw_pose = None

    def reset(self):
        """
        清空滤波器。
        通常在目标长期丢失后可以调用。
        当前程序中暂时不频繁 reset，以便短时丢帧后平稳恢复。
        """
        for buf in self.buffers.values():
            buf.clear()
        self.filtered = None
        self.last_valid_time = 0.0
        self.last_marker_id = INVALID_ID
        self.last_raw_pose = None

    def is_reasonable(self, pose):
        """
        对原始位姿做有效性检查。

        返回：
            True：认为数据可信，可以进入滤波器。
            False：认为数据异常，丢弃。
        """
        dist = pose["dist_mm"]

        if dist < MIN_DIST_MM or dist > MAX_DIST_MM:
            return False

        # 如果已经有上一帧有效数据，则检查跳变
        if self.filtered is not None:
            dx = pose["x_mm"] - self.filtered["x_mm"]
            dy = pose["y_mm"] - self.filtered["y_mm"]
            dz = pose["z_mm"] - self.filtered["z_mm"]
            jump = math.sqrt(dx * dx + dy * dy + dz * dz)

            if jump > MAX_JUMP_MM:
                return False

            if abs(angle_diff_x10(pose["roll_x10"], self.filtered["roll_x10"])) > MAX_ANGLE_JUMP_X10:
                return False

            if abs(angle_diff_x10(pose["pitch_x10"], self.filtered["pitch_x10"])) > MAX_ANGLE_JUMP_X10:
                return False

            if abs(angle_diff_x10(pose["yaw_x10"], self.filtered["yaw_x10"])) > MAX_ANGLE_JUMP_X10:
                return False

        return True

    def update(self, pose, now):
        """
        输入一帧通过有效性检查的原始 pose，更新滤波结果。
        """
        self.last_marker_id = pose["id"]
        self.last_valid_time = now
        self.last_raw_pose = pose

        for key in self.buffers:
            self.buffers[key].append(pose[key])

        median_pose = {
            "id": self.last_marker_id,
            "x_mm": float(np.median(self.buffers["x_mm"])),
            "y_mm": float(np.median(self.buffers["y_mm"])),
            "z_mm": float(np.median(self.buffers["z_mm"])),
            "dist_mm": float(np.median(self.buffers["dist_mm"])),
            "roll_x10": float(np.median(self.buffers["roll_x10"])),
            "pitch_x10": float(np.median(self.buffers["pitch_x10"])),
            "yaw_x10": float(np.median(self.buffers["yaw_x10"])),
        }

        if self.filtered is None:
            self.filtered = median_pose
        else:
            # 位置和距离低通
            for key in ("x_mm", "y_mm", "z_mm", "dist_mm"):
                self.filtered[key] = (
                    FILTER_ALPHA_POS * median_pose[key]
                    + (1.0 - FILTER_ALPHA_POS) * self.filtered[key]
                )

            # 角度低通
            for key in ("roll_x10", "pitch_x10", "yaw_x10"):
                self.filtered[key] = lowpass_angle_x10(
                    self.filtered[key],
                    median_pose[key],
                    FILTER_ALPHA_ANGLE
                )

            self.filtered["id"] = self.last_marker_id

    def get_status_and_pose(self, now):
        """
        根据最后一次有效识别时间，返回当前状态和要发送的数据。

        返回：
            status, pose

        status:
            1 = VALID/STALE 使用滤波值
            0 = LOST 无有效数据
        """
        if self.filtered is None:
            return STATUS_LOST, None

        age = now - self.last_valid_time

        if age <= STALE_TIMEOUT_S:
            # 注意：
            # 是否是 VALID 由主循环根据“当前循环是否刚刚识别成功”决定。
            # 这里先返回 STALE/可用数据，主循环可覆盖为 VALID。
            return STATUS_STALE, self.filtered

        if age <= LOST_TIMEOUT_S:
            return STATUS_STALE, self.filtered

        return STATUS_LOST, None


# ============================================================
# 十一、数据包生成函数
# ============================================================

def make_packet(seq, status, pose):
    """
    生成发送给 STM32 的文本协议数据包。

    协议格式：
        $ARUCO,seq,status,id,x_mm,y_mm,z_mm,roll_x10,pitch_x10,yaw_x10,dist_mm*CS#

    字段说明：
        seq:
            包序号，0~65535 循环递增。
            STM32 可用它判断是否丢包。

        status:
            0 = LOST，目标丢失，后面数据无效。
            1 = VALID，当前识别有效。
            2 = STALE，短时丢帧，使用上一帧滤波数据。

        id:
            ArUco 标签 ID。
            LOST 时为 65535。

        x_mm, y_mm, z_mm:
            标签中心相对摄像头坐标，单位 mm。
            坐标系与 OpenCV tvec 一致：
                X 正方向：图像右方
                Y 正方向：图像下方
                Z 正方向：摄像头前方

        roll_x10, pitch_x10, yaw_x10:
            欧拉角 ×10，单位 0.1 度。
            例如 123 表示 12.3 度。

        dist_mm:
            标签中心到摄像头的直线距离，单位 mm。

        CS:
            XOR 校验，2 位十六进制。
            校验范围是 $ 和 * 之间的 body。
    """
    if status == STATUS_LOST or pose is None:
        marker_id = INVALID_ID
        x_mm = y_mm = z_mm = 0
        roll_x10 = pitch_x10 = yaw_x10 = 0
        dist_mm = 0
    else:
        marker_id = int(pose["id"])
        x_mm = clamp_int16(pose["x_mm"])
        y_mm = clamp_int16(pose["y_mm"])
        z_mm = clamp_int16(pose["z_mm"])
        roll_x10 = clamp_int16(pose["roll_x10"])
        pitch_x10 = clamp_int16(pose["pitch_x10"])
        yaw_x10 = clamp_int16(pose["yaw_x10"])
        dist_mm = clamp_int16(pose["dist_mm"])

    body = (
        f"ARUCO,{seq},{status},{marker_id},"
        f"{x_mm},{y_mm},{z_mm},"
        f"{roll_x10},{pitch_x10},{yaw_x10},{dist_mm}"
    )

    cs = xor_checksum_ascii(body)

    return f"${body}*{cs}#"


def pose_from_opencv(marker_id, rvec, tvec):
    """
    将 OpenCV 的 rvec/tvec 转换为工程数据。

    输入：
        marker_id: ArUco ID
        rvec: 旋转向量
        tvec: 平移向量，单位 m

    输出：
        pose 字典，单位：
            x/y/z/dist：mm
            roll/pitch/yaw：0.1度
    """
    x_m = float(tvec[0])
    y_m = float(tvec[1])
    z_m = float(tvec[2])

    dist_m = math.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)

    roll_deg = float(euler_angles[0])
    pitch_deg = float(euler_angles[1])
    yaw_deg = float(euler_angles[2])

    pose = {
        "id": int(marker_id),

        "x_mm": int(round(x_m * 1000.0)),
        "y_mm": int(round(y_m * 1000.0)),
        "z_mm": int(round(z_m * 1000.0)),
        "dist_mm": int(round(dist_m * 1000.0)),

        "roll_x10": int(round(roll_deg * 10.0)),
        "pitch_x10": int(round(pitch_deg * 10.0)),
        "yaw_x10": int(round(yaw_deg * 10.0)),
    }

    return pose


# ============================================================
# 十二、主程序
# ============================================================

def main():
    ser = init_serial()

    # 使用 V4L2 打开 Linux 摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[CAMERA] 摄像头打开失败：{CAMERA_INDEX}")
        sys.exit(1)

    # 尝试设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 如果摄像头支持 MJPG，可以降低 USB 摄像头延迟
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    detector, aruco_dict, parameters = create_aruco_detector()
    pose_filter = PoseFilter()

    seq = 0
    send_counter = 0

    last_send_time = time.monotonic()
    last_valid_this_loop = False

    print("[SYSTEM] ArUco UART 最终调试版启动")
    print(f"[SYSTEM] CAMERA={CAMERA_INDEX}, {FRAME_WIDTH}x{FRAME_HEIGHT}@{CAMERA_FPS}")
    print(f"[SYSTEM] SHOW_WINDOW={SHOW_WINDOW}")
    print(f"[SYSTEM] UART={SERIAL_PORT}, {BAUD_RATE}, ENABLE_UART={ENABLE_UART}")
    print(f"[SYSTEM] SEND_HZ={SEND_HZ}")
    print("[SYSTEM] 协议：$ARUCO,seq,status,id,x_mm,y_mm,z_mm,roll_x10,pitch_x10,yaw_x10,dist_mm*CS#")
    print("[SYSTEM] status: 0=LOST, 1=VALID, 2=STALE")
    print("[SYSTEM] Ctrl+C 退出")

    try:
        while True:
            now = time.monotonic()
            last_valid_this_loop = False

            ret, frame = cap.read()

            if not ret:
                # 摄像头读取失败时，不退出，让程序继续尝试
                print("[CAMERA] 读取图像失败")
                time.sleep(0.005)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners, ids, rejected = detect_markers(
                    gray,
                    detector,
                    aruco_dict,
                    parameters
                )

                if ids is not None and len(ids) > 0:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        corners,
                        MARKER_SIZE,
                        camera_matrix,
                        dist_coeffs
                    )

                    # 选择一个目标标签：
                    #   1. 如果设置了 TARGET_IDS，只接受白名单中的 ID
                    #   2. 如果有多个目标，选择距离最近的
                    best_index = None
                    best_dist = None

                    for i in range(len(ids)):
                        marker_id = int(ids[i][0])

                        if TARGET_IDS is not None and marker_id not in TARGET_IDS:
                            continue

                        tvec = tvecs[i][0]
                        dist = float(np.linalg.norm(tvec))

                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_index = i

                    if best_index is not None:
                        marker_id = int(ids[best_index][0])
                        rvec = rvecs[best_index][0]
                        tvec = tvecs[best_index][0]

                        raw_pose = pose_from_opencv(marker_id, rvec, tvec)

                        if pose_filter.is_reasonable(raw_pose):
                            pose_filter.update(raw_pose, now)
                            last_valid_this_loop = True
                        else:
                            # 异常帧直接丢弃，不进入滤波器
                            pass

                        if SHOW_WINDOW:
                            aruco.drawDetectedMarkers(frame, corners, ids)
                            cv2.drawFrameAxes(
                                frame,
                                camera_matrix,
                                dist_coeffs,
                                rvec,
                                tvec,
                                0.05
                            )

            # 固定频率发布
            now = time.monotonic()

            if now - last_send_time >= SEND_INTERVAL:
                status, pose = pose_filter.get_status_and_pose(now)

                # 如果本轮刚刚识别到并更新了滤波器，则标记为 VALID
                if last_valid_this_loop and pose is not None:
                    status = STATUS_VALID

                packet = make_packet(seq, status, pose)
                send_uart(ser, packet)

                send_counter += 1
                if PRINT_TO_CONSOLE and (send_counter % PRINT_DECIMATION == 0):
                    print(packet)

                seq = (seq + 1) & 0xFFFF
                last_send_time = now

            if SHOW_WINDOW and ret:
                cv2.imshow("ArUco UART Sensor Node", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[SYSTEM] 用户按 q 退出")
                    break

    except KeyboardInterrupt:
        print("[SYSTEM] Ctrl+C 退出")

    finally:
        cap.release()

        if ser is not None:
            ser.close()
            print("[UART] 串口已关闭")

        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        print("[SYSTEM] 程序结束")


if __name__ == "__main__":
    main()