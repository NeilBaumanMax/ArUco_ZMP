"""
文件用途：ArUco 位姿检测（使用标定内参 + 畸变系数）
- 已写入标定内参与真实畸变系数 [k1, k2, p1, p2, k3]，分辨率 1280x720
- 画面显示标签 XYZ 位置、欧拉角、直线距离并绘制坐标轴
- 运行：python aruco_distortion.py，按 q 退出
"""
import cv2
import cv2.aruco as aruco
import numpy as np
import math


# 一、配置区域 
# =================================================================

# 1. 设置使用的 ArUco 字典
#    DICT_6X6_250 表示：6x6 的网格，字典里总共有 250 个不同的 ID
ARUCO_DICT_TYPE = aruco.DICT_6X6_250

# 2. 设置标签的实际物理尺寸
#    这里写 0.1 表示 0.1米 (100mm)
MARKER_SIZE = 0.1  

# 3. 设置相机内参 (Camera Matrix)
#    虽然无畸变，但焦距(fx,fy)和光心(cx,cy)依然要用标定算出来的，这样Z轴测距才准
#    格式：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
camera_matrix = np.array([
    [730.26272565,   0.        , 658.92852375],
    [  0.        , 739.47588552, 302.48694463],
    [  0.        ,   0.        ,   1.        ]
], dtype=np.float32)

# 4. 设置畸变系数 (Distortion Coefficients)
#    是无畸变相机，这里全部设为 0
dist_coeffs = np.array([[ 0.14970004, -0.25590836,  0.00061437, -0.0007279 ,  0.12281265]])


# 二、旋转向量转换为欧拉角的函数
# =================================================================

def rotation_matrix_to_euler_angles(R):
    """
    功能：将旋转矩阵转换为欧拉角 (Roll, Pitch, Yaw)
    输入：3x3 旋转矩阵
    输出：[Roll, Pitch, Yaw] (单位：度)
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

    # 转换为角度制
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])


# 三、MAIN
# =================================================================

def main():
    # 1. 初始化摄像头 (0 代表默认摄像头)
    cap = cv2.VideoCapture(0)
    
    # --- 设置分辨率和帧率 ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # ---------------------------

    # 获取字典和检测参数
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = aruco.DetectorParameters()

    print("运行中...按 'Q' 键退出")

    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        if not ret:
            break

        # 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 检测标记
        # corners: 列表，包含所有检测到的标记的角点坐标 (像素值)
        # ids: 列表，包含检测到的标记 ID
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # 如果检测到了至少一个标记
        if ids is not None:
            # 在画面上画出绿色的方框
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 3. 位姿估计 (Pose Estimation)
            # rvecs: 旋转向量列表 (每个标记一个)
            # tvecs: 平移向量列表 (每个标记一个)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            # 遍历每一个检测到的标记 (放多个标签)
            for i in range(len(ids)):
                # 获取当前标记的 旋转向量 和 平移向量
                # [0] 是因为 opencv 返回的是一个嵌套数组
                rvec = rvecs[i][0] 
                tvec = tvecs[i][0]

                # 变量说明：
        
                # 【tvec (Translation Vector)】：平移向量
                # 含义：标记中心点 相对于 摄像头光心(0,0,0) 的位置
                # 单位：和 MARKER_SIZE 一样 (这里是米)
                # 坐标系：
                #   X轴 (+): 摄像头画面向右
                #   Y轴 (+): 摄像头画面向下
                #   Z轴 (+): 摄像头画面向正前方 (深度)
                x = tvec[0]
                y = tvec[1]
                z = tvec[2] 

                # 【rvec (Rotation Vector)】：旋转向量
                # 含义：表示物体绕着某个轴转了多少弧度。它是紧凑的数学表示，很难直观理解。
                # 所以我们需要把它转成“欧拉角”或“旋转矩阵”。
                
                # 转换步骤：rvec -> 旋转矩阵 -> 欧拉角
                rotation_matrix, _ = cv2.Rodrigues(rvec) 
                euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
                
                # 【Euler Angles】：欧拉角 (度)
                # Roll (翻滚): 绕 Z 轴旋转
                # Pitch (俯仰): 绕 X 轴旋转 
                # Yaw (偏航): 绕 Y 轴旋转 
                roll_deg = euler_angles[0]
                pitch_deg = euler_angles[1]
                yaw_deg = euler_angles[2]

                # --- 绘制和显示 ---

                # 1. 在标记上画出 XYZ 坐标轴 (红:X, 绿:Y, 蓝:Z)    RGB -> XYZ
            
                # 0.05 表示坐标轴的长度为 0.05米 (5cm)
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                # 2. 准备要在屏幕上显示的文字
                # 距离 (勾股定理计算直线距离)
                distance = np.sqrt(x**2 + y**2 + z**2)
                
                # 格式化字符串 (.2f 表示保留两位小数)
                text_pos = f"X:{x*100:.1f} Y:{y*100:.1f} Z:{z*100:.1f} cm"
                text_rot = f"R:{roll_deg:.0f} P:{pitch_deg:.0f} Y:{yaw_deg:.0f} deg"
                text_dist = f"Dist: {distance*100:.1f} cm"

                # 3. 把文字写在标记旁边
                # 获取标记左上角的像素坐标，作为文字显示位置
                top_left_corner = corners[i][0][0]
                text_org_x = int(top_left_corner[0])
                text_org_y = int(top_left_corner[1])

                cv2.putText(frame, text_pos, (text_org_x, text_org_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, text_rot, (text_org_x, text_org_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, text_dist, (text_org_x, text_org_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

                # 在控制台打印
                print(f"[ID:{ids[i][0]}] X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m | Roll={roll_deg:.1f}, Pitch={pitch_deg:.1f}, Yaw={yaw_deg:.1f}")

        # 显示最终画面
        cv2.imshow('ArUco Optimized Demo', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()