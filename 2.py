import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time

# ================= 配置区域 =================
# 1. 跌倒阈值设置
#    当 俯仰角 (Pitch) 的绝对值超过这个度数，视为摔倒
FALL_ANGLE_THRESHOLD = 45.0  # 单位：度

# 2. 相机与标签设置 (沿用之前的)
ARUCO_DICT_TYPE = aruco.DICT_6X6_250
MARKER_SIZE = 0.1  # 标签大小 0.1米 (100mm)

# 3. 相机内参 (请替换为你标定后的真实数据)
camera_matrix = np.array([
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
# ===========================================

def rotation_matrix_to_euler_angles(R):
    """计算欧拉角 (Roll, Pitch, Yaw)"""
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
    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])

def main():
    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = aruco.DetectorParameters()

    print("启动...")
    print(f"当前设定：倾斜角度超过 {FALL_ANGLE_THRESHOLD} 度视为摔倒")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # 默认状态：安全 (绿色)
        status_color = (0, 255, 0) 
        status_text = "NORMAL"

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)

            for i in range(len(ids)):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # 1. 计算角度
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
                
                # 获取 Pitch (俯仰角) - 这是判断前后摔倒的关键
                # 注意：根据你的标签贴法（横贴还是竖贴），这里可能是 pitch 也可能是 roll
                # 通常竖着贴在肚子上，人前倾时变化的是 Pitch (euler_angles[1])
                roll = euler_angles[0]
                pitch = euler_angles[1]
                yaw = euler_angles[2]

                # 2. 核心逻辑：跌倒检测
                # 我们取 Pitch 的绝对值，因为人可能向前摔(+)也可能向后仰(-)
                if abs(pitch) > FALL_ANGLE_THRESHOLD:
                    status_text = "WARNING: FALL DETECTED!"
                    status_color = (0, 0, 255) # 红色
                    
                    # 在终端打印报警
                    print(f"!!! 检测到摔倒 !!! Pitch角度: {pitch:.1f}°")

                # --- 画面显示 ---
                # 画出坐标轴
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                
                # 显示实时角度数据
                info_txt = f"Pitch: {pitch:.1f}  Threshold: {FALL_ANGLE_THRESHOLD}"
                cv2.putText(frame, info_txt, (10, frame.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 3. 全局报警反馈
        if status_text != "NORMAL":
            # 如果检测到摔倒，在屏幕四周画一个粗红框
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), status_color, 10)
            # 在屏幕中央写大字
            cv2.putText(frame, status_text, (50, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 4)

        cv2.imshow('Walker Fall Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()