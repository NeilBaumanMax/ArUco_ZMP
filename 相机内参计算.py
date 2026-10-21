import cv2
import numpy as np

# ================= 配置区域 (保持不变) =================
# 1. 棋盘格的【内部角点】行列数
CHECKERBOARD = (9, 6)

# 2. 每个格子的实际物理边长 (单位: mm)
SQUARE_SIZE = 25 
# ======================================================

def run_calibration():
    # 1. 准备对象点 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE  # 转换为实际物理单位

    # 存储 3D 点和 2D 点
    objpoints = [] # 真实世界坐标
    imgpoints = [] # 图像像素坐标

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    
    # 【关键修改】强制设置分辨率为 1280x720
    # 必须与你最终运行识别程序的分辨率完全一致！
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 打印确认一下当前分辨率
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"当前相机分辨率: {int(actual_w)} x {int(actual_h)}")
    if int(actual_w) != 1280:
        print("警告：摄像头不支持 1280x720，可能导致标定误差！")

    print("\n【开始标定】")
    print("1. 拿棋盘格在屏幕各个角落移动（上下左右前后，以及轻微倾斜）")
    print("2. 确保棋盘格稳定检测（出现彩色连接线）时，按 's' 保存")
    print("3. 至少拍 15 张以上，覆盖画面不同位置")
    print("4. 按 'c' 开始计算")
    print("5. 按 'q' 退出")

    count = 0
    
    # 亚像素精确化参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OV9281 可能是单通道，处理一下
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # 转彩色方便画图显示

        display_frame = frame.copy()

        # 实时检测并画出角点（预览模式）
        # 这样你就知道什么时候按 S 是有效的
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if found:
            # 画出角点给你看
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, found)
            status_text = "Ready to Capture (Press 's')"
            color = (0, 255, 0)
        else:
            status_text = "No Board Detected"
            color = (0, 0, 255)

        cv2.putText(display_frame, f"Photos: {count} | {status_text}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Calibration', display_frame)
        key = cv2.waitKey(1) & 0xFF

        # --- 按 's' 保存 ---
        if key == ord('s'):
            if found:
                objpoints.append(objp)
                
                # 亚像素精确化
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                count += 1
                print(f"已保存第 {count} 张")
                
                # 闪烁一下屏幕表示保存成功
                cv2.rectangle(display_frame, (0,0), (1280,720), (255,255,255), -1)
                cv2.imshow('Calibration', display_frame)
                cv2.waitKey(50)
            else:
                print("未检测到棋盘，无法保存！")

        # --- 按 'c' 计算 ---
        elif key == ord('c'):
            if count < 10:
                print("照片太少！为了准度，请至少拍 10-15 张。")
                continue
            
            print("\n正在计算中... (可能需要几秒钟)")
            
            # 这里的 flags 可以控制是否计算畸变
            # 即使是无畸变镜头，建议还是让它正常算，之后我们可以只取内参矩阵
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            print("\n" + "="*40)
            print(f"【标定结果】")
            print(f"重投影误差 (越小越好，小于 0.5 算优秀): {ret:.4f}")
            print("-" * 20)
            print("相机内参矩阵 (Camera Matrix):")
            print("请把下面这行 array 完整复制到你的测距代码中：")
            print(np.array_repr(mtx))
            print("-" * 20)
            print("畸变系数 (Distortion):")
            print(np.array_repr(dist))
            print("="*40 + "\n")
            
            # 计算完退出
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration()