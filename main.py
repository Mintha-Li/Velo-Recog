import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to calculate speed based on optical flow
# 根据光流计算速度的函数
def calculate_speed(good_new, good_old, pixel_to_meter_ratio, frame_rate):
    dx_dy = (good_new - good_old) * pixel_to_meter_ratio  # Convert pixel displacement to meters 转换像素位移为米
    distances = np.linalg.norm(dx_dy, axis=1)  # Calculate Euclidean distances 计算欧几里得距离
    return distances * frame_rate  # Speed in meters/second 速度（米/秒）


# Function to calculate displacement based on optical flow
# 根据光流计算位移的函数
def calculate_displacement(good_new, good_old, pixel_to_meter_ratio):
    x_y = good_new * pixel_to_meter_ratio  # Convert pixel displacement to meters 转换像素位移为米
    displacements = x_y
    return displacements  # Return displacements in meters 返回位移（米）


# Function to initialize the video writer for saving output video
# 初始化视频写入器以保存输出视频的函数
def initialize_video_writer(output_path, frame_shape, frame_rate):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format AVI格式的编码器
    return cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_shape[1], frame_shape[0]))


def calculate_pixel_to_meter_ratio(sensor_width_mm, sensor_height_mm, resolution_width_px, resolution_height_px, focal_length_mm, distance_to_object_m):
    # Calculate the field of view in meters at the given distance
    # 在给定距离下计算视场的米数
    field_of_view_width_m = (sensor_width_mm * distance_to_object_m) / focal_length_mm
    field_of_view_height_m = (sensor_height_mm * distance_to_object_m) / focal_length_mm

    # Calculate the pixel-to-meter ratio
    # 计算像素到米的比率
    pixel_to_meter_ratio_width = field_of_view_width_m / resolution_width_px
    pixel_to_meter_ratio_height = field_of_view_height_m / resolution_height_px

    # Return the average pixel-to-meter ratio
    # 返回平均像素到米的比率
    return (pixel_to_meter_ratio_width + pixel_to_meter_ratio_height) / 2


# Main function to process the video and calculate displacements and speeds using optical flow
# 主函数，用于处理视频并使用光流法计算位移和速度
def process_video(video_path, output_path, frame_rate, pixel_to_meter_ratio):
    """
    Processes a video to calculate optical flow, displacement, and speed, and generates a combined output video
    with visualizations and labels.
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        frame_rate (float): Frame rate of the video in frames per second.
        pixel_to_meter_ratio (float): Conversion ratio from pixels to meters.
    Returns:
        list: A list of tuples containing time, displacement (x, y), and speed for each frame.
              Each tuple is in the format (time, (avg_displacement_x, avg_displacement_y), avg_speed).
    Notes:
        - The function uses OpenCV to process the video frame by frame.
        - Optical flow is calculated using the Lucas-Kanade method.
        - Displacement and speed are calculated based on the movement of tracked points.
        - The output video includes the original frame, grayscale frame, processed frame with optical flow tracks,
          and labels for visualization.
        - Press the 'Esc' key during execution to stop the video processing early.
    """
    cap = cv2.VideoCapture(video_path)  # Open the video file 打开视频文件
    if not cap.isOpened():
        print("Unable to open video file 无法打开视频文件")
        return []

    # Read the first frame
    # 读取第一帧
    ret, old_frame = cap.read()
    if not ret:
        print("Unable to read video frame 无法读取视频帧")
        cap.release()
        return []

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale 转换为灰度图
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)  # Detect corners 检测角点

    mask = np.zeros_like(old_frame)  # Mask for drawing optical flow 用于绘制光流的掩码

    # Initialize video writer for combined output
    # 初始化视频写入器以保存组合输出
    combined_frame_shape = (old_frame.shape[0] + 50, old_frame.shape[1] * 3)  # Include label height in frame size 包括标签高度在帧大小中
    out = initialize_video_writer(output_path, combined_frame_shape, frame_rate)

    time_displacement_sequence = []  # List to store time and displacement sequence 用于存储时间和位移序列

    frame_count = 0  # Frame counter 帧计数器
    while True:
        ret, frame = cap.read()  # Read the next frame 读取下一帧
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale 转换为灰度图

        # Calculate optical flow
        # 计算光流
        p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Select good points
        # 选择好的点
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        # 绘制轨迹
        processed_frame = frame.copy()
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            processed_frame = cv2.circle(processed_frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img_with_tracks = cv2.add(processed_frame, mask)  # Overlay the mask on the frame 将掩码叠加到帧上

        # Calculate displacement and speed
        # 计算位移和速度
        displacements = calculate_displacement(good_new, good_old, pixel_to_meter_ratio)
        speeds = calculate_speed(good_new, good_old, pixel_to_meter_ratio, frame_rate)
        avg_displacement_x = np.mean(displacements[:, 0]) if len(displacements) > 0 else 0
        avg_displacement_y = np.mean(displacements[:, 1]) if len(displacements) > 0 else 0
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0

        # Append time, displacement (x, y), and speed to the sequence
        # 将时间、位移（x, y）和速度添加到序列中
        time = frame_count / frame_rate
        time_displacement_sequence.append((time, (avg_displacement_x, avg_displacement_y), avg_speed))

        # Combine original, grayscale, and processed frames
        # 组合原始帧、灰度帧和处理后的帧
        grayscale_colored = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for visualization 转换灰度图为BGR以便可视化
        combined_frame = np.hstack((frame, grayscale_colored, img_with_tracks))  # Horizontally stack frames 横向堆叠帧

        # Add labels below each video
        # 在每个视频下面添加标签
        label_frame = np.zeros((50, combined_frame.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(label_frame, "Grayscale", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(label_frame, "Processed", (2 * frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        combined_frame_with_labels = np.vstack((combined_frame, label_frame))  # Stack labels below the combined frame

        # Add speed label to the top-right corner of the processed frame
        # 在处理后的帧的右上角添加速度标签
        cv2.putText(combined_frame_with_labels, f"Speed: {avg_speed:.2f} m/s",
                    (2 * frame.shape[1] + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(combined_frame_with_labels)  # Write the combined frame to the output video 将组合帧写入输出视频
        cv2.imshow('Optical Flow Visualization', combined_frame_with_labels)  # Show the combined frame 显示组合帧

        if cv2.waitKey(30) & 0xFF == 27:  # Exit on 'Esc' key 按下 'Esc' 键退出
            break

        old_gray = frame_gray.copy()  # Update the previous frame 更新上一帧
        p0 = good_new.reshape(-1, 1, 2)  # Update the points 更新点
        frame_count += 1  # Increment frame counter 增加帧计数器

    cap.release()  # Release the video capture 释放视频捕获
    out.release()  # Release the video writer 释放视频写入器
    cv2.destroyAllWindows()  # Close all OpenCV windows 关闭所有OpenCV窗口

    return time_displacement_sequence  # Return the time-displacement sequence 返回时间位移序列


# Function to calculate speed-time sequence from time-displacement sequence
# 根据时间位移序列计算时间速度序列的函数
def calculate_speed_time_sequence(time_displacement_sequence):
    speed_time_sequence = []  # List to store time and speed sequence 用于存储时间和速度序列

    for i in range(1, len(time_displacement_sequence)):
        time_prev, displacement_prev, _ = time_displacement_sequence[i - 1]
        time_curr, displacement_curr, _ = time_displacement_sequence[i]

        # Calculate speed as the Euclidean distance change over time
        # 计算速度为欧几里得距离随时间的变化
        time_diff = time_curr - time_prev
        if time_diff > 0:
            displacement_diff = np.linalg.norm(np.array(displacement_curr) - np.array(displacement_prev))
            speed = displacement_diff / time_diff
            speed_time_sequence.append((time_curr, speed))

    return speed_time_sequence  # Return the speed-time sequence 返回时间速度序列


# Function to calculate acceleration-time sequence from speed-time sequence
# 根据速度时间序列计算加速度时间序列的函数
def calculate_acceleration_time_sequence(speed_time_sequence):
    acceleration_time_sequence = []  # List to store time and acceleration sequence 用于存储时间和加速度序列

    for i in range(1, len(speed_time_sequence)):
        time_prev, speed_prev = speed_time_sequence[i - 1]
        time_curr, speed_curr = speed_time_sequence[i]

        # Calculate acceleration as the change in speed over time
        # 计算加速度为速度随时间的变化
        time_diff = time_curr - time_prev
        if time_diff > 0:
            acceleration = (speed_curr - speed_prev) / time_diff
            acceleration_time_sequence.append((time_curr, acceleration))

    return acceleration_time_sequence  # Return the acceleration-time sequence 返回加速度时间序列


# Function to plot time-displacement, time-speed, and time-acceleration curves
# 绘制时间-位移、时间-速度和时间-加速度曲线的函数
def plot_motion_curves(time_displacement_sequence, speed_time_sequence, acceleration_time_sequence, save_path=None):
    # Extract data for plotting
    times_displacement = [t for t, _, _ in time_displacement_sequence]
    displacements_x = [d[0] for _, d, _ in time_displacement_sequence]
    displacements_y = [d[1] for _, d, _ in time_displacement_sequence]

    times_speed = [t for t, _ in speed_time_sequence]
    speeds = [s for _, s in speed_time_sequence]

    times_acceleration = [t for t, _ in acceleration_time_sequence]
    accelerations = [a for _, a in acceleration_time_sequence]

    # Create subplots
    plt.figure(figsize=(15, 10))

    # Plot time-displacement curve
    plt.subplot(3, 1, 1)
    plt.plot(times_displacement, displacements_x, label="Displacement X (m)", color="blue")
    plt.plot(times_displacement, displacements_y, label="Displacement Y (m)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Time-Displacement Curve")
    plt.grid()
    plt.legend()

    # Plot time-speed curve
    plt.subplot(3, 1, 2)
    plt.plot(times_speed, speeds, label="Speed (m/s)", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Time-Speed Curve")
    plt.grid()
    plt.legend()

    # Plot time-acceleration curve
    plt.subplot(3, 1, 3)
    plt.plot(times_acceleration, accelerations, label="Acceleration (m/s²)", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Time-Acceleration Curve")
    plt.grid()
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.show()
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    video_path = "Video.avi"  # Input video file path 输入视频文件路径
    output_path = "Output.avi"  # Output video file path 输出视频文件路径
    result_path = "Result.png"  # Result text file path 结果文本文件路径
    cap = cv2.VideoCapture(video_path)  # Open the video file 打开视频文件
    if not cap.isOpened():
        print("Unable to open video file 无法打开视频文件")
        exit()

    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video 获取视频的帧率
    cap.release()  # Release the video capture 释放视频捕获

    # Camera parameters
    # 相机参数
    sensor_width_mm = 5.70  # Sensor width in mm 传感器宽度（毫米）
    sensor_height_mm = 4.28  # Sensor height in mm 传感器高度（毫米）
    resolution_width_px = 2592  # Resolution width in pixels 分辨率宽度（像素）
    resolution_height_px = 1944  # Resolution height in pixels 分辨率高度（像素）
    focal_length_mm = 5.0  # Focal length in mm 焦距（毫米）
    distance_to_object_m = 2.35  # Distance to the object in meters 物体到相机的距离（米）

    # Calculate pixel-to-meter ratio
    pixel_to_meter_ratio = calculate_pixel_to_meter_ratio(
        sensor_width_mm, sensor_height_mm, resolution_width_px, resolution_height_px, focal_length_mm, distance_to_object_m
    )

    time_displacement_sequence = process_video(video_path, output_path, frame_rate, pixel_to_meter_ratio)  # Process the video and get time-displacement sequence 处理视频并获取时间位移序列

    speed_time_sequence = calculate_speed_time_sequence(time_displacement_sequence)  # Calculate speed-time sequence 计算速度时间序列

    acceleration_time_sequence = calculate_acceleration_time_sequence(speed_time_sequence)  # Calculate acceleration-time sequence 计算加速度时间序列

    # Plot the motion curves
    plot_motion_curves(time_displacement_sequence, speed_time_sequence, acceleration_time_sequence, save_path=result_path)  # 绘制运动曲线
