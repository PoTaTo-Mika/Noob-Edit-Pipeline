import json
import cv2
import numpy as np
from PIL import Image
import math

# ==========================================
# OpenPose COCO-17 拓扑结构定义
# ==========================================

# 关键点索引:
# 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear, 
# 5:L_Shoulder, 6:R_Shoulder, 7:L_Elbow, 8:R_Elbow, 9:L_Wrist, 10:R_Wrist, 
# 11:L_Hip, 12:R_Hip, 13:L_Knee, 14:R_Knee, 15:L_Ankle, 16:R_Ankle

# 连接对 (Start, End)
POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), # 肩膀连接
    (5, 7), (7, 9), # 左臂
    (6, 8), (8, 10), # 右臂
    (11, 12), # 臀部连接
    (5, 11), (6, 12), # 躯干
    (11, 13), (13, 15), # 左腿
    (12, 14), (14, 16)  # 右腿
]

# 对应肢体的颜色 (BGR 格式，OpenCV 用)
# 这一组颜色是 ControlNet 训练时通常使用的标准色
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85)
]

def draw_pose_from_json(json_path):
    """
    读取 OpenPose JSON 并绘制成 ControlNet 可用的 RGB 图片
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 获取画布尺寸
    canvas_w = data.get("canvas_width", 512)
    canvas_h = data.get("canvas_height", 784)
    
    # 创建纯黑背景
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    people = data.get("people", [])
    if not people:
        return Image.fromarray(canvas)

    for person in people:
        keypoints = person.get("pose_keypoints_2d", [])
        
        # 提取关键点
        points = []
        # COCO 格式通常有 17 个点，数据长度 51
        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i+1]
            c = keypoints[i+2] # 置信度
            
            # 坐标转换：归一化(0-1) -> 像素坐标
            if c > 0.05: # 简单的置信度过滤
                px = int(x * canvas_w)
                py = int(y * canvas_h)
                points.append((px, py))
            else:
                points.append(None)

        # 绘制骨骼 (Limbs)
        for idx, pair in enumerate(POSE_PAIRS):
            partA = pair[0]
            partB = pair[1]

            if partA < len(points) and partB < len(points):
                pA = points[partA]
                pB = points[partB]

                if pA and pB:
                    # 使用特定颜色绘制肢体
                    col = COLORS[idx % len(COLORS)]
                    cv2.line(canvas, pA, pB, col, 3, cv2.LINE_AA)

        # 绘制关节 (Joints)
        for i, point in enumerate(points):
            if point:
                # 关节颜色逻辑：通常也根据索引变化，或者统一
                col = COLORS[i % len(COLORS)]
                cv2.circle(canvas, point, 4, col, -1, cv2.LINE_AA)

    # OpenCV (BGR) -> PIL (RGB)
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))