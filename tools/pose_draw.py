import cv2
import numpy as np
from PIL import Image

# ==========================================
# SDXL 黄金分辨率分桶 (Buckets)
# ==========================================
SDXL_BUCKETS = [
    (1024, 1024), 
    (1152, 896),  (896, 1152),
    (1216, 832),  (832, 1216), # NoobAI 常用 (竖图)
    (1344, 768),  (768, 1344),
    (1536, 640),  (640, 1536),
]

def get_optimal_bucket(orig_w, orig_h):
    """计算最接近的 SDXL 分辨率"""
    if orig_h == 0 or orig_w == 0:
        return (1024, 1024)
    
    target_ar = orig_w / orig_h
    best_bucket = (1024, 1024)
    min_diff = float('inf')

    for (bw, bh) in SDXL_BUCKETS:
        bucket_ar = bw / bh
        diff = abs(bucket_ar - target_ar)
        if diff < min_diff:
            min_diff = diff
            best_bucket = (bw, bh)
    return best_bucket

# ==========================================
# OpenPose 绘制逻辑
# ==========================================
POSE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]
COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
    (255, 0, 170), (255, 0, 85)
]

def draw_pose_from_data(data_dict):
    """
    接收内存中的 JSON Dict，绘制并 Resize 到 SDXL Bucket 尺寸
    """
    # 1. 获取原图尺寸 (默认 1024 防报错)
    canvas_w = data_dict.get("canvas_width", 1024)
    canvas_h = data_dict.get("canvas_height", 1024)
    
    # 2. 绘制画布 (必须在原图尺寸上绘制，否则骨架位置会偏)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    people = data_dict.get("people", [])
    if people:
        for person in people:
            keypoints = person.get("pose_keypoints_2d", [])
            points = []
            
            # 解析关键点
            for i in range(0, len(keypoints), 3):
                x = keypoints[i]
                y = keypoints[i+1]
                c = keypoints[i+2]
                
                if c > 0.05: # 置信度阈值
                    # 兼容归一化坐标(0-1)和绝对坐标
                    px = int(x * canvas_w) if x <= 1.5 else int(x)
                    py = int(y * canvas_h) if y <= 1.5 else int(y)
                    # 边界保护
                    px = max(0, min(px, canvas_w - 1))
                    py = max(0, min(py, canvas_h - 1))
                    points.append((px, py))
                else:
                    points.append(None)

            # 绘制肢体
            for idx, pair in enumerate(POSE_PAIRS):
                if pair[0] < len(points) and pair[1] < len(points):
                    pA, pB = points[pair[0]], points[pair[1]]
                    if pA and pB:
                        cv2.line(canvas, pA, pB, COLORS[idx % len(COLORS)], 3, cv2.LINE_AA)

            # 绘制关节
            for i, point in enumerate(points):
                if point:
                    cv2.circle(canvas, point, 4, COLORS[i % len(COLORS)], -1, cv2.LINE_AA)

    # 3. 转换并缩放
    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    
    # 计算目标尺寸
    target_w, target_h = get_optimal_bucket(canvas_w, canvas_h)
    
    # 使用 Lanczos 进行高质量缩放
    img_resized = img_pil.resize((target_w, target_h), Image.LANCZOS)
    
    return img_resized