import cv2, sys, json
p = "data/videos/building.mp4" # 给出需要处理的视频路径
cap = cv2.VideoCapture(p)
assert cap.isOpened(), f"Open failed: {p}"

info = {
    "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    "fps":    float(cap.get(cv2.CAP_PROP_FPS)),
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
}
cap.release()
print(json.dumps(info, indent=2))
