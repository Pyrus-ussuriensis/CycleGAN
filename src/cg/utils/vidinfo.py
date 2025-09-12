# tools/vidinfo.py
import cv2, sys, json
#p = sys.argv[1]
p = "data/videos/building.mp4"
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
