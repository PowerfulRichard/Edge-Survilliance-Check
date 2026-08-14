import cv2
from scrfd import SCRFD
from mobilefacenet import MobileFaceNet
from utils import cosine
from db import load_db, save_db
import config

# load models
detector = SCRFD(
    "/root/model/scrfd_500m_kps-opt2.param",
    "/root/model/scrfd_500m_kps-opt2.bin"
)

recognizer = MobileFaceNet(
    "/root/model/mobilefacenets.param",
    "/root/model/mobilefacenets.bin"
)

#db = load_db()

cap = cv2.VideoCapture(config.RTSP_URL)

frame_id = 0

print("System started...")


ret, frame = cap.read()

if ret:
    cv2.imwrite("snapshot.jpg", frame)
    print("截图已保存：snapshot.jpg")
else:
    print("读取视频帧失败")

cap.release()
"""
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_id += 1
    if frame_id % config.FRAME_SKIP != 0:
        continue

    faces = detector.detect(frame)

    for box in faces:
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        feat = recognizer.get_embedding(face)

        name = "unknown"
        best_score = 0

        for k, v in db.items():
            score = cosine(feat, v)
            if score > best_score:
                best_score = score
                name = k

        if best_score > config.THRESHOLD:
            print(f"[KNOWN] {name} score={best_score:.2f}")
        else:
            print("[ALERT] Unknown person!")

    cv2.imshow("face", frame)

    if cv2.waitKey(1) == 27:
        break
"""