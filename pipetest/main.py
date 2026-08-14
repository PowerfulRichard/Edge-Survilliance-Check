import cv2
import ncnn
import numpy as np
import math

# SCRFD
scrfd = ncnn.Net()
scrfd.opt.use_vulkan_compute = False
scrfd.load_param("scrfd_500m-opt2.param")
scrfd.load_model("scrfd_500m-opt2.bin")

# MobileFaceNet
facenet = ncnn.Net()
facenet.opt.use_vulkan_compute = False
facenet.load_param("model/mobilefacenets.param")
facenet.load_model("model/mobilefacenets.bin")

face_db = {
    "Alice": np.load("alice.npy"),   # (128,)
    "Bob": np.load("bob.npy"),
    "Cindy": np.load("cindy.npy"),
}

def cosine(a, b):
    return np.dot(a, b)

def detect_faces(img):
    h, w = img.shape[:2]

    ex = scrfd.create_extractor()
    ex.set_light_mode(True)

    mat = ncnn.Mat.from_pixels(
        img.tobytes(),
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        w, h
    )

    ex.input("input", mat)

    # ⚠️ scrfd_500m-opt2 常见输出
    _, scores = ex.extract("score_8")
    _, bboxes = ex.extract("bbox_8")

    faces = []

    for i in range(scores.w):
        score = scores[i]
        if score < 0.6:
            continue

        x1 = int(bboxes[i * 4 + 0] * w)
        y1 = int(bboxes[i * 4 + 1] * h)
        x2 = int(bboxes[i * 4 + 2] * w)
        y2 = int(bboxes[i * 4 + 3] * h)

        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        faces.append((x1, y1, x2, y2, score))

    return faces

def get_embedding(face_img):
    h, w = face_img.shape[:2]

    ex = facenet.create_extractor()

    mat = ncnn.Mat.from_pixels(
        face_img.tobytes(),
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        w, h
    )

    ex.input("input", mat)

    _, feat = ex.extract("output")

    vec = np.array(feat, dtype=np.float32)

    # L2 normalize（非常重要）
    vec = vec / (np.linalg.norm(vec) + 1e-6)

    return vec

def recognize(feat):
    best_name = "unknown"
    best_score = -1

    for name, db_feat in face_db.items():
        sim = cosine(feat, db_feat)

        if sim > best_score:
            best_score = sim
            best_name = name

    # 阈值（MobileFaceNet常用 0.35~0.5）
    if best_score < 0.4:
        return "unknown", best_score

    return best_name, best_score

def run_camera():
    cap = cv2.VideoCapture(0)

    # 提升摄像头性能
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for (x1, y1, x2, y2, score) in faces:
            face = frame[y1:y2, x1:x2]

            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            face = cv2.resize(face, (112, 112))

            feat = get_embedding(face)

            name, sim = recognize(feat)

            label = f"{name} {sim:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

        cv2.imshow("SCRFD + MobileFaceNet", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()




def run_image():
    img = cv2.imread("face.png")

    if img is None:
        print("❌ 无法读取 face.png")
        return

    faces = detect_faces(img)

    print(f"检测到人脸数量: {len(faces)}")

    for (x1, y1, x2, y2, score) in faces:
        face = img[y1:y2, x1:x2]

        if face.shape[0] < 20 or face.shape[1] < 20:
            continue

        # resize 给 MobileFaceNet
        face = cv2.resize(face, (112, 112))

        feat = get_embedding(face)

        name, sim = recognize(feat)

        print(f"识别结果: {name}, 相似度: {sim:.3f}")

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            img,
            f"{name} {sim:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imwrite("result.png", img)
    print("✅ 结果已保存到 result.png")


if __name__ == "__main__":
    run_image()
    # run_camera()