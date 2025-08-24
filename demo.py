import cv2
import torch
import numpy as np
import matplotlib
import mediapipe as mp
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from PIL import Image, ImageTk
from collections import deque
from model.emonex.model import EmoNeXt

# Mediapipe face detection
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

plt.ion()

history = deque(maxlen=100)  # lưu 100 dự đoán gần nhất

fig, axs = plt.subplots(
    2, 2,           # 2 dòng, 2 cột
    figsize=(7, 5),
    gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [4, 3]}  # tỉ lệ chiều rộng
)

ax1 = axs[0, 0]  # dòng 1 - cột 1
ax2 = axs[0, 1]  # dòng 1 - cột 2
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)  # dòng 2 trải ngang
plt.subplots_adjust(hspace=0.8)

# mngr = plt.get_current_fig_manager()
# mngr.window.wm_geometry("+1+1")
fig.canvas.manager.set_window_title("Biểu đồ cảm xúc realtime")
root = fig.canvas.manager.window
icon = ImageTk.PhotoImage(Image.open("demo_icon_fixed.png"))
root.iconphoto(False, icon)
fig.text(0.35, 0.95, "Biểu đồ phân bố các cảm xúc", ha="center", va="bottom", fontsize=12)
fig.text(0.75, 0.95, "Biểu đồ theo nhóm cảm xúc", ha="center", va="bottom", fontsize=12)
fig.text(0.5, 0.4, "Biểu đồ diễn biến cảm xúc", ha="center", va="bottom", fontsize=12)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()

# Webcam
cap = cv2.VideoCapture(0)

def normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0        # scale về [0,1]
    img = (img - mean) / std
    return img

GROUPS = {
    "Positive": ["Happy", "Surprise"],
    "Negative": ["Angry", "Disgust", "Fear", "Sad"],
    "Neutral": ["Neutral"]
}
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ORDERED_CLASSES = [
    "Angry", "Disgust", "Fear", "Sad",   # Negative
    "Neutral",                           # Neutral
    "Happy", "Surprise"                  # Positive
]
CLASS_TO_ORDER = {cls: i for i, cls in enumerate(ORDERED_CLASSES)}

COLORS_BGR = [
    (0, 128, 255),    # Angry - cam rực
    (0, 153, 0),      # Disgust - xanh lá đậm
    (255, 128, 0),    # Fear - xanh dương đậm
    (255, 178, 0),    # Happy - vàng đậm
    (153, 51, 255),   # Sad - tím lam đậm
    (178, 51, 178),   # Surprise - tím hồng đậm
    (128, 128, 128)   # Neutral - xám đậm
]
COLORS_PIE_GROUPS = [
    (0, 153, 0),      # Positive - xanh lá đậm
    (0, 0, 200),      # Negative - đỏ đậm dịu
    (128, 128, 128)   # Neutral  - xám đậm
]
COLORS_RGB = [(r/255, g/255, b/255) for (b,g,r) in COLORS_BGR]
COLORS_PIE_GROUPS = [(r/255, g/255, b/255) for (b,g,r) in COLORS_PIE_GROUPS]
FRAME_DET_ID = 0

num_classes = 7
model = EmoNeXt(num_classes=num_classes)
checkpoint = torch.load("checkpoint\checkpoint_EmoNeXt.pt", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
        for det in results.detections:
            box = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1, y1 = int(box.xmin * w), int(box.ymin * h)
            x2, y2 = int((box.xmin + box.width) * w), int((box.ymin + box.height) * h)

            # Crop face
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            img = cv2.resize(face, (48, 48))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # shape (48,48)
            img = cv2.merge([gray, gray, gray])              # (48,48,3)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = normalize(img, mean, std).astype(np.float32)
            img_tensor = torch.from_numpy(img.transpose(2,0,1))
            img_tensor = img_tensor.unsqueeze(0)

            # Dự đoán
            with torch.no_grad():
                outputs = model(img_tensor)
                predicted = outputs[0].item()

            if FRAME_DET_ID % 5 == 0:
                history.append(predicted)
                plt.savefig("Emotion report")

            # Vẽ kết quả
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLORS_BGR[predicted], 2)
            cv2.putText(frame, CLASSES[predicted], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS_BGR[predicted], 2)
            cv2.rectangle(frame, (0,0), (w-1, h-1), COLORS_BGR[predicted], 2)

            FRAME_DET_ID = FRAME_DET_ID + 1

    ax1.clear()
    ax2.clear()
    ax3.clear()

    counts = [history.count(i) for i in range(len(CLASSES))]
    ax1.bar(CLASSES, counts, color=COLORS_RGB)
    ax1.tick_params(axis='x', rotation=45)

    group_counts = {g: 0 for g in GROUPS}
    for label in history:
        for g, lst in GROUPS.items():
            if CLASSES[label] in lst:
                group_counts[g] += 1
    labels = list(group_counts.keys())
    sizes = list(group_counts.values())
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=COLORS_PIE_GROUPS, startangle=90)

    ordered_history = [CLASS_TO_ORDER[CLASSES[label]] for label in history]
    ax3.plot(range(len(history)), ordered_history, color="blue", linewidth=1)
    ax3.set_yticks(range(len(ORDERED_CLASSES)))
    ax3.set_yticklabels(ORDERED_CLASSES)

    plt.pause(0.001)

    cv2.imshow("Webcam", frame)

    # Lấy kích thước màn hìnhq
    screen_w = cv2.getWindowImageRect("Webcam")[2]
    screen_h = cv2.getWindowImageRect("Webcam")[3]

    x = screen_w + 60
    y = 50
    cv2.moveWindow("Webcam", x, y)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
