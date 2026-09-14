import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# Remove duplicate detections
# -----------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


def nms(boxes, scores, threshold=0.4):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        remaining = []

        for j in order[1:]:
            if iou(boxes[i], boxes[j]) < threshold:
                remaining.append(j)

        order = np.array(remaining)

    return keep


# -----------------------------
# Tiled prediction
# -----------------------------
def predict_whole_scan(image, confidence):
    image = image.convert("RGB")
    img = np.array(image)

    height, width = img.shape[:2]

    tile_height = 512
    overlap = 128
    stride = tile_height - overlap

    starts = list(range(0, max(height - tile_height + 1, 1), stride))

    last_start = max(0, height - tile_height)

    if last_start not in starts:
        starts.append(last_start)

    all_boxes = []
    all_scores = []

    for start_y in starts:

        end_y = min(start_y + tile_height, height)

        tile = img[start_y:end_y, :]

        results = model.predict(
            tile,
            imgsz=640,
            conf=confidence,
            verbose=False
        )[0]

        if results.boxes is None:
            continue

        for box in results.boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = xyxy

            # Map tile coordinates back to whole scan
            y1 += start_y
            y2 += start_y

            score = float(box.conf[0])

            all_boxes.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ])

            all_scores.append(score)

    # Merge duplicate detections caused by overlapping tiles
    keep = nms(all_boxes, all_scores, threshold=0.4)

    final_boxes = [all_boxes[i] for i in keep]
    final_scores = [all_scores[i] for i in keep]

    # Draw detections
    output = image.copy()
    draw = ImageDraw.Draw(output)

    for box, score in zip(final_boxes, final_scores):

        x1, y1, x2, y2 = box

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"{score:.2f}",
            fill="red"
        )

    return output, len(final_boxes)


# -----------------------------
# Streamlit webpage
# -----------------------------
st.set_page_config(
    page_title="Bone Scan AI Demo",
    layout="wide"
)

st.title("Bone Scan AI Demo")

st.write(
    "Upload a whole-body bone scintigraphy image. "
    "The AI will highlight regions of suspected abnormal uptake."
)

st.warning(
    "Research prototype only — not for clinical diagnosis or patient management."
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.05,
    max_value=0.80,
    value=0.10,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload bone scan image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Original image")
    st.image(image, use_container_width=True)

    if st.button("Run AI Prediction"):

        with st.spinner("Analysing bone scan..."):

            output, count = predict_whole_scan(
                image,
                confidence
            )

        st.subheader("AI prediction")

        st.image(
            output,
            use_container_width=True
        )

        st.success(
            f"Suspicious regions detected: {count}"
        )
