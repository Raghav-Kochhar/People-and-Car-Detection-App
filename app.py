import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import tempfile
from collections import Counter
import time
from typing import Tuple, List, Dict
from sklearn.cluster import KMeans

st.set_page_config(page_title="People and Car Detection App", layout="wide")

MALE_COLOR = (0, 255, 0)
FEMALE_COLOR = (255, 0, 0)
CAR_COLOR = (0, 0, 255)

car_colors: Dict[str, Tuple[List[int], List[int]]] = {
    "White": ([0, 0, 200], [180, 30, 255]),
    "Black": ([0, 0, 0], [180, 255, 30]),
    "Gray": ([0, 0, 40], [180, 30, 220]),
    "Silver": ([0, 0, 180], [180, 30, 240]),
    "Red": ([0, 50, 50], [10, 255, 255]),
    "Blue": ([100, 50, 50], [130, 255, 255]),
    "Green": ([40, 50, 50], [80, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Orange": ([10, 100, 100], [20, 255, 255]),
    "Brown": ([0, 40, 40], [20, 255, 160]),
    "Purple": ([130, 50, 50], [160, 255, 255]),
    "Pink": ([140, 50, 50], [170, 255, 255]),
}


@st.cache_resource
def load_models() -> Tuple[YOLO, tf.keras.Model]:
    try:
        yolo_model = YOLO("yolov8x.pt")

        gender_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(96, 96, 3)
        )
        gender_model = tf.keras.Sequential(
            [
                gender_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        return yolo_model, gender_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()


yolo_model, gender_model = load_models()


car_colors = {
    "White": ([0, 0, 200], [180, 30, 255]),
    "Black": ([0, 0, 0], [180, 255, 30]),
    "Gray": ([0, 0, 40], [180, 30, 220]),
    "Silver": ([0, 0, 180], [180, 30, 240]),
    "Red": ([0, 50, 50], [10, 255, 255]),
    "Blue": ([100, 50, 50], [130, 255, 255]),
    "Green": ([40, 50, 50], [80, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Orange": ([10, 100, 100], [20, 255, 255]),
    "Brown": ([0, 40, 40], [20, 255, 160]),
    "Purple": ([130, 50, 50], [160, 255, 255]),
    "Pink": ([140, 50, 50], [170, 255, 255]),
}


@st.cache_data
def get_car_color(image_rgb: np.ndarray) -> str:
    resized_image = cv2.resize(image_rgb, (100, 100), interpolation=cv2.INTER_AREA)
    reshaped_image = resized_image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=3, random_state=0).fit(reshaped_image)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[
        0
    ][0]

    color_name = "Unknown"
    min_distance = float("inf")
    for name, (lower, upper) in car_colors.items():
        lower = np.array(lower)
        upper = np.array(upper)
        if (lower <= dominant_color_hsv).all() and (dominant_color_hsv <= upper).all():
            color_name = name
            break
        else:
            center_hsv = (np.array(lower) + np.array(upper)) / 2
            distance = np.linalg.norm(center_hsv - dominant_color_hsv)
            if distance < min_distance:
                min_distance = distance
                color_name = name

    return color_name


@tf.function
def predict_gender(face_rgb: tf.Tensor) -> tf.Tensor:
    face = tf.image.resize(face_rgb, (96, 96))
    face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
    face = tf.expand_dims(face, axis=0)
    prediction = gender_model(face, training=False)
    return tf.where(prediction[0][0] > 0.5, "Male", "Female")


def draw_boxes_and_labels(
    image: np.ndarray,
    boxes: List[np.ndarray],
    labels: List[str],
    colors: List[Tuple[int, int, int]],
) -> np.ndarray:
    for box, label, color in zip(boxes, labels, colors):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
    return image


def process_frame(
    frame_rgb: np.ndarray,
) -> Tuple[
    List, List[str], List[str], List[np.ndarray], List[str], List[Tuple[int, int, int]]
]:
    results = yolo_model(frame_rgb, classes=[0, 2])
    genders, car_colors, boxes, labels, colors = [], [], [], [], []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append(bbox)

            if cls == 0:
                face = frame_rgb[
                    max(0, bbox[1]) : min(frame_rgb.shape[0], bbox[3]),
                    max(0, bbox[0]) : min(frame_rgb.shape[1], bbox[2]),
                ]
                if face.size > 0:
                    gender = predict_gender(face).numpy().decode("utf-8")
                    genders.append(gender)
                    labels.append(gender)
                    colors.append(MALE_COLOR if gender == "Male" else FEMALE_COLOR)
            elif cls == 2:
                car_roi = frame_rgb[
                    max(0, bbox[1]) : min(frame_rgb.shape[0], bbox[3]),
                    max(0, bbox[0]) : min(frame_rgb.shape[1], bbox[2]),
                ]
                if car_roi.size > 0:
                    color_name = get_car_color(car_roi)
                    car_colors.append(color_name)
                    labels.append(color_name)
                    colors.append(CAR_COLOR)

    return results, genders, car_colors, boxes, labels, colors


st.title("People and Car Detection App")

uploaded_file = st.file_uploader(
    "Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    if uploaded_file.type.startswith("image"):
        try:
            image_bgr = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            results, genders, car_colors, boxes, labels, colors = process_frame(
                image_rgb
            )
            processing_time = time.time() - start_time

            processed_image = draw_boxes_and_labels(
                image_rgb.copy(), boxes, labels, colors
            )

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Original Image", use_column_width=True)
            with col2:
                st.image(
                    processed_image, caption="Processed Image", use_column_width=True
                )

            st.write(f"Number of males detected: {genders.count('Male')}")
            st.write(f"Number of females detected: {genders.count('Female')}")
            st.write(f"Number of cars detected: {len(car_colors)}")
            st.write("Car colors detected:", ", ".join(set(car_colors)))
            st.write(f"Processing time: {processing_time:.2f} seconds")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    elif uploaded_file.type.startswith("video"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            vf = cv2.VideoCapture(temp_file_path)

            stframe1 = st.empty()
            stframe2 = st.empty()
            gender_counts = Counter()
            car_colors = Counter()

            start_time = time.time()
            frame_count = 0

            progress_bar = st.progress(0)
            stop_button = st.button("Stop Processing")

            total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))

            while vf.isOpened() and not stop_button:
                ret, frame_bgr = vf.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results, genders, colors, boxes, labels, box_colors = process_frame(
                    frame_rgb
                )

                gender_counts.update(genders)
                car_colors.update(colors)

                processed_frame = draw_boxes_and_labels(
                    frame_rgb.copy(), boxes, labels, box_colors
                )

                col1, col2 = st.columns(2)
                with col1:
                    stframe1.image(
                        frame_rgb, caption="Original Video", use_column_width=True
                    )
                with col2:
                    stframe2.image(
                        processed_frame,
                        caption="Processed Video",
                        use_column_width=True,
                    )

                frame_count += 1
                progress_bar.progress(frame_count / total_frames)

            vf.release()
            processing_time = time.time() - start_time

            st.write(f"Number of males detected: {gender_counts['Male']}")
            st.write(f"Number of females detected: {gender_counts['Female']}")
            st.write(f"Number of cars detected: {sum(car_colors.values())}")
            st.write("Car colors detected:", ", ".join(car_colors.keys()))
            st.write(f"Total frames processed: {frame_count}")
            st.write(f"Total processing time: {processing_time:.2f} seconds")
            st.write(f"Average FPS: {frame_count / processing_time:.2f}")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

else:
    st.info("Please upload an image or video file to begin processing.")
