import os
import cv2
import pickle
import numpy as np
from collections import Counter, deque
import fix_tf_keras  # ensures tensorflow.keras compatibility
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, OneClassSVM
from sklearn.decomposition import PCA

# ===============================
# CONFIGURATION
# ===============================
DATASET_DIR = "dataset"
ENCODINGS_FILE = "face_encodings_df.pkl"
MODEL_FILE = "face_recognition_model_df.pkl"
PCA_FILE = "pca_transform_df.pkl"

NORMALIZE = True
PCA_COMPONENTS = 100

DETECTOR_BACKEND = "retinaface"
EMBEDDING_MODEL = "Facenet512"

# Thresholds
FACE_DETECTION_CONFIDENCE = 0.65  # RetinaFace sometimes underreports confidence
RECOGNITION_THRESHOLD = 45.0      # minimum recognition probability %
STABLE_LABEL_FRAMES = 5           # show label after consistent frames

# ===============================
# HELPER: extract face embedding
# ===============================
def extract_face_embedding_from_image(image_bgr):
    try:
        faces = DeepFace.extract_faces(
            img_path=image_bgr,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True
        )
        if not faces:
            return None

        face_img = faces[0]["face"]
        rep = DeepFace.represent(face_img, model_name=EMBEDDING_MODEL, enforce_detection=False)
        if isinstance(rep, list) and len(rep) > 0 and "embedding" in rep[0]:
            emb = np.array(rep[0]["embedding"], dtype=np.float32)
        elif isinstance(rep, dict) and "embedding" in rep:
            emb = np.array(rep["embedding"], dtype=np.float32)
        else:
            emb = np.array(rep, dtype=np.float32).flatten()

        if NORMALIZE:
            emb -= np.mean(emb)
            emb /= np.linalg.norm(emb) + 1e-8

        return emb
    except Exception as e:
        print(f"⚠️ DeepFace failed for an image: {e}")
        return None


# ===============================
# STEP 1: Build encodings
# ===============================
if not os.path.exists(ENCODINGS_FILE):
    print("[INFO] Generating face embeddings with DeepFace (this may take a while)...")
    known_encodings, known_names = [], []

    for person in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        image_files = sorted([f for f in os.listdir(person_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for image_name in image_files:
            image_path = os.path.join(person_dir, image_name)
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Skipping unreadable file: {image_path}")
                continue

            emb = extract_face_embedding_from_image(img)
            if emb is not None:
                known_encodings.append(emb)
                known_names.append(person)
                print(f"✅ Processed {image_name} for {person}")
            else:
                print(f"⚠️ No face / failed embedding for {image_name} ({person})")

    if len(known_encodings) == 0:
        print("❌ No valid faces found in the dataset.")
        raise SystemExit(1)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(known_names)
    print("🧩 Samples per label:", Counter(known_names))

    print("[INFO] Applying PCA reduction...")
    n_samples = len(known_encodings)
    n_features = len(known_encodings[0])
    n_components = min(PCA_COMPONENTS, max(1, n_samples - 1), n_features)

    if n_samples < 2:
        pca = None
        reduced_encodings = np.array(known_encodings)
    else:
        pca = PCA(n_components=n_components)
        reduced_encodings = pca.fit_transform(np.vstack(known_encodings))

    data = {"encodings": reduced_encodings, "labels": labels, "label_encoder": label_encoder}
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    if pca is not None:
        with open(PCA_FILE, "wb") as f:
            pickle.dump(pca, f)
    print("✅ Saved embeddings and PCA.")
else:
    print("[INFO] Loading existing embeddings and PCA...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    pca = None
    if os.path.exists(PCA_FILE):
        with open(PCA_FILE, "rb") as f:
            pca = pickle.load(f)

# ===============================
# STEP 2: Train / Load Model
# ===============================
unique_labels = np.unique(data["labels"])
one_class_mode = len(unique_labels) < 2

if not os.path.exists(MODEL_FILE):
    if one_class_mode:
        print("[INFO] Only one person found — using One-Class SVM.")
        X = np.array(data["encodings"])
        clf = OneClassSVM(kernel="rbf", gamma="auto").fit(X)
        model_info = {"model": clf, "label_encoder": data["label_encoder"], "one_class": True}
    else:
        print("[INFO] Training SVM classifier...")
        X, y = np.array(data["encodings"]), np.array(data["labels"])
        clf = SVC(C=10, kernel="rbf", gamma="scale", probability=True)
        clf.fit(X, y)
        model_info = {"model": clf, "label_encoder": data["label_encoder"], "one_class": False}

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_info, f)
    print("✅ Model trained and saved.")
else:
    print("[INFO] Loading trained model...")
    with open(MODEL_FILE, "rb") as f:
        model_data = pickle.load(f)
    clf = model_data["model"]
    data["label_encoder"] = model_data["label_encoder"]
    one_class_mode = model_data.get("one_class", False)

# ===============================
# STEP 3: Real-time Recognition
# ===============================
print("[INFO] Starting real-time recognition (Press 'q' to quit)...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    raise SystemExit(1)

DETECTION_SCALE = 0.6  # downscale for faster detection

recent_predictions = deque(maxlen=STABLE_LABEL_FRAMES)
stable_label = "No face detected"

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)

    try:
        faces = DeepFace.extract_faces(
            img_path=small_frame,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print(f"⚠️ DeepFace.extract_faces error: {e}")
        faces = []

    display_label = "No face detected"
    display_color = (0, 0, 255)

    if faces and faces[0].get("confidence", 0) > FACE_DETECTION_CONFIDENCE:
        face_info = faces[0]
        face_img = face_info.get("face")
        region = face_info.get("region", None)

        emb = None
        try:
            rep = DeepFace.represent(face_img, model_name=EMBEDDING_MODEL, enforce_detection=False)
            if isinstance(rep, list) and rep and "embedding" in rep[0]:
                emb = np.array(rep[0]["embedding"], dtype=np.float32)
            elif isinstance(rep, dict) and "embedding" in rep:
                emb = np.array(rep["embedding"], dtype=np.float32)
            else:
                emb = np.array(rep, dtype=np.float32).flatten()
            if NORMALIZE:
                emb -= np.mean(emb)
                emb /= np.linalg.norm(emb) + 1e-8
        except Exception as e:
            print(f"⚠️ Failed to compute embedding: {e}")

        if emb is not None:
            embedding_pca = pca.transform([emb]) if pca else [emb]

            if one_class_mode:
                pred = clf.predict(embedding_pca)[0]
                name = data["label_encoder"].classes_[0] if pred == 1 else "Unknown"
                confidence = 100.0 if pred == 1 else 0.0
            else:
                probs = clf.predict_proba(embedding_pca)[0]
                max_idx = np.argmax(probs)
                confidence = probs[max_idx] * 100
                name = data["label_encoder"].inverse_transform([max_idx])[0]
                if confidence < RECOGNITION_THRESHOLD:
                    name = "Unknown"

            recent_predictions.append(name)
            if len(recent_predictions) == STABLE_LABEL_FRAMES and len(set(recent_predictions)) == 1:
                stable_label = name

            display_label = f"{stable_label} ({confidence:.1f}%)"
            display_color = (0, 255, 0) if stable_label != "Unknown" else (0, 0, 255)

            if region:
                x = int(region["x"] / DETECTION_SCALE)
                y = int(region["y"] / DETECTION_SCALE)
                w_face = int(region["w"] / DETECTION_SCALE)
                h_face = int(region["h"] / DETECTION_SCALE)
                x2, y2 = x + w_face, y + h_face
                cv2.rectangle(frame, (x, y), (x2, y2), display_color, 2)
                cv2.putText(frame, display_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
    else:
        recent_predictions.clear()
        stable_label = "No face detected"
        cv2.putText(frame, stable_label, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

    cv2.imshow("Face Recognition (DeepFace + SVM)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
