from deepface import DeepFace
from deepface.modules.verification import find_distance
import cv2
import time
import sys
import pickle
import cvzone
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Charger le modèle YOLOv8 pré-entraîné
yolo_model = YOLO("yolov8n.pt")

# Définir les classes COCO (du modèle pré-entraîné)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

cap = cv2.VideoCapture(0)  # 0 correspond à la caméra par défaut
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra.")
    sys.exit(0)

start_time = time.time()
fps = 0
frame_count = 0
detected_faces = []
last_detection_time = 0

model_name = "Facenet512"
metrics = [
    {"cosine": 0.35},
    {"euclidean": 25.0},
    {"euclidean_l2": 0.85}
]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
ret, frame = cap.read()
frame_width = frame.shape[1]
frame_height = frame.shape[0]
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

# Chargement des embeddings existants
try:
    with open(f"./embeddings/embs_facenet512.pkl", "rb") as file:
        embs = pickle.load(file)
        print("Fichier d'embeddings existant chargé avec succès.")
except FileNotFoundError:
    print("Aucun fichier d'embeddings trouvé. Vérifiez le chemin.")
    sys.exit(0)

def calculate_fps(start_time):
    current_time = time.time()
    fps = 1.0 / (current_time - start_time)
    start_time = current_time
    return fps, start_time

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Configuration personnalisée de l'annotation
class CustomBoxAnnotator:
    def __init__(self, thickness=2, color=(255, 0, 0)):
        self.thickness = thickness
        self.color = color

    def annotate(self, scene, detections):
        # Récupérer les coordonnées des boîtes englobantes, les classes et les scores
        for bbox, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            # Extraire les coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, bbox)
            
            # Dessiner le rectangle
            cv2.rectangle(scene, (x1, y1), (x2, y2), self.color, self.thickness)
            
            # Ajouter le texte avec le nom de la classe et la confiance
            if class_id is not None:
                label = f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                cv2.putText(scene, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2)
        return scene

# Créer l'annotateur personnalisé
box_annotator = CustomBoxAnnotator()

while True:
    new_frame_time = time.time()
    ret, frame = cap.read()

    if not ret:
        print("Error: Frame not read successfully")
        break

    # Détection d'objets avec YOLOv8
    yolo_results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(yolo_results)
    
    # Filtrer les détections avec un seuil de confiance
    detections = detections[detections.confidence > 0.5]
    print(f"YOLO Detected {len(detections)} objects.")
    
    # Annotation des objets détectés
    frame = box_annotator.annotate(frame, detections)

    fps, start_time = calculate_fps(start_time)

    if frame_count % 5 == 0:
        detected_faces = []
        results = DeepFace.extract_faces(frame, detector_backend="yolov8", enforce_detection=False)
        print(f"DeepFace found {len(results)} faces.")

        for result in results:
            if result["confidence"] >= 0.5:
                x = result["facial_area"]["x"]
                y = result["facial_area"]["y"]
                w = result["facial_area"]["w"]
                h = result["facial_area"]["h"]

                x1, y1 = x, y
                x2, y2 = x + w, y + h

                cropped_face = frame[y:y+h, x:x+w]
                cropped_face_resized = cv2.resize(cropped_face, (224, 224))
                cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
                cropped_face_norm = clahe(cropped_face_gray)
                cropped_face_gray = cv2.cvtColor(cropped_face_norm, cv2.COLOR_GRAY2RGB)

                emb = DeepFace.represent(
                    cropped_face_gray,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]

                min_dist = float("inf")
                match_name = None

                for name, emb2 in embs.items():
                    if isinstance(emb2, list):
                        # Calculate distances for all embeddings of this person
                        distances = [find_distance(emb, e, list(metrics[2].keys())[0]) for e in emb2]
                        dst = min(distances)  # Use the best match from multiple embeddings
                    else:
                        dst = find_distance(emb, emb2, list(metrics[2].keys())[0])
                    
                    if dst < min_dist:
                        min_dist = dst
                        match_name = name

                if min_dist < list(metrics[2].values())[0]:
                    detected_faces.append(
                        (x1, y1, x2, y2, match_name, min_dist, (0, 255, 0))
                    )
                    print(f"Face detected as: {match_name} with distance: {min_dist:.2f}")
                else:
                    detected_faces.append(
                        (x1, y1, x2, y2, "Unknown", min_dist, (0, 0, 255))
                    )

        last_detection_time = frame_count

    # Dessiner les visages reconnus
    for x1, y1, x2, y2, name, min_dist, color in detected_faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cvzone.putTextRect(
            frame,
            f"{name} {min_dist:.2f}",
            (x1 + 10, y1 - 12),
            scale=1.5,
            thickness=2,
            colorR=color,
        )

    cv2.imshow("frame", frame)
    out.write(frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
