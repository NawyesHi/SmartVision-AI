from deepface import DeepFace
from deepface.modules.verification import find_distance # pour trouver la distance entre les embeddings
import cv2 # pour les operations d'image
import time # pour le temps
import sys # pour les operations de systeme
import pickle # pour la serialisation des donnees
import cvzone # pour les textes sur les image
from ultralytics import YOLO # pour la detection d'objets
import supervision as sv # pour la supervision des objets
import numpy as np # pour les operations numeriques

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

cap = cv2.VideoCapture(0)  # 0 correspond a la camera par defaut
if not cap.isOpened(): # si la camera n'est pas ouverte 
    print("Erreur : Impossible d'accéder à la caméra.") # afficher un message d'erreur
    sys.exit(0) # quitter le programme

start_time = time.time()
fps = 0 # pour le fps
frame_count = 0 # pour le nombre de frames
detected_faces = [] # pour les visages detectes
last_detection_time = 0 # pour le temps de la derniere detection

model_name = "Facenet512" # pour le modele de representation des visages
metrics = [ # pour les differentes distances entre les embeddings
    {"cosine": 0.35}, # pour la distance cosine
    {"euclidean": 25.0}, # pour la distance euclidienne
    {"euclidean_l2": 0.85} # pour la distance euclidienne L2
]
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # pour le codec de la video
ret, frame = cap.read() # pour lire la frame
frame_width = frame.shape[1] # pour la largeur de la frame
frame_height = frame.shape[0] # pour la hauteur de la frame
out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height)) # pour enregistrer la video

# Chargement des embeddings existants
try: # si le fichier d'embeddings existe    
    with open(f"./embeddings/embs_facenet512.pkl", "rb") as file: # ouvrir le fichier d'embeddings
        embs = pickle.load(file) # charger les embeddings
        print("Fichier d'embeddings existant chargé avec succès.") # afficher un message de succes
except FileNotFoundError: # si le fichier d'embeddings n'existe pas
    print("Aucun fichier d'embeddings trouvé. Vérifiez le chemin.") # afficher un message d'erreur
    sys.exit(0) # quitter le programme

def calculate_fps(start_time): # pour calculer le fps
    current_time = time.time() # pour le temps actuel
    fps = 1.0 / (current_time - start_time) # pour le fps
    start_time = current_time # pour le temps de la derniere detection
    return fps, start_time

def clahe(image): # pour l'egalisation de l'histogramme
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # pour l'egalisation de l'histogramme
    return clahe.apply(image) # pour l'egalisation de l'histogramme

# Configuration personnalisée de l'annotation
class CustomBoxAnnotator: # pour l'annotation des boites englobantes
    def __init__(self, thickness=2, color=(255, 0, 0)): # pour l'annotation des boites englobantes
        self.thickness = thickness # pour l'annotation des boites englobantes
        self.color = color # pour l'annotation des boites englobantes

    def annotate(self, scene, detections): 
        # Récupérer les coordonnées des boîtes englobantes, les classes et les scores
        for bbox, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            # Extraire les coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, bbox)
            
            # Dessiner le rectangle
            cv2.rectangle(scene, (x1, y1), (x2, y2), self.color, self.thickness)
            
            # Ajouter le texte avec le nom de la classe et la confiance
            if class_id is not None: # si la classe n'est pas None  
                label = f"{COCO_CLASSES[class_id]} {confidence:.2f}" # pour le label
                cv2.putText(scene, label, (x1, y1 - 10), # pour le texte
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, 2) # pour le font, la taille, la couleur et l'epaisseur du texte
        return scene

# Créer l'annotateur personnalisé
box_annotator = CustomBoxAnnotator() # pour l'annotation des boites englobantes

while True: # pour la boucle infinie
    new_frame_time = time.time() # pour le temps de la nouvelle frame   
    ret, frame = cap.read() # pour lire la frame

    if not ret: # si la frame n'est pas lue correctement
        print("Error: Frame not read successfully") # afficher un message d'erreur
        break # quitter la boucle

    # Détection d'objets avec YOLOv8
    yolo_results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(yolo_results) 
    
    # Filtrer les détections avec un seuil de confiance
    detections = detections[detections.confidence > 0.5] # pour filtrer les detections avec un seuil de confiance
    print(f"YOLO Detected {len(detections)} objects.") # pour afficher le nombre de detections
    
    # Annotation des objets détectés
    frame = box_annotator.annotate(frame, detections)

    fps, start_time = calculate_fps(start_time) # pour calculer le fps

    if frame_count % 5 == 0: # pour chaque 5eme frame
        detected_faces = [] # pour les visages detectes 
        results = DeepFace.extract_faces(frame, detector_backend="yolov8", enforce_detection=False) # pour extraire les visages
        print(f"DeepFace found {len(results)} faces.") # pour afficher le nombre de visages detectes
            
        for result in results: # pour chaque visage
            if result["confidence"] >= 0.5: # si la confiance est superieure a 0.5
                x = result["facial_area"]["x"] # pour la position x
                y = result["facial_area"]["y"] # pour la position y
                w = result["facial_area"]["w"] # pour la largeur
                h = result["facial_area"]["h"] # pour la hauteur

                x1, y1 = x, y # pour la position x1 et y1
                x2, y2 = x + w, y + h # pour la position x2 et y2

                cropped_face = frame[y:y+h, x:x+w] # pour le visage recadre
                cropped_face_resized = cv2.resize(cropped_face, (224, 224)) # pour redimensionner le visage
                cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY) # pour convertir le visage en grayscale
                cropped_face_norm = clahe(cropped_face_gray) # pour normaliser le visage
                cropped_face_gray = cv2.cvtColor(cropped_face_norm, cv2.COLOR_GRAY2RGB) # pour convertir le visage en RGB           

                emb = DeepFace.represent( # pour representer le visage          
                    cropped_face_gray, # pour le visage recadre     
                    model_name=model_name, # pour le modele de representation des visages
                    enforce_detection=False, # pour ne pas detecter les visages
                    detector_backend="skip", # pour ne pas utiliser le detecteur de visages
                )[0]["embedding"] # pour l'embedding

                min_dist = float("inf") # pour la distance minimale
                match_name = None # pour le nom du match

                for name, emb2 in embs.items():
                    if isinstance(emb2, list):
                        # Calculate distances for all embeddings of this person
                        distances = [find_distance(emb, e, list(metrics[2].keys())[0]) for e in emb2]
                        dst = min(distances)  # Use the best match from multiple embeddings
                    else: # si la distance est unique
                        dst = find_distance(emb, emb2, list(metrics[2].keys())[0]) # pour la distance
                    
                    if dst < min_dist: # si la distance est inferieure a la distance minimale
                        min_dist = dst # pour la distance minimale
                        match_name = name # pour le nom du match

                if min_dist < list(metrics[2].values())[0]: # si la distance minimale est inferieure a la valeur de la distance 
                    detected_faces.append( # pour ajouter le visage detecte
                        (x1, y1, x2, y2, match_name, min_dist, (0, 255, 0)) # pour la position, le nom, la distance et la couleur
                    )
                    print(f"Face detected as: {match_name} with distance: {min_dist:.2f}")
                else:  
                    detected_faces.append(
                        (x1, y1, x2, y2, "Unknown", min_dist, (0, 0, 255))
                    )

        last_detection_time = frame_count  # pour le temps de la derniere detection

    # Dessiner les visages reconnus
    for x1, y1, x2, y2, name, min_dist, color in detected_faces: # pour chaque visage detecte
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1) # pour dessiner le rectangle
        cvzone.putTextRect( # pour ajouter le texte
            frame, # pour la frame
            f"{name} {min_dist:.2f}", # pour le texte
            (x1 + 10, y1 - 12), # pour la position du texte
            scale=1.5, # pour la taille du texte
            thickness=2, # pour l'epaisseur du texte
            colorR=color, # pour la couleur du texte
        )

    cv2.imshow("frame", frame) # pour afficher la frame
    out.write(frame) # pour enregistrer la frame    

    frame_count += 1 # pour le nombre de frames

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() # pour liberer la camera  
out.release() # pour liberer la video
cv2.destroyAllWindows() # pour fermer toutes les fenetres
