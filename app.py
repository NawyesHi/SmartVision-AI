from flask import Flask, render_template, Response, jsonify, make_response, request, redirect, url_for, send_from_directory #creer web application
import cv2 #importation du module cv2
import traceback #importation du module traceback
import time #importation du module time
import threading #importation du module threading
import logging #importation du module logging
from deepface import DeepFace #importation du module DeepFace
from deepface.modules.verification import find_distance #Fonction utilisée pour calculer la distance entre deux vecteurs d'embedding facial, permettant de comparer deux visages
import pickle #importation du module pickle
import numpy as np #importation du module numpy
from ultralytics import YOLO #importation du module YOLO
import supervision as sv #est une bibliothèque complémentaire qui facilite l'utilisation de YOLO, notamment pour la gestion et l'annotation des détections.
from werkzeug.utils import secure_filename #pour sécuriser les fichiers téléchargés par les utilisateurs    
import os #importation du module os
from crop_face import crop #importation de la fonction crop
from extract_embs import extract_embedding #importation de la fonction extract_embedding
import shutil #importation du module shutil
import json #importation du module json

# Configure logging (bech ye5dou les messages d'erreur haja marbouta bel systeme enfaite ) pour resoudre error
logging.basicConfig(level=logging.DEBUG) #capture les messages bch tra chnouwa isir meme les details 
logger = logging.getLogger(__name__) #pour logger le nom de l'application prech tchouf d'ou vienn
# Initialisation de l'application Flask
app = Flask(__name__) 
camera = None
camera_lock = threading.Lock() 

# Load YOLO model and embeddings
try:
    yolo_model = YOLO("yolov8n.pt")
    with open("./embeddings/embs_facenet512.pkl", "rb") as file:
        embs = pickle.load(file)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    
# Face recognition settings(facenet512 avec deepface)
model_name = "Facenet512"
#metriques de distance pour la verification de visage 
metrics = [
    {"cosine": 0.25},           # kan score de similarité inferieur a 0.25 indique eli les visages sont differents
    {"euclidean": 15.0},        # distance entre deux embeddings de visage (si d>15 les visages sont differents)
    {"euclidean_l2": 0.68}      # version normalisée de la distance euclidienne (si d>0.68 les visages sont differents)
]

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
# Configuration pour le téléchargement de fichiers
UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Fonction pour vérifier si un fichier a une extension autorisée
def allowed_file(filename): 
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fonction pour libérer la caméra (thbet kan mhloula fi plasa ou pas)
def release_camera():
    global camera #pour liberer la camera   
    with camera_lock: #pour verrouiller la camera
        if camera is not None: #si la camera est ouverte
            camera.release() #pour liberer la camera
            camera = None
            logger.info("Camera released")

# Fonction pour obtenir une instance de la caméra (tchouf esem ou thlha site )
def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0) #pour ouvrir la camera
            time.sleep(2) #pour attendre 2 secondes pour que la camera soit ouverte
        return camera

# Fonction pour traiter une frame t5ou bl frame bl frame cap yani
def process_frame(frame):
    try:
        # YOLO object detection
        yolo_results = yolo_model(frame)[0]
        detections = sv.Detections.from_ultralytics(yolo_results)
        detections = detections[detections.confidence > 0.3] # filtrer les detection avec un 0.3 est le seuil de confiance

        # Draw object detection boxes
        for bbox, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = map(int, bbox) #pour convertir les coordonnees en entiers
            # Get class name from COCO_CLASSES
            class_name = COCO_CLASSES[class_id] #pour obtenir le nom de la classe       
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #pour dessiner le rectangle
            cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1-10), #pour ajouter le texte
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) #pour ajouter le texte

        # Face detection and recognition
        results = DeepFace.extract_faces(frame, detector_backend="yolov8", 
                                       enforce_detection=False) #pour detecter les visages

        for result in results:
            if result["confidence"] >= 0.5: #si la confiance est superieur a 0.5
                x = result["facial_area"]["x"] #pour obtenir la coordonnee x
                y = result["facial_area"]["y"] #pour obtenir la coordonnee y
                w = result["facial_area"]["w"] #pour obtenir la largeur
                h = result["facial_area"]["h"] #pour obtenir la hauteur

                face_img = frame[y:y+h, x:x+w] #pour obtenir l'image du visage
                if face_img.size == 0: #si l'image du visage est vide
                    continue #pour continuer la boucle

                # Get face embedding 
                emb = DeepFace.represent(
                    face_img,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]

                # Normalize embedding   
                emb = np.array(emb) / np.linalg.norm(emb) #pour normaliser l'embedding

                # Trouver la correspondance la plus proche
                min_dist = float("inf") #pour initialiser la distance minimale
                match_name = "Unknown" #pour initialiser le nom de la correspondance
                
                # Use the same thresholds as in face_recognition.py
                cosine_threshold = 0.35 #seuil de similarité pour la verification de visage
                euclidean_l2_threshold = 0.85 #seuil de distance pour la verification de visage

                for name, stored_emb in embs.items(): #pour chaque nom et chaque embedding stocké
                    if isinstance(stored_emb, list): #si l'embedding est une liste
                        # Calculate distances for all embeddings of this person
                        distances = [find_distance(emb, se, "euclidean_l2") for se in stored_emb] #pour calculer les distances entre l'embedding et les embeddings de la personne
                        dist = min(distances) #pour obtenir la distance minimale
                    else:
                        stored_emb = np.array(stored_emb) / np.linalg.norm(stored_emb) #pour normaliser l'embedding
                        dist = find_distance(emb, stored_emb, "euclidean_l2") #pour calculer la distance entre l'embedding et l'embedding de la personne

                    if dist < min_dist: #si la distance est inferieure a la distance minimale
                        min_dist = dist #pour mettre a jour la distance minimale
                        if dist < euclidean_l2_threshold: #si la distance est inferieure au seuil de distance
                            match_name = name #pour mettre a jour le nom de la correspondance

                # Final verification
                if min_dist >= euclidean_l2_threshold: #si la distance est superieure au seuil de distance
                    match_name = "Unknown" #pour mettre a jour le nom de la correspondance

                # Draw face detection box and name
                color = (0, 255, 0) if match_name != "Unknown" else (0, 0, 255) #pour dessiner le rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2) #pour dessiner le rectangle  
                cv2.putText(frame, f"{match_name} ({min_dist:.2f})", #pour ajouter le texte 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, #pour ajouter le texte
                           0.9, color, 2) #pour ajouter le texte
    
    except Exception as e: #si une erreur se produit
        logger.error(f"Error processing frame: {e}") #pour logger l'erreur
        traceback.print_exc() #pour logger l'erreur

    return frame 

# Fonction pour générer des frames jbna process frame et care esm et las9na houma 2 frame diff
def generate_frames():
    while True: #tant que la camera est ouverte
        try: 
            cam = get_camera() #pour obtenir la camera
            if cam is not None and cam.isOpened(): #si la camera est ouverte
                ret, frame = cam.read()
                if ret:
                    # Process frame with detection and recognition
                    processed_frame = process_frame(frame) #pour traiter la frame
                    
                    # Encode and yield the frame
                    ret, buffer = cv2.imencode('.jpg', processed_frame) #pour encoder la frame en jpeg
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n') #pour envoyer la frame
                else:
                    break
            else:
                break
        except Exception as e: #si une erreur se produit
            logger.error(f"Frame generation error: {e}") #pour logger l'erreur
            break #pour sortir de la boucle

# Fonction pour vérifier la camera
@app.route('/api/check_camera', methods=['GET'])
def check_camera():
    logger.info("Checking camera...") #pour logger le message
    try:
        test_cap = cv2.VideoCapture(0) #pour ouvrir la camera   
        if not test_cap.isOpened(): #si la camera n'est pas ouverte
            logger.error("Could not open camera") #pour logger le message
            return make_response(jsonify({
                "status": "error", 
                "message": "Could not open camera"
            }), 500)
        
        ret, frame = test_cap.read() #pour lire la frame
        test_cap.release() #pour liberer la camera
        
        if not ret: #si la frame n'est pas lue
            logger.error("Could not read from camera") #pour logger le message
            return make_response(jsonify({
                "status": "error", 
                "message": "Could not read from camera"
            }), 500) 
            
        logger.info(f"Camera check successful - frame size: {frame.shape}") #pour logger le message
        return make_response(jsonify({
            "status": "success",
            "message": "Camera check successful"
        }), 200)
        
    except Exception as e: #si une erreur se produit
        logger.error(f"Camera check error: {str(e)}") #pour logger l'erreur
        traceback.print_exc() #pour logger l'erreur
        return make_response(jsonify({ #pour renvoyer une erreur
            "status": "error",
            "message": str(e)
        }), 500) #pour renvoyer une erreur

# Fonction pour démarrer la camera
@app.route('/api/start_camera', methods=['GET'])
def start_camera():
    logger.info("Starting camera...") #pour logger le message
    try:
        cam = get_camera()
        if cam is None:
            logger.error("Failed to initialize camera")
            return make_response(jsonify({
                "status": "error",
                "message": "Failed to initialize camera"
            }), 500)
            
        ret, frame = cam.read()
        if not ret:
            logger.error("Failed to read from camera") #pour logger le message
            release_camera() #pour liberer la camera
            return make_response(jsonify({ #pour renvoyer une erreur
                "status": "error",
                "message": "Failed to read from camera"
            }), 500)
            
        logger.info(f"Camera started successfully - frame size: {frame.shape}") #pour logger le message
        return make_response(jsonify({ #pour renvoyer une erreur
            "status": "success",
            "message": "Camera started successfully"
        }), 200)
        
    except Exception as e: #si une erreur se produit
        logger.error(f"Start camera error: {str(e)}") #pour logger l'erreur
        traceback.print_exc() #pour logger l'erreur
        return make_response(jsonify({ #pour renvoyer une erreur
            "status": "error",
            "message": str(e)
        }), 500) #pour renvoyer une erreur

# Fonction pour stopper la camera
@app.route('/api/stop_camera', methods=['GET'])
def stop_camera():
    logger.info("Stopping camera...") #pour logger le message
    try:
        release_camera()
        logger.info("Camera stopped successfully")
        return make_response(jsonify({
            "status": "success",
            "message": "Camera stopped successfully"
        }), 200)
    except Exception as e:
        logger.error(f"Stop camera error: {str(e)}")
        traceback.print_exc()
        return make_response(jsonify({
            "status": "error",
            "message": str(e)
        }), 500) #pour renvoyer une erreur

# Fonction pour générer des frames vidéo, utilisée pour le streaming (process+generate win thot cadrage )
@app.route('/api/video_feed') #tb3th lel frontend tatifichih fi video 
def video_feed(): 
    logger.info("Video feed requested")
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Video feed error: {str(e)}")
        traceback.print_exc()
        return make_response(jsonify({
            "status": "error",
            "message": str(e)
        }), 500)

# Route pour afficher une page d'accueil
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

# Route par défaut redirigeant vers la page d'accueil
@app.route('/')
def root():
    return redirect(url_for('welcome'))

# Route pour afficher la page principale
@app.route('/home')
def home():
    return render_template('index.html')

# Route pour afficher une page d'upload
@app.route('/upload')
def upload_page():
    return render_template('upload.html')

# API pour gérer l'upload des fichiers
@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        # Récupérer le nom d'utilisateur depuis le formulaire
        username = request.form.get('username')
        if not username:
            return jsonify({
                'status': 'error',
                'message': 'Username is required'
            }), 400

        # Créer un répertoire spécifique pour l'utilisateur
        user_dir = os.path.join(UPLOAD_FOLDER, secure_filename(username))
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Created directory for user: {username}")

        # Vérifier si des fichiers ont été uploadés
        if 'files[]' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No files uploaded'
            }), 400

        # Récupérer les fichiers uploadés
        files = request.files.getlist('files[]')
        saved_files = []

        # Sauvegarder chaque fichier valide
        for file in files: #pour chaque fichier
            if file and allowed_file(file.filename): #si le fichier est valide
                filename = secure_filename(file.filename) #pour sécuriser le nom du fichier
                file_path = os.path.join(user_dir, filename) #pour enregistrer le fichier
                file.save(file_path) #pour enregistrer le fichier
                saved_files.append(filename) #pour ajouter le nom du fichier a la liste
                logger.info(f"Saved file: {filename}") #pour logger le message

        # Vérifier si des fichiers valides ont été sauvegardés
        if not saved_files: #si aucun fichier valide n'est sauvegardé
            return jsonify({ #pour renvoyer une erreur
                'status': 'error',
                'message': 'No valid files were uploaded'
            }), 400

        # Traiter les images uploadées
        try:
            logger.info("Starting face cropping...") #pour logger le message
            crop("./data", "./cropped_faces") #pour couper les visages
            logger.info("Face cropping completed") #pour logger le message
            logger.info("Starting embedding extraction...") #pour logger le message
            extract_embedding(
                input_dir="./cropped_faces", #pour le répertoire des visages coupés
                output_dir="./embeddings", #pour le répertoire des embeddings
                emb_file="embs_facenet512.pkl", #pour le fichier d'embeddings
                norm_dir="./norm_faces" #pour le répertoire des visages normés
            )
            logger.info("Embedding extraction completed")

            # Recharger les embeddings
            global embs #pour globaliser la variable embs
            with open("./embeddings/embs_facenet512.pkl", "rb") as file: #pour ouvrir le fichier d'embeddings
                embs = pickle.load(file) #pour charger les embeddings
            logger.info("Embeddings reloaded successfully") #pour logger le message

            return jsonify({ #pour renvoyer une erreur
                'status': 'success',
                'message': 'Files uploaded and processed successfully',
                'files': saved_files
            }), 200

        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing images: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }), 500

# API pour supprimer les données d'un utilisateur
@app.route('/api/users/<username>', methods=['DELETE'])
def delete_user(username):
    try:
        username = secure_filename(username)  # Sécuriser le nom d'utilisateur

        # Définir les chemins des répertoires et fichiers à supprimer
        directories = {
            'data': os.path.join('./data', username),
            'cropped_faces': os.path.join('./cropped_faces', username),
            'norm_faces': os.path.join('./norm_faces', username)
        }

        deleted = False #pour initialiser la variable deleted
        for dir_name, dir_path in directories.items(): #pour chaque répertoire
            if os.path.exists(dir_path): #si le répertoire existe
                shutil.rmtree(dir_path) #pour supprimer le répertoire
                deleted = True #pour mettre a jour la variable deleted
                logger.info(f"Deleted {dir_name} directory for user: {username}") #pour logger le message

        # Supprimer les fichiers d'embeddings spécifiques
        emb_file = os.path.join('./embeddings', f'{username}.pkl')
        if os.path.exists(emb_file):
            os.remove(emb_file) #pour supprimer le fichier d'embeddings
            deleted = True #pour mettre a jour la variable deleted
            logger.info(f"Deleted embeddings file for user: {username}") #pour logger le message

        if deleted: #si le répertoire a été supprimé
            # Recharger les embeddings après suppression
            try:
                global embs #pour globaliser la variable embs
                with open("./embeddings/embs_facenet512.pkl", "rb") as file:
                    embs = pickle.load(file) #pour charger les embeddings
                logger.info("Embeddings reloaded successfully after user deletion") #pour logger le message
            except Exception as e:
                logger.warning(f"Could not reload embeddings after deletion: {str(e)}") #pour logger le message

            return jsonify({ #pour renvoyer une erreur
                'status': 'success',
                'message': f'Successfully deleted all data for user: {username}'
            })
        else:
            return jsonify({ #pour renvoyer une erreur
                'status': 'error',
                'message': 'User data not found'
            }), 404

    except Exception as e:
        logger.error(f"Error deleting user {username}: {str(e)}") #pour logger le message
        return jsonify({ #pour renvoyer une erreur
            'status': 'error',
            'message': f'Error deleting user: {str(e)}'
        }), 500

# Lancer l'application Flask
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, threaded=True) # 