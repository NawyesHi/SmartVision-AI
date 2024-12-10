from flask import Flask, render_template, Response, jsonify, make_response, request, redirect, url_for, send_from_directory
import cv2
import traceback
import time
import threading
import logging
from deepface import DeepFace
from deepface.modules.verification import find_distance
import pickle
import numpy as np
from ultralytics import YOLO
import supervision as sv
from werkzeug.utils import secure_filename
import os
from crop_face import crop
from extract_embs import extract_embedding
import shutil
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    
# Face recognition settings
model_name = "Facenet512"
metrics = [
    {"cosine": 0.25},           # Reduced from 0.30
    {"euclidean": 15.0},        # Reduced from 20.0
    {"euclidean_l2": 0.68}      # Reduced from 0.78
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

UPLOAD_FOLDER = './data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            logger.info("Camera released")

def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            time.sleep(2)
        return camera

def process_frame(frame):
    try:
        # YOLO object detection
        yolo_results = yolo_model(frame)[0]
        detections = sv.Detections.from_ultralytics(yolo_results)
        detections = detections[detections.confidence > 0.3]

        # Draw object detection boxes
        for bbox, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = map(int, bbox)
            # Get class name from COCO_CLASSES
            class_name = COCO_CLASSES[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Face detection and recognition
        results = DeepFace.extract_faces(frame, detector_backend="yolov8", 
                                       enforce_detection=False)

        for result in results:
            if result["confidence"] >= 0.5:
                x = result["facial_area"]["x"]
                y = result["facial_area"]["y"]
                w = result["facial_area"]["w"]
                h = result["facial_area"]["h"]

                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                # Get face embedding
                emb = DeepFace.represent(
                    face_img,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]

                # Normalize embedding
                emb = np.array(emb) / np.linalg.norm(emb)

                # Find closest match with improved threshold checking
                min_dist = float("inf")
                match_name = "Unknown"
                
                # Use the same thresholds as in face_recognition.py
                cosine_threshold = 0.35
                euclidean_l2_threshold = 0.85

                for name, stored_emb in embs.items():
                    if isinstance(stored_emb, list):
                        # Calculate distances for all embeddings of this person
                        distances = [find_distance(emb, se, "euclidean_l2") for se in stored_emb]
                        dist = min(distances)
                    else:
                        stored_emb = np.array(stored_emb) / np.linalg.norm(stored_emb)
                        dist = find_distance(emb, stored_emb, "euclidean_l2")

                    if dist < min_dist:
                        min_dist = dist
                        if dist < euclidean_l2_threshold:
                            match_name = name

                # Final verification
                if min_dist >= euclidean_l2_threshold:
                    match_name = "Unknown"

                # Draw face detection box and name
                color = (0, 255, 0) if match_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{match_name} ({min_dist:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, color, 2)

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        traceback.print_exc()

    return frame

def generate_frames():
    while True:
        try:
            cam = get_camera()
            if cam is not None and cam.isOpened():
                ret, frame = cam.read()
                if ret:
                    # Process frame with detection and recognition
                    processed_frame = process_frame(frame)
                    
                    # Encode and yield the frame
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               buffer.tobytes() + b'\r\n')
                else:
                    break
            else:
                break
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            break


@app.route('/api/check_camera', methods=['GET'])
def check_camera():
    logger.info("Checking camera...")
    try:
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            logger.error("Could not open camera")
            return make_response(jsonify({
                "status": "error", 
                "message": "Could not open camera"
            }), 500)
        
        ret, frame = test_cap.read()
        test_cap.release()
        
        if not ret:
            logger.error("Could not read from camera")
            return make_response(jsonify({
                "status": "error", 
                "message": "Could not read from camera"
            }), 500)
            
        logger.info(f"Camera check successful - frame size: {frame.shape}")
        return make_response(jsonify({
            "status": "success",
            "message": "Camera check successful"
        }), 200)
        
    except Exception as e:
        logger.error(f"Camera check error: {str(e)}")
        traceback.print_exc()
        return make_response(jsonify({
            "status": "error",
            "message": str(e)
        }), 500)

@app.route('/api/start_camera', methods=['GET'])
def start_camera():
    logger.info("Starting camera...")
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
            logger.error("Failed to read from camera")
            release_camera()
            return make_response(jsonify({
                "status": "error",
                "message": "Failed to read from camera"
            }), 500)
            
        logger.info(f"Camera started successfully - frame size: {frame.shape}")
        return make_response(jsonify({
            "status": "success",
            "message": "Camera started successfully"
        }), 200)
        
    except Exception as e:
        logger.error(f"Start camera error: {str(e)}")
        traceback.print_exc()
        return make_response(jsonify({
            "status": "error",
            "message": str(e)
        }), 500)

@app.route('/api/stop_camera', methods=['GET'])
def stop_camera():
    logger.info("Stopping camera...")
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
        }), 500)

@app.route('/api/video_feed')
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

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/')
def root():
    return redirect(url_for('welcome'))

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def upload_files():
    try:
        username = request.form.get('username')
        if not username:
            return jsonify({
                'status': 'error',
                'message': 'Username is required'
            }), 400

        # Create user directory
        user_dir = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(username))
        os.makedirs(user_dir, exist_ok=True)

        if 'files[]' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No files uploaded'
            }), 400

        files = request.files.getlist('files[]')
        saved_files = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(user_dir, filename)
                file.save(file_path)
                saved_files.append(filename)

        if saved_files:
            # Run face cropping
            logger.info("Starting face cropping...")
            crop("./data", "./cropped_faces")
            logger.info("Face cropping completed")

            # Run embedding extraction
            logger.info("Starting embedding extraction...")
            extract_embedding(
                input_dir="./cropped_faces",
                output_dir="./embeddings",
                emb_file="embs_facenet512.pkl",
                norm_dir="./norm_faces"
            )
            logger.info("Embedding extraction completed")

            # Reload embeddings for face recognition
            global embs
            with open("./embeddings/embs_facenet512.pkl", "rb") as file:
                embs = pickle.load(file)
            logger.info("Embeddings reloaded successfully")

        return jsonify({
            'status': 'success',
            'message': f'Successfully processed {len(saved_files)} files for {username}',
            'files': saved_files
        }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/users/<username>', methods=['DELETE'])
def delete_user(username):
    try:
        # Secure the username to prevent directory traversal
        username = secure_filename(username)
        
        # Define all directories where user data might exist
        directories = {
            'data': os.path.join('./data', username),
            'cropped_faces': os.path.join('./cropped_faces', username),
            'norm_faces': os.path.join('./norm_faces', username)
        }

        deleted = False
        for dir_name, dir_path in directories.items():
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                deleted = True
                logger.info(f"Deleted {dir_name} directory for user: {username}")

        # Also remove any user-specific files in embeddings directory
        emb_file = os.path.join('./embeddings', f'{username}.pkl')
        if os.path.exists(emb_file):
            os.remove(emb_file)
            deleted = True
            logger.info(f"Deleted embeddings file for user: {username}")

        if deleted:
            # Reload embeddings after deletion
            try:
                global embs
                with open("./embeddings/embs_facenet512.pkl", "rb") as file:
                    embs = pickle.load(file)
                logger.info("Embeddings reloaded successfully after user deletion")
            except Exception as e:
                logger.warning(f"Could not reload embeddings after deletion: {str(e)}")

            return jsonify({
                'status': 'success',
                'message': f'Successfully deleted all data for user: {username}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'User data not found'
            }), 404

    except Exception as e:
        logger.error(f"Error deleting user {username}: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error deleting user: {str(e)}'
        }), 500

@app.route('/api/users')
def get_users():
    try:
        users = []
        data_dir = './data'
        
        # Debug log
        print(f"Checking directory: {data_dir}")
        
        if os.path.exists(data_dir):
            # Get only directories, not files
            users = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
            
            # Debug log
            print(f"Found users: {users}")
            
            return jsonify(users)
        else:
            print(f"Data directory not found: {data_dir}")
            return jsonify([])
            
    except Exception as e:
        print(f"Error getting users: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/users/<username>/images')
def get_user_images(username):
    try:
        # Secure the username and create the path
        username = secure_filename(username)
        user_dir = os.path.join('./data', username)
        
        # Debug logging
        print(f"Checking directory: {user_dir}")
        
        if not os.path.exists(user_dir):
            print(f"Directory not found: {user_dir}")
            return jsonify([])
        
        # Get all image files
        images = [f for f in os.listdir(user_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        print(f"Found images: {images}")  # Debug logging
        return jsonify(images)
        
    except Exception as e:
        print(f"Error in get_user_images: {str(e)}")  # Debug logging
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Add a route to serve images directly
@app.route('/data/<username>/<filename>')
def serve_user_image(username, filename):
    try:
        username = secure_filename(username)
        filename = secure_filename(filename)
        return send_from_directory('data', f'{username}/{filename}')
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 404

@app.route('/api/users/edit', methods=['POST'])
def edit_user():
    try:
        new_username = secure_filename(request.form['newUsername'])
        original_username = secure_filename(request.form['originalUsername'])
        kept_images = json.loads(request.form.get('keptImages', '[]'))
        
        print(f"Editing user: {original_username} -> {new_username}")  # Debug log
        
        # Handle username change
        if new_username != original_username:
            directories = ['data', 'cropped_faces', 'norm_faces']
            for dir_name in directories:
                old_path = os.path.join(f'./{dir_name}', original_username)
                new_path = os.path.join(f'./{dir_name}', new_username)
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed directory: {old_path} -> {new_path}")  # Debug log

        # Get the user directory path
        user_dir = os.path.join('./data', new_username)
        os.makedirs(user_dir, exist_ok=True)

        # Handle deleted files
        existing_files = set(os.listdir(user_dir))
        files_to_keep = set(kept_images)
        files_to_delete = existing_files - files_to_keep

        for filename in files_to_delete:
            file_path = os.path.join(user_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")  # Debug log

        # Handle new files
        if 'newFiles' in request.files:
            new_files = request.files.getlist('newFiles')
            for file in new_files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(user_dir, filename)
                    file.save(file_path)
                    print(f"Saved new file: {file_path}")  # Debug log

        print("Starting face processing...")  # Debug log

        # Process faces
        try:
            # Run face cropping
            print("Running face cropping...")
            crop("./data", "./cropped_faces")
            
            # Run embedding extraction
            print("Running embedding extraction...")
            extract_embedding(
                input_dir="./cropped_faces",
                output_dir="./embeddings",
                emb_file="embs_facenet512.pkl",
                norm_dir="./norm_faces"
            )
            
            # Reload embeddings
            print("Reloading embeddings...")
            global embs
            with open("./embeddings/embs_facenet512.pkl", "rb") as file:
                embs = pickle.load(file)
                
            print("Face processing completed successfully")  # Debug log
            
        except Exception as e:
            print(f"Error during face processing: {str(e)}")  # Debug log
            raise Exception(f"Face processing failed: {str(e)}")

        return jsonify({
            'status': 'success',
            'message': f'User {new_username} updated successfully! All faces have been processed and embeddings updated.'
        })

    except Exception as e:
        print(f"Error in edit_user: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

# Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return make_response(jsonify({
        "status": "error",
        "message": "Not found"
    }), 404)

@app.errorhandler(500)
def internal_error(error):
    return make_response(jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500)

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    traceback.print_exc()
    return make_response(jsonify({
        "status": "error",
        "message": str(e)
    }), 500)

@app.route('/api/reset-embeddings')
def reset_embeddings():
    try:
        # 1. Clear all embeddings files
        embeddings_dir = "./embeddings"
        if os.path.exists(embeddings_dir):
            for file in os.listdir(embeddings_dir):
                file_path = os.path.join(embeddings_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

        # 2. Clear cropped_faces directory
        cropped_dir = "./cropped_faces"
        if os.path.exists(cropped_dir):
            shutil.rmtree(cropped_dir)
            os.makedirs(cropped_dir)
            print("Reset cropped_faces directory")

        # 3. Clear norm_faces directory
        norm_dir = "./norm_faces"
        if os.path.exists(norm_dir):
            shutil.rmtree(norm_dir)
            os.makedirs(norm_dir)
            print("Reset norm_faces directory")

        # 4. Rebuild everything from data directory
        print("Starting rebuild process...")
        
        # Crop faces
        crop("./data", "./cropped_faces")
        
        # Extract embeddings
        extract_embedding(
            input_dir="./cropped_faces",
            output_dir="./embeddings",
            emb_file="embs_facenet512.pkl",
            norm_dir="./norm_faces"
        )
        
        # Reload embeddings
        global embs
        with open("./embeddings/embs_facenet512.pkl", "rb") as file:
            embs = pickle.load(file)
        
        print("Rebuild complete")
        
        # 5. Verify current embeddings
        current_users = list(embs.keys()) if embs else []
        print(f"Current users in embeddings: {current_users}")

        return jsonify({
            'status': 'success',
            'message': 'Embeddings reset and rebuilt successfully',
            'current_users': current_users
        })

    except Exception as e:
        print(f"Error in reset_embeddings: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, threaded=True)