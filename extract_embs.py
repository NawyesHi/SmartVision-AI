from deepface import DeepFace
import os
from tqdm import tqdm
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

model_name = "Facenet512"

#appliquer la normalisation CLI
def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

#extrait les caract
def extract_embedding(input_dir, output_dir, emb_file, norm_dir, model_name=model_name):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    
    try:
        with open(f"./{output_dir}/{emb_file}", "rb") as file:
            embs = pickle.load(file)
            print("Existing embeddings file loaded successfully.")
            print(embs.keys())
    except FileNotFoundError:
        embs = {}
        print("No existing embeddings file found. Creating a new one.")

    # Iterate through person folders
    for person_folder in tqdm(os.listdir(input_dir)):
        person_path = os.path.join(input_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        # Create person folder in norm_dir
        person_norm_dir = os.path.join(norm_dir, person_folder)
        os.makedirs(person_norm_dir, exist_ok=True)
        
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img_name = f"{person_folder}/{img_file.split('.')[0]}"  # Include person name in the key

            if img_name not in embs:
                face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face_norm = clahe(face)
                face_norm = cv2.cvtColor(face_norm, cv2.COLOR_GRAY2RGB)
                
                # Save normalized image in person's subfolder
                norm_path = os.path.join(person_norm_dir, img_file)
                plt.imsave(norm_path, face_norm)
                
                emb = DeepFace.represent(
                    face_norm,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="skip",
                )[0]["embedding"]
                
                # Store embedding with person's name
                person_name = person_folder  # Use folder name as person's name
                if person_name not in embs:
                    embs[person_name] = []
                embs[person_name].append(emb)

    # Save the updated dictionary
    with open(f"./{output_dir}/{emb_file}", "wb") as file:
        pickle.dump(embs, file)
        print("Embeddings updated and saved.")
        print(embs.keys())

# Usage example:
extract_embedding(
    input_dir="./cropped_faces",
    output_dir="./embeddings",
    emb_file="embs_facenet512.pkl",
    norm_dir="./norm_faces",
)
