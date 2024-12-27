from deepface import DeepFace # pour la representation des visages
import os # pour les operations de fichiers et de répertoires
from tqdm import tqdm # pour la barre de progression
import pickle # pour la serialisation des   données(save and delete les embeddings)
import cv2 # pour les opérations d'image
import matplotlib.pyplot as plt # pour les visualisations
import numpy as np # pour les opérations numériques

model_name = "Facenet512" # le modele de representation des visages

#appliquer la normalisation CLI pour ameliorer la contraste des images (mettre en temps de gris)
def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # creer un objet CLAHE avec un clipLimit de 2.0 et une taille de grille de 8x8
    return clahe.apply(image) # appliquer la normalisation CLAHE a l'image et retourner l'image normalisee

#extrait les caract 
def extract_embedding(input_dir, output_dir, emb_file, norm_dir, model_name=model_name): 
    os.makedirs(output_dir, exist_ok=True) # creer le dossier de sortie si il n'existe pas
    os.makedirs(norm_dir, exist_ok=True) # creer le dossier de normalisation si il n'existe pas
    
    try: # si le fichier d'embeddings existe
        with open(f"./{output_dir}/{emb_file}", "rb") as file: # ouvrir le fichier d'embeddings
            embs = pickle.load(file) # charger les embeddings
            print("Existing embeddings file loaded successfully.")  # afficher un message de succes
            print(embs.keys()) # afficher les personnes dans le fichier d'embeddings
    except FileNotFoundError: # si le fichier d'embeddings n'existe pas
        embs = {} # creer un dictionnaire vide
        print("No existing embeddings file found. Creating a new one.") # afficher un message 

    # boucle sur les dossiers de personnes dans le dossier source
    for person_folder in tqdm(os.listdir(input_dir)):
        person_path = os.path.join(input_dir, person_folder) # creer le chemin du dossier de la personne
        if not os.path.isdir(person_path): # si le dossier n'est pas un dossier, on passe a la prochaine iteration
            continue
            
        # Create person folder in norm_dir
        person_norm_dir = os.path.join(norm_dir, person_folder) # creer le chemin du dossier de normalisation de la personne
        os.makedirs(person_norm_dir, exist_ok=True) # creer le dossier de normalisation de la personne si il n'existe pas
        
        for img_file in os.listdir(person_path): # boucle sur les images dans le dossier de la personne
            img_path = os.path.join(person_path, img_file) # creer le chemin de l'image
            img_name = f"{person_folder}/{img_file.split('.')[0]}"  # Include person name in the key

            if img_name not in embs: # si l'image n'est pas dans le fichier d'embeddings
                face = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # lire l'image en mode gris
                face_norm = clahe(face) # appliquer la normalisation CLAHE a l'image
                face_norm = cv2.cvtColor(face_norm, cv2.COLOR_GRAY2RGB) # convertir l'image en RGB
                
                # Save normalized image in person's subfolder
                norm_path = os.path.join(person_norm_dir, img_file)
                plt.imsave(norm_path, face_norm)
                
                emb = DeepFace.represent( # representer le visage avec le modele choisi
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
        pickle.dump(embs, file) # enregistrer les embeddings dans le fichier
        print("Embeddings updated and saved.")
        print(embs.keys())

# Usage example:
extract_embedding(
    input_dir="./cropped_faces", # le dossier source des images
    output_dir="./embeddings", # le dossier de sortie des embeddings
    emb_file="embs_facenet512.pkl", # le nom du fichier d'embeddings
    norm_dir="./norm_faces", # le dossier de normalisation des images
)
