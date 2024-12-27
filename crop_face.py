from deepface import DeepFace # Pour la détection de visages
import os # Pour les opérations de fichiers et de répertoires
import cv2 # Pour les opérations d'image        
import matplotlib.pyplot as plt # Pour les visualisations   



#prend en entree le dossier source et le dossier cible et creer kan famech 
def crop(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    
    # boucle sur les dossiers de personnes dans le dossier source
    for person_folder in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_folder)
        # si le dossier n'est pas un dossier, on passe a la prochaine iteration
        if not os.path.isdir(person_path):
            continue
            
        # creer le dossier correspondant dans le dossier cible
        person_target_dir = os.path.join(target_dir, person_folder)
        os.makedirs(person_target_dir, exist_ok=True) # creer le dossier si il n'existe pas
        
        # boucle sur les images dans le dossier de la personne
        for filename in os.listdir(person_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')): # si l'image est un jpg, png ou jpeg
                img_path = os.path.join(person_path, filename)
                
                try:    # si il y a une erreur, on passe a la prochaine iteration
                    faces = DeepFace.extract_faces(
                        img_path=img_path, 
                        detector_backend='opencv',  # detecteur de visage
                        enforce_detection=False # si la detection est fausse, on passe a la prochaine iteration
                    )
                    
                    for i, face_dict in enumerate(faces): # boucle sur les visages detectes (kan barcha face kol face ihotou fi dictionnaire)
                        x = face_dict['facial_area']['x']
                        y = face_dict['facial_area']['y']
                        w = face_dict['facial_area']['w']
                        h = face_dict['facial_area']['h']  # les coordonnees du visage dans l'image
                        
                        img = cv2.imread(img_path) # lire l'image
                        face_img = img[y:y+h, x:x+w] # extraire le visage de l'image
                        
                        # Save with person's name in the filename 
                        output_path = os.path.join(person_target_dir, f'{person_folder}_{filename}_{i}.jpg')
                        cv2.imwrite(output_path, face_img) # enregistrer le visage dans le dossier cible
                        
                        print(f"Saved face: {output_path}")
                
                except Exception as e: # si il y a une erreur, on passe a la prochaine iteration
                    print(f"Error with {filename}: {e}") # afficher l'erreur

# Usage example:
crop("./data", "./cropped_faces")