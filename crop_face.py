from deepface import DeepFace
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt




def crop(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    
    # Iterate through person folders
    for person_folder in os.listdir(source_dir):
        person_path = os.path.join(source_dir, person_folder)
        if not os.path.isdir(person_path):
            continue
            
        # Create corresponding folder in target directory
        person_target_dir = os.path.join(target_dir, person_folder)
        os.makedirs(person_target_dir, exist_ok=True)
        
        # Process each image in the person's folder
        for filename in os.listdir(person_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_path, filename)
                
                try:
                    faces = DeepFace.extract_faces(
                        img_path=img_path, 
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    
                    for i, face_dict in enumerate(faces):
                        x = face_dict['facial_area']['x']
                        y = face_dict['facial_area']['y']
                        w = face_dict['facial_area']['w']
                        h = face_dict['facial_area']['h']
                        
                        img = cv2.imread(img_path)
                        face_img = img[y:y+h, x:x+w]
                        
                        # Save with person's name in the filename
                        output_path = os.path.join(person_target_dir, f'{person_folder}_{filename}_{i}.jpg')
                        cv2.imwrite(output_path, face_img)
                        
                        print(f"Saved face: {output_path}")
                
                except Exception as e:
                    print(f"Error with {filename}: {e}")

# Usage example:
crop("./data", "./cropped_faces")