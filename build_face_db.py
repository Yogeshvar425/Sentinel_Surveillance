import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace

KNOWN_FACES_DIR = "/home/villain8001/known_faces"
DB_PATH         = "/home/villain8001/face_db.pkl"
MODEL_NAME      = "Facenet512"   # best accuracy/speed balance

known_faces = {}

print("Building face database...")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []
    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_dir, img_file)
        try:
            result = DeepFace.represent(
                img_path     = img_path,
                model_name   = MODEL_NAME,
                detector_backend = "retinaface",
                enforce_detection = True
            )
            emb = np.array(result[0]['embedding'])
            embeddings.append(emb / np.linalg.norm(emb))  # normalize
            print(f"  ✅ {person_name}: {img_file}")
        except Exception as e:
            print(f"  ❌ {person_name}/{img_file}: {e}")

    if embeddings:
        # PATCH 1: re-normalize the mean — the average of unit vectors is NOT unit length,
        # so storing it raw deflates every cosine score in the engine.
        mean_emb = np.mean(embeddings, axis=0)
        known_faces[person_name] = mean_emb / np.linalg.norm(mean_emb)
        print(f"✅ {person_name} enrolled with {len(embeddings)} photos")
    else:
        print(f"❌ {person_name}: no valid faces found")

with open(DB_PATH, 'wb') as f:
    pickle.dump(known_faces, f)

print(f"\n✅ Database saved! {len(known_faces)} persons: {list(known_faces.keys())}")
