import os
import face_recognition
import psycopg2
import json
import numpy as np

# Database Configuration
DB_CONFIG = {
    "dbname": "your_db_name",
    "user": "your_db_user",
    "password": "your_db_password",
    "host": "localhost"
}

DATASET_PATH = "dataset" # Folder containing student images

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def import_dataset():
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Loop through every folder in 'dataset'
    for student_id in os.listdir(DATASET_PATH):
        student_folder = os.path.join(DATASET_PATH, student_id)
        
        if os.path.isdir(student_folder):
            print(f"Processing Student: {student_id}...")
            
            # We will average the encodings of all images for better accuracy
            all_encodings = []
            
            for filename in os.listdir(student_folder):
                if filename.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(student_folder, filename)
                    
                    # Load image & detect face
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        all_encodings.append(encodings[0])
                    else:
                        print(f"  Warning: No face found in {filename}")

            if len(all_encodings) > 0:
                # Calculate the average vector (centroid) of the face
                avg_encoding = np.mean(all_encodings, axis=0).tolist()
                
                # Check if student exists in 'students' table first
                cur.execute('SELECT id FROM "public"."students" WHERE roll_number = %s', (student_id,))
                res = cur.fetchone()
                
                if res:
                    db_id = res[0] # The primary key from students table
                    
                    # Insert into face_data table
                    cur.execute(
                        'INSERT INTO "public"."student_face_data" (student_id, face_encoding) VALUES (%s, %s)',
                        (str(db_id), json.dumps(avg_encoding))
                    )
                    print(f"  ✅ Saved encoding for {student_id}")
                else:
                    print(f"  ❌ Student {student_id} not found in main database table.")
            else:
                print(f"  ⚠️ No valid face data for {student_id}")

    conn.commit()
    cur.close()
    conn.close()
    print("\n--- Import Complete ---")

if __name__ == "__main__":
    import_dataset()
