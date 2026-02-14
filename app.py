import os
import json
import numpy as np
import psycopg2
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables your Flutter app to talk to this server

# --- DATABASE CONNECTION ---
def get_db_connection():
    """
    Connects to the Render PostgreSQL database using the URL 
    stored in the environment variable 'DATABASE_URL'.
    """
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise ValueError("No DATABASE_URL set. Check your Render Environment variables.")
    return psycopg2.connect(url)

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    """Health check route to verify server is running."""
    return "✅ Attendance API is Live & Running!"

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    """
    Receives an image from Flutter, matches it against the DB, 
    and logs attendance if a match is found.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    conn = None
    try:
        file = request.files['file']
        
        # 1. Process the uploaded image
        # Load image from the uploaded file directly
        unknown_image = face_recognition.load_image_file(file)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) == 0:
            return jsonify({"error": "No face detected in the image"}), 400
        
        # We assume the first face is the student
        unknown_encoding = unknown_encodings[0]

        # 2. Fetch all known faces from the Database
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get face data + student details in one query
        cur.execute("""
            SELECT s.id, s.name, s.roll_number, f.face_encoding 
            FROM "public"."student_face_data" f
            JOIN "public"."students" s ON f.student_id = CAST(s.id AS TEXT)
        """)
        rows = cur.fetchall()
        
        if not rows:
            return jsonify({"error": "No registered students found in database"}), 404

        known_ids = []
        known_encodings = []
        student_info = {}

        for row in rows:
            s_id, name, roll_no, encoding_json = row
            # Convert JSON string back to numpy array
            known_ids.append(s_id)
            known_encodings.append(np.array(json.loads(encoding_json)))
            student_info[s_id] = {"name": name, "roll_number": roll_no}

        # 3. Compare the uploaded face with all DB faces
        # tolerance=0.5 is strict, 0.6 is standard. Lower is stricter.
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        
        # Find the best match (smallest distance)
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]
        
        # We use 0.5 as the cutoff for a positive match
        if min_distance < 0.5:
            matched_id = known_ids[best_match_index]
            student = student_info[matched_id]
            
            # Calculate a confidence score (0 to 100%)
            confidence = (1 - min_distance) * 100

            # 4. Log the attendance in the database
            cur.execute(
                'INSERT INTO "public"."attendance_logs" (student_id, confidence_score, status) VALUES (%s, %s, %s)',
                (str(matched_id), float(confidence), 'Present')
            )
            conn.commit()

            return jsonify({
                "match": True,
                "student_id": matched_id,
                "name": student["name"],
                "roll_no": student["roll_number"],
                "confidence": f"{confidence:.2f}%"
            })
        else:
            return jsonify({
                "match": False, 
                "message": "Face not recognized. Please try again."
            })

    except Exception as e:
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: conn.close()


@app.route('/register_face', methods=['POST'])
def register_face():
    """
    Registers a single student's face from the mobile app.
    Expects 'file' (image) and 'student_id' (DB ID) in the request.
    """
    if 'file' not in request.files or 'student_id' not in request.form:
        return jsonify({"error": "Missing file or student_id"}), 400

    conn = None
    try:
        file = request.files['file']
        student_id = request.form['student_id']

        # Encode face
        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            return jsonify({"error": "No face detected"}), 400
            
        face_encoding = encodings[0].tolist() # Convert to list for JSON storage

        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert or Update face data
        # Note: This requires a UNIQUE constraint on student_id in your table
        cur.execute('''
            INSERT INTO "public"."student_face_data" (student_id, face_encoding) 
            VALUES (%s, %s)
            ON CONFLICT (student_id) 
            DO UPDATE SET face_encoding = EXCLUDED.face_encoding;
        ''', (student_id, json.dumps(face_encoding)))
        
        conn.commit()
        return jsonify({"message": "Face registered successfully"}), 201

    except Exception as e:
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: conn.close()


@app.route('/update_db', methods=['GET', 'POST'])
def update_db():
    """
    ADMIN TOOL: Scans the 'dataset/' folder on the server (from GitHub)
    and updates the database with new student images.
    Access this URL in your browser to trigger a sync.
    """
    dataset_path = "dataset" # Folder name in your repo
    
    if not os.path.exists(dataset_path):
        return jsonify({
            "error": "Dataset folder not found.",
            "hint": "Did you push the 'dataset' folder to GitHub?"
        }), 404

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        added_count = 0
        errors = []
        
        # Loop through folders in dataset/ (e.g., CS23001, CS23002...)
        for student_roll_no in os.listdir(dataset_path):
            student_folder = os.path.join(dataset_path, student_roll_no)
            
            if os.path.isdir(student_folder):
                all_encodings = []
                
                # Process all images for this student
                for filename in os.listdir(student_folder):
                    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                        image_path = os.path.join(student_folder, filename)
                        try:
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            if len(encodings) > 0:
                                all_encodings.append(encodings[0])
                        except Exception as e:
                            errors.append(f"Skipped {filename}: {str(e)}")

                # If we found valid faces, average them and save
                if len(all_encodings) > 0:
                    avg_encoding = np.mean(all_encodings, axis=0).tolist()
                    
                    # 1. Find the student's DB ID using their Roll Number (Folder Name)
                    cur.execute('SELECT id FROM "public"."students" WHERE roll_number = %s', (student_roll_no,))
                    res = cur.fetchone()
                    
                    if res:
                        db_id = res[0]
                        # 2. Save to Face Data Table
                        cur.execute('''
                            INSERT INTO "public"."student_face_data" (student_id, face_encoding) 
                            VALUES (%s, %s)
                            ON CONFLICT (student_id) 
                            DO UPDATE SET face_encoding = EXCLUDED.face_encoding;
                        ''', (str(db_id), json.dumps(avg_encoding)))
                        added_count += 1
                    else:
                        errors.append(f"Roll No {student_roll_no} not found in 'students' table.")
        
        conn.commit()
        return jsonify({
            "status": "success",
            "students_processed": added_count,
            "errors": errors
        })

    except Exception as e:
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: conn.close()


if __name__ == '__main__':
    # Running locally? Use port 5000.
    # On Render, Gunicorn handles the port.
    app.run(host='0.0.0.0', port=5000, debug=True)
