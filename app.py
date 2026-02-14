from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import psycopg2
import json
import os

app = Flask(__name__)

# Database Configuration (Update with your credentials)
DB_CONFIG = {
    "dbname": "your_db_name",
    "user": "your_db_user",
    "password": "your_db_password",
    "host": "localhost"
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@app.route('/register_face', methods=['POST'])
def register_face():
    """
    Endpoint to register a student's face.
    Expects 'student_id' and 'file' (image) in the request.
    """
    if 'file' not in request.files or 'student_id' not in request.form:
        return jsonify({"error": "Missing file or student_id"}), 400

    file = request.files['file']
    student_id = request.form['student_id']

    # Load image and get encoding
    image = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face detected"}), 400
    
    # Take the first face found
    face_encoding = encodings[0].tolist() # Convert to standard list

    # Save to Database
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO "public"."student_face_data" (student_id, face_encoding) VALUES (%s, %s)',
            (student_id, json.dumps(face_encoding))
        )
        conn.commit()
        return jsonify({"message": "Face registered successfully"}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    """
    Endpoint to recognize a face and mark attendance.
    Expects 'file' (image) in the request.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    # 1. Process incoming image
    unknown_image = face_recognition.load_image_file(file)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if len(unknown_encodings) == 0:
        return jsonify({"error": "No face detected in camera feed"}), 400
    
    unknown_encoding = unknown_encodings[0]

    # 2. Fetch all known faces from DB
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT student_id, face_encoding FROM "public"."student_face_data"')
    rows = cur.fetchall()
    
    known_ids = []
    known_encodings = []

    for row in rows:
        known_ids.append(row[0])
        known_encodings.append(np.array(json.loads(row[1])))
    
    if not known_encodings:
        return jsonify({"error": "No students registered"}), 404

    # 3. Compare faces
    # matches returns a list of True/False
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
    
    # Calculate distance (lower is better) to find the *best* match
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)

    response = {"match": False, "student": None}

    if matches[best_match_index]:
        matched_student_id = known_ids[best_match_index]
        confidence = 1 - face_distances[best_match_index] # Simple confidence metric
        
        # Log Attendance
        cur.execute(
            'INSERT INTO "public"."attendance_logs" (student_id, confidence_score) VALUES (%s, %s)',
            (matched_student_id, float(confidence))
        )
        conn.commit()
        
        # Get Student Details for UI
        cur.execute('SELECT name, roll_number FROM "public"."students" WHERE id = %s', (matched_student_id,))
        student_details = cur.fetchone()
        
        response = {
            "match": True,
            "student_id": matched_student_id,
            "name": student_details[0],
            "roll_no": student_details[1],
            "confidence": f"{confidence:.2%}"
        }

    cur.close()
    conn.close()
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
