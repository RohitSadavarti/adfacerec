import os
import json
import numpy as np
import psycopg2
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- DATABASE CONNECTION ---
def get_db_connection():
    # Render provides the database URL automatically in the environment
    # If testing locally, you can set a DATABASE_URL environment variable
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise ValueError("No DATABASE_URL set for Flask application")
    return psycopg2.connect(url)

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return "Attendance API is Running!"

@app.route('/register_face', methods=['POST'])
def register_face():
    if 'file' not in request.files or 'student_id' not in request.form:
        return jsonify({"error": "Missing file or student_id"}), 400

    file = request.files['file']
    student_id = request.form['student_id']

    # Load image and get encoding
    try:
        image = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) == 0:
            return jsonify({"error": "No face detected"}), 400
            
        # Take the first face found
        face_encoding = encodings[0].tolist()

        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if student exists
        cur.execute('SELECT id FROM "public"."students" WHERE roll_number = %s', (student_id,))
        student_row = cur.fetchone()
        
        if not student_row:
             return jsonify({"error": f"Student {student_id} not found in database"}), 404

        db_student_id = student_row[0]

        # Insert face data
        cur.execute(
            'INSERT INTO "public"."student_face_data" (student_id, face_encoding) VALUES (%s, %s)',
            (str(db_student_id), json.dumps(face_encoding))
        )
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({"message": "Face registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files['file']
        unknown_image = face_recognition.load_image_file(file)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) == 0:
            return jsonify({"error": "No face detected"}), 400
        
        unknown_encoding = unknown_encodings[0]

        # Fetch known faces
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT s.id, s.name, s.roll_number, f.face_encoding 
            FROM "public"."student_face_data" f
            JOIN "public"."students" s ON f.student_id = CAST(s.id AS TEXT)
        """)
        rows = cur.fetchall()
        
        if not rows:
            return jsonify({"error": "No registered faces found"}), 404

        known_ids = []
        known_encodings = []
        student_info = {}

        for row in rows:
            s_id, name, roll, encoding_json = row
            known_ids.append(s_id)
            known_encodings.append(np.array(json.loads(encoding_json)))
            student_info[s_id] = {"name": name, "roll_number": roll}

        # Compare faces
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Threshold: 0.6 is standard, 0.5 is stricter
        if face_distances[best_match_index] < 0.5:
            matched_id = known_ids[best_match_index]
            student = student_info[matched_id]
            confidence = (1 - face_distances[best_match_index]) * 100

            # Log attendance
            cur.execute(
                'INSERT INTO "public"."attendance_logs" (student_id, confidence_score) VALUES (%s, %s)',
                (str(matched_id), float(confidence))
            )
            conn.commit()

            response = {
                "match": True,
                "student_id": matched_id,
                "name": student["name"],
                "roll_no": student["roll_number"],
                "confidence": f"{confidence:.2f}%"
            }
        else:
            response = {"match": False, "message": "Face not recognized"}

        cur.close()
        conn.close()
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
