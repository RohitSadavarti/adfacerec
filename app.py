import os
import json
import numpy as np
import psycopg2
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
from geopy.distance import geodesic

app = Flask(__name__)
CORS(app)  # Enables your Flutter app to talk to this server

# COORDINATES OF YOUR COLLEGE (Example: Mumbai University)
app.config['COLLEGE_COORDS'] = (19.2110, 72.1408) 
app.config['GEOFENCE_RADIUS_METERS'] = 200 # Allowed radius

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
    try:
        # Test DB connection
        conn = get_db_connection()
        conn.close()
        return "✅ Attendance API is Live & Database is Connected!"
    except Exception as e:
        return f"❌ API Live, but DB Error: {str(e)}"

@app.route('/api/student/stats', methods=['GET'])
def get_student_stats():
    roll_number = request.args.get('roll_number') 
    
    if not roll_number:
        return jsonify({"error": "Roll number required"}), 400

    try:
        conn = get_db_connection() # FIX: Was get_pg_connection
        cur = conn.cursor()
        
        # Query 1: Overall Attendance
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE attendance = 'P') as present,
                COUNT(*) FILTER (WHERE attendance = 'A') as absent
            FROM Attendance 
            WHERE roll_number = %s
        """, (roll_number,))
        overall = cur.fetchone()
        
        # Query 2: Semester/Subject-wise Breakdown
        cur.execute("""
            SELECT 
                subject,
                COUNT(*) FILTER (WHERE attendance = 'P') as present,
                COUNT(*) as total
            FROM Attendance 
            WHERE roll_number = %s
            GROUP BY subject
        """, (roll_number,))
        subjects = cur.fetchall()
        
        cur.close()
        conn.close()

        subject_data = []
        for sub in subjects:
            pct = (sub[1] / sub[2] * 100) if sub[2] > 0 else 0
            subject_data.append({
                "subject": sub[0],
                "present": sub[1],
                "total": sub[2],
                "percentage": round(pct, 1)
            })

        return jsonify({
            "overall": {
                "present": overall[0] or 0,
                "absent": overall[1] or 0,
                "total": (overall[0] or 0) + (overall[1] or 0)
            },
            "subjects": subject_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mobile/mark_attendance', methods=['POST'])
def mobile_mark_attendance():
    # 1. Check Location First
    try:
        student_lat = float(request.form.get('latitude'))
        student_long = float(request.form.get('longitude'))
        student_coords = (student_lat, student_long)
        
        distance = geodesic(app.config['COLLEGE_COORDS'], student_coords).meters
        
        if distance > app.config['GEOFENCE_RADIUS_METERS']:
            return jsonify({
                "match": False, 
                "message": f"You are {int(distance)}m away. Please be inside college campus."
            }), 403
            
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid location data"}), 400

    # 2. Proceed with Face Recognition (Reusing mark_attendance logic)
    return mark_attendance()

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    conn = None
    try:
        file = request.files['file']
        
        # 1. Process image
        unknown_image = face_recognition.load_image_file(file)
        unknown_encodings = face_recognition.face_encodings(unknown_image)

        if len(unknown_encodings) == 0:
            return jsonify({"error": "No face detected in the image"}), 400
        
        unknown_encoding = unknown_encodings[0]

        # 2. Fetch DB faces
        conn = get_db_connection()
        cur = conn.cursor()
        
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
            known_ids.append(s_id)
            known_encodings.append(np.array(json.loads(encoding_json)))
            student_info[s_id] = {"name": name, "roll_number": roll_no}

        # 3. Compare
        face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
        best_match_index = np.argmin(face_distances)
        min_distance = face_distances[best_match_index]
        
        if min_distance < 0.5:
            matched_id = known_ids[best_match_index]
            student = student_info[matched_id]
            confidence = (1 - min_distance) * 100

            # 4. Log
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

@app.route('/api/student/login', methods=['POST'])
def student_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    try:
        conn = get_db_connection() # FIX: Was get_pg_connection
        cur = conn.cursor()
        
        # Verify Creds
        cur.execute("""
            SELECT username 
            FROM std_user_login 
            WHERE username = %s AND password_hash = %s
        """, (username, password))
        
        user = cur.fetchone()
        
        if user:
            # Fetch Profile
            cur.execute("""
                SELECT name, department, class 
                FROM students 
                WHERE roll_number = %s
            """, (username,))
            student_details = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if student_details:
                return jsonify({
                    "success": True,
                    "data": {
                        "roll_number": username,
                        "name": student_details[0],
                        "department": student_details[1],
                        "class": student_details[2]
                    }
                })
            else:
                return jsonify({"success": False, "message": "Login OK, but Profile not found in 'students' table"}), 404
                
        cur.close()
        conn.close()
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    except Exception as e:
        print(f"LOGIN ERROR: {str(e)}")
        return jsonify({"success": False, "message": f"Server Error: {str(e)}"}), 500

@app.route('/setup_face_table', methods=['GET'])
def setup_face_table():
    """
    Creates the table to store Face Encodings.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create table to link Student ID with their Face Encoding (stored as text/json)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS student_face_data (
                student_id VARCHAR(50) PRIMARY KEY,
                face_encoding TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"status": "success", "message": "Table 'student_face_data' created successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/update_db', methods=['GET'])
def update_db():
    """
    Scans the 'dataset/' folder, generates vectors, and saves them to Supabase.
    """
    dataset_path = "dataset" # Folder must exist in your repo
    
    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset folder not found. Please push 'dataset/' to GitHub."}), 404

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        added_count = 0
        errors = []
        
        # 1. Loop through Student Folders (e.g., dataset/CS23001)
        for student_roll_no in os.listdir(dataset_path):
            student_folder = os.path.join(dataset_path, student_roll_no)
            
            if os.path.isdir(student_folder):
                all_encodings = []
                
                # 2. Process all images for this student
                for filename in os.listdir(student_folder):
                    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                        image_path = os.path.join(student_folder, filename)
                        try:
                            # Load image & Generate Vector
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            
                            if len(encodings) > 0:
                                all_encodings.append(encodings[0])
                        except Exception as e:
                            errors.append(f"Error processing {filename}: {e}")

                # 3. Average the vectors (if multiple images) and Save
                if len(all_encodings) > 0:
                    # Calculate mean vector across all images for better accuracy
                    avg_encoding = np.mean(all_encodings, axis=0).tolist()
                    
                    # 4. Insert into Supabase
                    cur.execute("""
                        INSERT INTO student_face_data (student_id, face_encoding) 
                        VALUES (%s, %s)
                        ON CONFLICT (student_id) 
                        DO UPDATE SET face_encoding = EXCLUDED.face_encoding;
                    """, (student_roll_no, json.dumps(avg_encoding)))
                    
                    added_count += 1
                else:
                    errors.append(f"No faces found for {student_roll_no}")
        
        conn.commit()
        return jsonify({
            "status": "success",
            "message": f"Successfully updated vectors for {added_count} students.",
            "errors": errors
        })

    except Exception as e:
        if conn: conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if conn: conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
