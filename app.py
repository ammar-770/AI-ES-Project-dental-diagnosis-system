from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from auth import db, bcrypt, User, login_manager
from models import Patient, MedicalRecord, AnalysisResult, AccessLog, Doctor, Booking, PatientRecord
from flask_login import login_required  # Add this import
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key_here'

CORS(app, origins=["http://127.0.0.1:5500"])
db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    is_doctor = db.Column(db.Boolean, default=False)

class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    specialty = db.Column(db.String(100), nullable=False)
    experience = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, default=0.0)
    contact = db.Column(db.String(100), nullable=False)
    bio = db.Column(db.Text)

class Booking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'))
    date = db.Column(db.String(50), nullable=False)
    time = db.Column(db.String(50), nullable=False)

class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    details = db.Column(db.Text)
    image_path = db.Column(db.String(255))
    analysis_result = db.Column(db.Text)

# Update the model path to use absolute path or correct relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "oral_disease_classifier.keras")  # Change .h5 to .keras
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
CLASS_NAMES = ['cavity', 'healthy', 'discoloration', 'caries', 'gingivitis', 'ulcer']

# Optimize prediction with tf.function
@tf.function
def fast_predict(img_array):
    return model(img_array, training=False)

# Warm up the model at startup
_dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
_ = fast_predict(_dummy)

def preprocess_image(image_bytes):
    # Convert bytes to image
    img = Image.open(io.BytesIO(image_bytes))
    # Resize image
    img = img.resize((224, 224))
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        app.logger.info("Predict endpoint called")
        if 'file' not in request.files:
            app.logger.error('No file part in the request')
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        app.logger.info(f"File received: {file.filename}")
        if file.filename == '':
            app.logger.error('No file selected for uploading')
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the file once
        img_bytes = file.read()
        app.logger.info(f"Received file: {file.filename}, Size: {len(img_bytes)} bytes")
        
        # Preprocess the image
        img_array = preprocess_image(img_bytes)
        
        # Make prediction (optimized)
        predictions = fast_predict(tf.convert_to_tensor(img_array, dtype=tf.float32)).numpy()[0]
        
        # Return all predictions, sorted by confidence
        results = []
        for i, confidence in enumerate(predictions):
            results.append({
                'disease': CLASS_NAMES[i],
                'confidence': float(confidence * 100)
            })
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return jsonify({
            'results': results,
            'success': True
        })
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user = User(name=data['name'], email=data['email'], password=hashed_pw, is_doctor=data.get('is_doctor', False))
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user_type = data.get('user_type', 'patient')
    if user_type == 'admin':
        # Only allow admin login with email 'admin' and keyword 'ammar'
        keyword = data.get('keyword')
        if data['email'] != 'admin' or keyword != 'ammar':
            return jsonify({'message': 'Invalid admin credentials'}), 401
        # Simulate admin user
        return jsonify({'message': 'Login successful', 'user_id': 0, 'is_doctor': False, 'name': 'Admin'}), 200
    else:
        user = User.query.filter_by(email=data['email']).first()
        if user and bcrypt.check_password_hash(user.password, data['password']):
            return jsonify({'message': 'Login successful', 'user_id': user.id, 'is_doctor': user.is_doctor, 'name': user.name})
        return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/patients', methods=['GET'])
def get_patients():
    patients = Patient.query.all()
    return jsonify([patient.to_dict() for patient in patients])

@app.route('/api/patients/<int:id>', methods=['GET'])
def get_patient(id):
    patient = Patient.query.get_or_404(id)
    return jsonify(patient.to_dict())

@app.route('/api/patients', methods=['POST'])
def create_patient():
    try:
        data = request.get_json()
        if not data or 'first_name' not in data or 'last_name' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        patient = Patient(
            first_name=data['first_name'],
            last_name=data['last_name'],
            email=data.get('email', '')
        )
        db.session.add(patient)
        db.session.commit()
        return jsonify(patient.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/records/<int:patient_id>', methods=['GET'])
def get_patient_records(patient_id):
    records = MedicalRecord.query.filter_by(patient_id=patient_id).all()
    return jsonify([record.to_dict() for record in records])

@app.route('/api/records', methods=['POST'])
@login_required
def create_record():
    data = request.get_json()
    record = MedicalRecord(
        patient_id=data['patient_id'],
        diagnosis=data['diagnosis'],
        treatment=data['treatment']
    )
    db.session.add(record)
    db.session.commit()
    # Log the analysis action
    log = AccessLog(
        doctor_name=data.get('doctor_name'),
        patient_name=data.get('patient_name'),
        patient_id=data.get('patient_id'),
        action='Analyze Image',
        details=f"Analyzed image for {data.get('patient_name')} (ID: {data.get('patient_id')})"
    )
    db.session.add(log)
    db.session.commit()
    return jsonify(record.to_dict()), 201

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = AccessLog.query.order_by(AccessLog.timestamp.desc()).all()
    return jsonify([
        {
            'timestamp': log.timestamp.strftime('%Y-%m-%d %H:%M'),
            'doctor_name': log.doctor_name,
            'patient_name': log.patient_name,
            'patient_id': log.patient_id,
            'action': log.action,
            'details': log.details
        } for log in logs
    ])

# Configure logging
LOG_FOLDER = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_FOLDER, exist_ok=True)

# Create a rotating file handler
handler = RotatingFileHandler(
    os.path.join(LOG_FOLDER, 'app.log'),
    maxBytes=10000,
    backupCount=3
)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Load the fine-tuned GPT2 model and tokenizer for AI text generation
GPT2_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "gpt2-dental-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_PATH)
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_PATH)

def generate_response(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    if request.method == 'OPTIONS':
        # CORS preflight
        return '', 204
    data = request.get_json()
    question = data.get('question') or data.get('message') or ''
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    prompt = f"Patient: {question}\n    AI:"
    response = generate_response(prompt)
    ai_response = response.split('AI:')[-1].strip()
    return jsonify({
        'question': question,
        'response': ai_response
    })

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_html(filename):
    # Serve any HTML file from the templates directory
    if filename.endswith('.html'):
        return render_template(filename)
    # Optionally serve static files if needed
    return send_from_directory('../static', filename)

@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    doctors = Doctor.query.all()
    return jsonify([
        {'id': d.id, 'name': d.name, 'specialty': d.specialty, 'experience': d.experience, 'rating': d.rating, 'contact': d.contact, 'bio': d.bio}
        for d in doctors
    ])

@app.route('/api/book', methods=['POST'])
def book():
    data = request.get_json()
    booking = Booking(user_id=data['user_id'], doctor_id=data['doctor_id'], date=data['date'], time=data['time'])
    db.session.add(booking)
    db.session.commit()
    return jsonify({'message': 'Booking successful'})

@app.route('/api/bookings', methods=['GET'])
def get_bookings():
    user_id = request.args.get('user_id')
    doctor_id = request.args.get('doctor_id')
    if user_id:
        bookings = Booking.query.filter_by(user_id=user_id).all()
    elif doctor_id:
        bookings = Booking.query.filter_by(doctor_id=doctor_id).all()
    else:
        bookings = Booking.query.all()
    return jsonify([
        {'id': b.id, 'user_id': b.user_id, 'doctor_id': b.doctor_id, 'date': b.date, 'time': b.time}
        for b in bookings
    ])

@app.route('/api/patient/records', methods=['GET', 'POST'])
def patient_records():
    if request.method == 'GET':
        user_id = request.args.get('user_id')
        records = PatientRecord.query.filter_by(user_id=user_id).all()
        return jsonify([
            {'id': r.id, 'user_id': r.user_id, 'details': r.details, 'image_path': r.image_path, 'analysis_result': r.analysis_result}
            for r in records
        ])
    else:
        data = request.get_json()
        record = PatientRecord(user_id=data['user_id'], details=data.get('details', ''), image_path=data.get('image_path', ''), analysis_result=data.get('analysis_result', ''))
        db.session.add(record)
        db.session.commit()
        return jsonify({'message': 'Record added'})

@app.route('/api/users', methods=['GET'])
def get_users():
    search = request.args.get('search', '')
    query = User.query
    if search:
        query = query.filter(
            (User.name.ilike(f'%{search}%')) |
            (User.email.ilike(f'%{search}%'))
        )
    users = query.all()
    return jsonify([
        {'id': u.id, 'name': u.name, 'email': u.email, 'is_doctor': u.is_doctor}
        for u in users
    ])

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000, threaded=True)