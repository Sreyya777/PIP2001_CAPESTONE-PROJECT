from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Needed for session management

db = SQLAlchemy(app)

# Define Database Model for User History
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    image_name = db.Column(db.String(200), nullable=False)
    predicted_label = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Load pre-trained model
model = tf.keras.models.load_model('plant_identification_model.h5')

# Label mapping
label_mapping = {
    0: 'Alpinia Galanga (Rasna)', 1: 'Amaranthus Viridis (Arive-Dantu)',
    2: 'Artocarpus Heterophyllus (Jackfruit)', 3: 'Azadirachta Indica (Neem)',
    4: 'Basella Alba (Basale)', 5: 'Brassica Juncea (Indian Mustard)',
    6: 'Carissa Carandas (Karanda)', 7: 'Citrus Limon (Lemon)',
    8: 'Ficus Auriculata (Roxburgh fig)', 9: 'Ficus Religiosa (Peepal Tree)',
    10: 'Hibiscus Rosa-sinensis', 11: 'Jasminum (Jasmine)',
    12: 'Mangifera Indica (Mango)', 13: 'Mentha (Mint)',
    14: 'Moringa Oleifera (Drumstick)', 15: 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    16: 'Murraya Koenigii (Curry)', 17: 'Nerium Oleander (Oleander)',
    18: 'Nyctanthes Arbor-tristis (Parijata)', 19: 'Ocimum Tenuiflorum (Tulsi)',
    20: 'Piper Betle (Betel)', 21: 'Plectranthus Amboinicus (Mexican Mint)',
    22: 'Pongamia Pinnata (Indian Beech)', 23: 'Psidium Guajava (Guava)',
    24: 'Punica Granatum (Pomegranate)', 25: 'Santalum Album (Sandalwood)',
    26: 'Syzygium Cumini (Jamun)', 27: 'Syzygium Jambos (Rose Apple)',
    28: 'Tabernaemontana Divaricata (Crape Jasmine)', 29: 'Trigonella Foenum-graecum (Fenugreek)'
}

# Image Preprocessing Function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions
    return image_array

# Process Model Predictions
def process_predictions(predictions):
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    return predicted_label, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    uploaded_image = request.files['image']

    if uploaded_image.filename != '':
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)

        # Extract classification details
        predicted_label, confidence = process_predictions(predictions)

        # Generate a unique user session ID if not set
        if 'user_id' not in session:
            session['user_id'] = str(os.urandom(16))

        # Save classification to database
        new_entry = History(
            user_id=session['user_id'],
            image_name=uploaded_image.filename,
            predicted_label=predicted_label,
            confidence=float(confidence)
        )
        db.session.add(new_entry)
        db.session.commit()

        return render_template('result.html', result=f'Predicted: {predicted_label}, Confidence: {confidence:.2f}')
    else:
        return redirect(url_for('index'))

# View User's Classification History
@app.route('/history')
def history():
    if 'user_id' in session:
        user_history = History.query.filter_by(user_id=session['user_id']).order_by(History.timestamp.desc()).all()
        return render_template('history.html', history=user_history)
    else:
        return "No history found!"

# View Analytics Page
@app.route('/analytics')
def analytics():
    total_predictions = History.query.count()
    most_common = db.session.query(History.predicted_label, db.func.count(History.predicted_label))\
        .group_by(History.predicted_label).order_by(db.func.count(History.predicted_label).desc()).limit(5).all()

    return render_template('analytics.html', total=total_predictions, most_common=most_common)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if not exists
    app.run(debug=True)
