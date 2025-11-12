from flask import Flask, render_template, request, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
import sqlite3
from model import EmotionCNN  # Import model class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = EmotionCNN()
model.load_state_dict(torch.load('trained_models/groovy_emotion_detector_v1.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
            elif 'image' in request.form:  # From webcam (base64)
                import base64
                img_data = request.form['image'].split(',')[1]
                img = Image.open(base64.b64decode(img_data))
                filename = 'captured.png'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img.save(filepath)
            else:
                return "No image provided", 400

            # Predict
            img = Image.open(filepath)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img)
                pred = classes[torch.argmax(output).item()]

            # Store in DB (using dummy user_id=1)
            conn = sqlite3.connect('database.sqlite3')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO predictions (user_id, image_path, predicted_emotion) VALUES (?, ?, ?)",
                           (1, filepath, pred))
            conn.commit()
            conn.close()

            return render_template('result.html', image_path=filepath, emotion=pred)
        except Exception as e:
            return f"Error: {str(e)}", 500

    return render_template('index.html')

@app.route('/predictions')
def predictions():
    preds = query_database.get_predictions(user_id=1)
    return render_template('result.html', predictions=preds)  # Adjust result.html to handle list if needed

if __name__ == '__main__':
    app.run(debug=True)