from flask import Flask, request, render_template, session
import torch
import torch.nn as nn
from model import EmotionCNN
import os
from PIL import Image
import torchvision.transforms as transforms
from query_database import save_prediction, get_recent_predictions

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure key
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy load model
model = None
transform = None
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def load_model():
    global model, transform
    if model is None:
        model = EmotionCNN()
        model.load_state_dict(torch.load('trained_models/groovy_emotion_detector_v1.pth'))
        model.eval()
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return model, transform

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files or 'name' not in request.form:
                return render_template('index.html', message='No file or name provided')
            file = request.files['file']
            name = request.form['name']
            if not file.filename or not name:
                return render_template('index.html', message='No selected file or name')
            if file:
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                model, transform = load_model()
                img = Image.open(filename).convert('L')
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    _, predicted = torch.max(output, 1)
                    emotion = classes[predicted.item()]
                save_prediction(name, emotion, filename)
                return render_template('result.html', emotion=emotion, image_path=file.filename)
        except Exception as e:
            return render_template('index.html', message=f'Error processing image: {str(e)}')
    return render_template('index.html')

@app.route('/history')
def history():
    predictions = get_recent_predictions()
    return render_template('history.html', predictions=predictions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)