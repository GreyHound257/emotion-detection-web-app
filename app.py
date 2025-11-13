from flask import Flask, request, render_template
import torch
import torch.nn as nn
from model import EmotionCNN
import os
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy load model (load on first request)
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
            transforms.Resize((48, 48)),  # Reduced for faster inference
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    return model, transform

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('index.html', message='No file part')
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', message='No selected file')
            if file:
                # Save image
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                # Load and process
                model, transform = load_model()
                img = Image.open(filename).convert('L')
                img_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    _, predicted = torch.max(output, 1)
                    emotion = classes[predicted.item()]
                return render_template('result.html', emotion=emotion, image_path=filename)
        except Exception as e:
            return render_template('index.html', message=f'Error processing image: {str(e)}')
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)