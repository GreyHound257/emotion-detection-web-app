from flask import Flask, request, render_template
import torch
import torch.nn as nn
from model import EmotionCNN
import os
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Load model
model = EmotionCNN()
model.load_state_dict(torch.load('trained_models/groovy_emotion_detector_v1.pth'))
model.eval()

# Define transformation for input image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((54, 54)),  # Adjusted to match 9216 input size (trial-based)
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            img = Image.open(file.stream).convert('L')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][predicted.item()]
            return render_template('result.html', emotion=emotion)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))