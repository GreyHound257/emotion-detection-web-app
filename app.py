# app.py
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
    transforms.Resize((54, 54)),  # Match original training size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

from query_database import get_recent_predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    recent_predictions = get_recent_predictions()  # Query database
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part', recent_predictions=recent_predictions)
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file', recent_predictions=recent_predictions)
        if file:
            img = Image.open(file.stream).convert('L')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][predicted.item()]
            # Save to database (example)
            save_prediction(emotion)  # Implement in query_database.py
            recent_predictions = get_recent_predictions()  # Refresh
            return render_template('result.html', emotion=emotion, recent_predictions=recent_predictions)
    return render_template('index.html', recent_predictions=recent_predictions)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))