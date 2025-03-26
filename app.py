from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
import os
import torch
import torch.nn as nn
from PIL import Image
import logging
import cv2
import numpy as np
import uuid
import firebase_admin
from firebase_admin import credentials, auth
import json
import gc
import boto3
from botocore.client import Config

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Define transforms
class transforms:
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img
    class Resize:
        def __init__(self, size):
            self.size = size
        def __call__(self, img):
            return img.resize(self.size, Image.BILINEAR)
    class Grayscale:
        def __init__(self, num_output_channels):
            self.num_output_channels = num_output_channels
        def __call__(self, img):
            return img.convert('L').convert('RGB') if self.num_output_channels == 3 else img.convert('L')
    class ToTensor:
        def __call__(self, img):
            return torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0

# Initialize Firebase
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
if not firebase_creds_json:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not set")
cred_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# R2 Configuration
R2_ACCOUNT_ID = 'your-account-id'  # e.g., d4f5e6...
R2_ACCESS_KEY = 'your-access-key'  # From API token or R2 settings
R2_SECRET_KEY = 'your-secret-key'  # From API token or R2 settings
R2_BUCKET = 'diagnostic-models'
R2_ENDPOINT = f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com'

s3_client = boto3.client(
    's3',
    aws_access_key_id=R2_ACCESS_KEY,
    aws_secret_access_key=R2_SECRET_KEY,
    endpoint_url=R2_ENDPOINT,
    config=Config(signature_version='s3v4')
)

# Manual ResNet50
class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.Softmax(dim=1)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# Model paths in R2
xray_models = {
    'pneumonia': 'Pneumonia.pth',
    'tuberculosis': 'Tuberculosis_model.pth',
}

# Class names
class_names = {
    'pneumonia': ['Normal', 'Pneumonia'],
    'tuberculosis': ['Normal', 'Tuberculosis'],
}

# Load ResNet model from R2
def load_resnet_model(model_key, num_classes):
    model = ResNet50(num_classes)
    local_path = f"/tmp/{model_key}"
    try:
        s3_client.download_file(R2_BUCKET, model_key, local_path)
        logger.debug(f"Downloaded {model_key} from R2 to {local_path}")
        
        state_dict = torch.load(local_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded ResNet model from R2: {model_key}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model from R2 {model_key}: {str(e)}")
        return None
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)  # Clean up

# Simplified Grad-CAM
def generate_gradcam(model, image_tensor, disease, predicted_class):
    with torch.no_grad():
        features = None
        def hook(module, input, output):
            nonlocal features
            features = output
        handle = model.layer4.register_forward_hook(hook)
        output = model(image_tensor)
        handle.remove()

        target_class = class_names[disease].index(predicted_class)
        score = output[0, target_class]
        grads = torch.autograd.grad(score, features, retain_graph=False)[0]
        pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
        heatmap = torch.mean(features * pooled_grads, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10

        img_np = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        if img_np.max() <= 1:
            img_np *= 255
        img_np = img_np.astype(np.uint8)

        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

        gradcam_filename = f"static/gradcam/heatmap_{uuid.uuid4().hex}.png"
        os.makedirs("static/gradcam", exist_ok=True)
        cv2.imwrite(gradcam_filename, superimposed_img)

        del features, grads, pooled_grads, heatmap
        gc.collect()
        return gradcam_filename

# Prediction
def predict_resnet_image(model, image, transform, disease):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probability = output[0][predicted].item()
    predicted_class = class_names[disease][predicted.item()]
    return predicted_class, probability, image_tensor

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/account')
def account():
    return render_template('account.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('account'))
    return render_template('dashboard.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/login', methods=['POST'])
def login():
    token = request.json.get('idToken')
    try:
        decoded_token = auth.verify_id_token(token)
        session['logged_in'] = True
        session['uid'] = decoded_token['uid']
        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 401

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image = request.files['image']
        scan_type = request.form['type']
        disease = request.form['disease']
        logger.debug(f"Analyze request: type={scan_type}, disease={disease}")

        if scan_type != 'xray':
            return jsonify({'message': "<p><strong>Error:</strong> Only X-ray scans are supported.</p>"})

        if disease not in ['pneumonia', 'tuberculosis']:
            return jsonify({'message': "<p><strong>Error:</strong> Only pneumonia and tuberculosis analysis are supported.</p>"})

        img = Image.open(image)
        model_key = xray_models.get(disease)
        model = load_resnet_model(model_key, 2)
        if model:
            predicted_class, probability, image_tensor = predict_resnet_image(model, img, transform, disease)
            gradcam_path = generate_gradcam(model, image_tensor, disease, predicted_class)
            message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
            del model, image_tensor
            gc.collect()
            return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
        else:
            message = f"<p><strong>Status:</strong> {disease.replace('_', ' ').title()} model failed to load.</p>"
            return jsonify({'message': message})

    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        return jsonify({'message': f"<p><strong>Error:</strong> Server error: {str(e)}</p>"})

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
