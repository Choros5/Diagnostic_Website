from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
import os
import torch
import torch.nn as nn
from PIL import Image
import logging
import cv2
import numpy as np
import uuid
import json
import firebase_admin
from firebase_admin import credentials, auth
import boto3
from botocore.exceptions import ClientError

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback-secret-key")

# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Device
device = torch.device("cpu")  # Render free tier, no GPU
logger.info(f"Using device: {device}")

# Simplified transforms
class Transforms:
    @staticmethod
    def resize(img): return img.resize((224, 224), Image.BILINEAR)
    @staticmethod
    def grayscale(img): return img.convert('L').convert('RGB')
    @staticmethod
    def to_tensor(img): return torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
    @staticmethod
    def apply(img): return Transforms.to_tensor(Transforms.grayscale(Transforms.resize(img)))

# Firebase setup
firebase_creds = os.environ.get("FIREBASE_CREDENTIALS")
if not firebase_creds:
    raise ValueError("FIREBASE_CREDENTIALS not set")
cred = credentials.Certificate(json.loads(firebase_creds))
firebase_admin.initialize_app(cred)

# R2 setup
R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
R2_BUCKET = "diagnostic-models"
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("R2_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("R2_SECRET_KEY"),
    endpoint_url=R2_ENDPOINT
)

# Check R2 at startup (non-blocking)
try:
    s3_client.head_bucket(Bucket=R2_BUCKET)
    logger.info(f"Bucket {R2_BUCKET} exists and is accessible")
except ClientError as e:
    logger.warning(f"Bucket check failed for {R2_BUCKET}: {e} - Proceeding anyway")

# Simplified ResNet50
class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(*[Bottleneck(64, 64) for _ in range(3)])
        self.layer2 = nn.Sequential(Bottleneck(256, 128, stride=2), *[Bottleneck(512, 128) for _ in range(3)])
        self.layer3 = nn.Sequential(Bottleneck(512, 256, stride=2), *[Bottleneck(1024, 256) for _ in range(5)])
        self.layer4 = nn.Sequential(Bottleneck(1024, 512, stride=2), *[Bottleneck(2048, 512) for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
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
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        ) if stride != 1 or in_channels != out_channels * 4 else None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

# Models and classes
xray_models = {"pneumonia": "Pneumonia.pth", "tuberculosis": "Tuberculosis_model.pth"}
class_names = {"pneumonia": ["Normal", "Pneumonia"], "tuberculosis": ["Normal", "Tuberculosis"]}

def load_model(model_key):
    local_path = f"/tmp/{model_key}"
    try:
        s3_client.head_object(Bucket=R2_BUCKET, Key=model_key)
        logger.debug(f"Confirmed {model_key} exists in {R2_BUCKET}")
        s3_client.download_file(R2_BUCKET, model_key, local_path)
        logger.debug(f"Downloaded {model_key} from R2")
        model = ResNet50().to(device)
        model.load_state_dict(torch.load(local_path, map_location=device))
        model.eval()
        os.remove(local_path)
        return model
    except ClientError as e:
        logger.error(f"R2 error for {model_key}: {e}")
        return None

def predict(model, image, disease):
    tensor = Transforms.apply(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        prob = output[0][pred].item()
    return class_names[disease][pred.item()], prob, tensor

def generate_gradcam(model, tensor, disease, pred_class):
    features = None
    def hook(module, input, output): nonlocal features; features = output
    handle = model.layer4.register_forward_hook(hook)
    output = model(tensor)
    handle.remove()
    score = output[0, class_names[disease].index(pred_class)]
    grads = torch.autograd.grad(score, features)[0]
    heatmap = torch.mean(features * torch.mean(grads, dim=[2, 3], keepdim=True), dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    img_np = (tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    heatmap = cv2.resize(np.uint8(255 * heatmap), (224, 224))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    path = f"static/gradcam/heatmap_{uuid.uuid4().hex}.png"
    os.makedirs("static/gradcam", exist_ok=True)
    cv2.imwrite(path, superimposed)
    return path

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
        decoded = auth.verify_id_token(token)
        session['logged_in'] = True
        session['uid'] = decoded['uid']
        return jsonify({'success': True, 'redirect': '/dashboard'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 401

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        image = request.files['image']
        scan_type = request.form['type']
        disease = request.form['disease']
        logger.debug(f"Analyze: type={scan_type}, disease={disease}")

        if scan_type != 'xray' or disease not in xray_models:
            return jsonify({'message': "<p><strong>Error:</strong> Invalid scan type or disease</p>"})

        img = Image.open(image)
        model = load_model(xray_models[disease])
        if not model:
            return jsonify({'message': f"<p><strong>Error:</strong> Failed to load {disease} model</p>"})

        pred_class, prob, tensor = predict(model, img, disease)
        gradcam_path = generate_gradcam(model, tensor, disease, pred_class)
        return jsonify({
            'message': f"<p><strong>Predicted:</strong> {pred_class}</p><p><strong>Confidence:</strong> {prob:.4f}</p>",
            'gradcam_url': f"/{gradcam_path}"
        })
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return jsonify({'message': f"<p><strong>Error:</strong> {e}</p>"})

@app.route('/health')
def health():
    return "OK", 200

@app.route('/debug/r2', methods=['GET'])
def debug_r2():
    try:
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        return jsonify({'buckets': buckets, 'endpoint': R2_ENDPOINT})
    except ClientError as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
