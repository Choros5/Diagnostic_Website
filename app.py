from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
import os
import torch
import torch.nn as nn
from torchvision import models
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

try:
    s3_client.head_bucket(Bucket=R2_BUCKET)
    logger.info(f"Bucket {R2_BUCKET} exists and is accessible")
except ClientError as e:
    logger.warning(f"Bucket check failed for {R2_BUCKET}: {e} - Proceeding anyway")

xray_models = {"pneumonia": "Pneumonia.pth", "tuberculosis": "Tuberculosis_model.pth"}
class_names = {"pneumonia": ["Normal", "Pneumonia"], "tuberculosis": ["Normal", "Tuberculosis"]}

def load_model(model_key):
    local_path = f"/tmp/{model_key}"
    try:
        s3_client.head_object(Bucket=R2_BUCKET, Key=model_key)
        logger.debug(f"Confirmed {model_key} exists in {R2_BUCKET}")
        s3_client.download_file(R2_BUCKET, model_key, local_path)
        logger.debug(f"Downloaded {model_key} from R2")
        
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_features, 2), nn.Softmax(dim=1))
        
        state_dict = torch.load(local_path, map_location=device)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        os.remove(local_path)
        logger.debug(f"Loaded and evaluated {model_key}")
        return model
    except Exception as e:
        logger.error(f"Model loading error for {model_key}: {e}")
        return None

def predict(model, image, disease):
    tensor = Transforms.apply(image).unsqueeze(0).to(device)
    logger.debug("Starting prediction")
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        prob = output[0][pred].item()
    logger.debug(f"Prediction completed: {class_names[disease][pred.item()]}")
    return class_names[disease][pred.item()], prob, tensor

def generate_gradcam(model, tensor, disease, pred_class):
    logger.debug("Starting Grad-CAM")
    model.eval()
    features = None
    gradients = None
    
    def save_features(module, input, output): nonlocal features; features = output
    def save_gradients(module, grad_in, grad_out): nonlocal gradients; gradients = grad_out[0]
    
    target_layer = model.layer4
    target_layer.register_forward_hook(save_features)
    target_layer.register_backward_hook(save_gradients)
    
    tensor.requires_grad = True
    output = model(tensor)
    score = output[0, class_names[disease].index(pred_class)]
    model.zero_grad()
    score.backward()
    
    weights = torch.mean(gradients, dim=[2, 3])[0]
    cam = torch.zeros(tensor.shape[2:], dtype=torch.float32).to(device)
    for i, w in enumerate(weights):
        cam += w * features[0, i]
    
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-10)
    logger.debug("Grad-CAM heatmap computed")
    
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    img_np = (tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    
    path = f"static/gradcam/heatmap_{uuid.uuid4().hex}.png"
    os.makedirs("static/gradcam", exist_ok=True)
    cv2.imwrite(path, superimposed)
    logger.debug(f"Grad-CAM saved to {path}")
    return path

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
