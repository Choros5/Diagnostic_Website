from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging
import cv2
import numpy as np
import uuid
import firebase_admin
from firebase_admin import credentials, auth
import json


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a secure key

# Initialize Firebase using environment variable
firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS")
if not firebase_creds_json:
    raise ValueError("FIREBASE_CREDENTIALS environment variable not set")
cred_dict = json.loads(firebase_creds_json)
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

kidney_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

breast_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Load ResNet models
def load_resnet_model(model_path, num_classes, resnet_type='resnet50'):
    if resnet_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif resnet_type == 'resnet18':
        model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes),
        nn.Softmax(dim=1)
    )
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            if torch.cuda.device_count() > 1 and next(iter(state_dict)).startswith('module'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            logger.info(f"Loaded ResNet model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load ResNet model from {model_path}: {str(e)}")
            return None
    logger.warning(f"Model file not found: {model_path}")
    return None

# Define Kidney Cancer model
class NNUnetEncoder2D(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super(NNUnetEncoder2D, self).__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_filters),
                nn.ReLU(),
                nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_filters),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_filters * 2),
                nn.ReLU(),
                nn.Conv2d(base_filters * 2, base_filters * 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_filters * 2),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_filters * 4),
                nn.ReLU(),
                nn.Conv2d(base_filters * 4, base_filters * 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_filters * 4),
                nn.ReLU()
            )
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.conv_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features

class KidneyClassifier2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super(KidneyClassifier2D, self).__init__()
        self.encoder = NNUnetEncoder2D(in_channels=in_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        x = features[-1]
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_kidney_model(model_path):
    model = KidneyClassifier2D(in_channels=1, num_classes=4)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            if torch.cuda.device_count() > 1 and next(iter(state_dict)).startswith('module'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            logger.info(f"Loaded Kidney model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Kidney model from {model_path}: {str(e)}")
            return None
    logger.warning(f"Kidney model file not found: {model_path}")
    return None

# Load models
xray_models = {
    'pneumonia': load_resnet_model("models/Pneumonia.pth", 2, 'resnet50'),
    'tuberculosis': load_resnet_model("models/Tuberculosis_model.pth", 2, 'resnet50'),
    'lung_cancer': load_resnet_model("models/Lung_Cancer_model.pth", 4, 'resnet50')  # X-ray model
}

mri_models = {
    'brain_cancer': None  # Placeholder; provide model later
}

ct_models = {
    'kidney_cancer': load_kidney_model("models/Kidney_model.pth"),
    'breast_cancer': load_resnet_model("models/Lung_Cancer_Biopsy.pth", 4, 'resnet18')
}

biopsy_models = {
    'cervical_cancer': load_resnet_model("models/Cervical_Cancer_2.pth", 4, 'resnet50'),
    'lung_cancer': load_resnet_model("models/LungCancer_model.pth.pth", 3, 'resnet18')  # Biopsy model
}

endoscopy_models = {
    'esophageal_cancer': load_resnet_model("models/Esophageal_model.pth", 4, 'resnet50')
}

# Log loaded models for debugging
logger.debug(f"Loaded xray_models: { {k: 'Loaded' if v is not None else 'Not Loaded' for k, v in xray_models.items()} }")
logger.debug(f"Loaded mri_models: { {k: 'Loaded' if v is not None else 'Not Loaded' for k, v in mri_models.items()} }")
logger.debug(f"Loaded ct_models: { {k: 'Loaded' if v is not None else 'Not Loaded' for k, v in ct_models.items()} }")
logger.debug(f"Loaded biopsy_models: { {k: 'Loaded' if v is not None else 'Not Loaded' for k, v in biopsy_models.items()} }")
logger.debug(f"Loaded endoscopy_models: { {k: 'Loaded' if v is not None else 'Not Loaded' for k, v in endoscopy_models.items()} }")

# Class names
class_names = {
    'pneumonia': ['Normal', 'Pneumonia'],
    'tuberculosis': ['Normal', 'Tuberculosis'],
    'lung_cancer': {  # Nested dictionary to differentiate X-ray and Biopsy
        'xray': ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma'],
        'biopsy': ['adenocarcinoma', 'benign', 'carcinoma']
    },
    'brain_cancer': ['Negative', 'Positive'],
    'kidney_cancer': ['Cyst', 'Normal', 'Stone', 'Tumor'],
    'breast_cancer': ['Stage 0', 'Stage I', 'Stage II', 'Stage III'],
    'cervical_cancer': ['High squamous intra-epithelial lesion', 'Low squamous intra-epithelial lesion',
                        'Negative Intraepithelial malignancy', 'Squamous cell carcinoma'],
    'esophageal_cancer': ['Dysplasia_Cancer', 'Gastric_Metaplasia', 'Intestinal_Metaplasia', 'Squamous_Epithelium']
}

# Grad-CAM for ResNet Models
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

    def save_gradient(self, grad):
        self.gradients = grad

    def hook_features(self, module, input, output):
        self.features = output

    def __call__(self, x, target_class):
        self.model.zero_grad()
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self.hook_features)
                module.register_backward_hook(lambda m, gi, go: self.save_gradient(go[0]))
        output = self.model(x)
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot.to(device))
        return output

def generate_gradcam(model, image_tensor, preprocess, disease, predicted_class, target_layer, scan_type=None):
    grad_cam = GradCAM(model, target_layer)
    if disease == 'lung_cancer':
        target_class = class_names[disease][scan_type].index(predicted_class)
    else:
        target_class = class_names[disease].index(predicted_class)
    output = grad_cam(image_tensor, target_class)

    gradients = grad_cam.gradients
    pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
    features = grad_cam.features

    heatmap = features * pooled_gradients
    heatmap = torch.mean(heatmap, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    img_np = image_tensor.squeeze().cpu().numpy()
    if len(img_np.shape) == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    else:
        logger.error(f"Unexpected image tensor shape for ResNet Grad-CAM: {img_np.shape}")
        raise ValueError(f"Unexpected image tensor shape: {img_np.shape}")

    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if img_np.max() <= 1:
        img_np = img_np * 255
    img_np = img_np.astype(np.uint8)
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    gradcam_filename = f"static/gradcam/heatmap_{uuid.uuid4().hex}.png"
    os.makedirs("static/gradcam", exist_ok=True)
    cv2.imwrite(gradcam_filename, superimposed_img)
    return gradcam_filename

# Grad-CAM for Kidney Model
def generate_kidney_gradcam(model, image_tensor, predicted_class):
    gradients = None
    features = None

    def save_gradients_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def save_features_hook(module, input, output):
        nonlocal features
        features = output

    target_layer = model.encoder.conv_blocks[2]
    target_layer.register_forward_hook(save_features_hook)
    target_layer.register_backward_hook(save_gradients_hook)

    model.zero_grad()
    output = model(image_tensor)
    target_class = class_names['kidney_cancer'].index(predicted_class)
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot.to(device))

    pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
    heatmap = features * pooled_gradients
    heatmap = torch.mean(heatmap, dim=1).squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    img_np = image_tensor.squeeze().cpu().numpy()
    if len(img_np.shape) == 2:
        img_np = np.repeat(img_np[..., np.newaxis], 3, axis=-1)
    else:
        logger.error(f"Unexpected kidney image tensor shape: {img_np.shape}")
        raise ValueError(f"Unexpected kidney image tensor shape: {img_np.shape}")

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    if img_np.max() <= 1:
        img_np = (img_np * 255).astype(np.uint8)

    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    gradcam_filename = f"static/gradcam/heatmap_{uuid.uuid4().hex}.png"
    os.makedirs("static/gradcam", exist_ok=True)
    cv2.imwrite(gradcam_filename, superimposed_img)
    return gradcam_filename

# Prediction functions
def predict_resnet_image(model, image, transform, disease, scan_type=None):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probability = output[0][predicted].item()
    if disease == 'lung_cancer':
        predicted_class = class_names[disease][scan_type][predicted.item()]
    else:
        predicted_class = class_names[disease][predicted.item()]
    return predicted_class, probability, image_tensor

def predict_kidney_image(model, image, preprocess):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names['kidney_cancer'][predicted.item()]
        confidence_score = confidence.item()
    return predicted_class, confidence_score, image_tensor

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

        img = Image.open(image)

        if scan_type == 'xray':
            model = xray_models.get(disease)
            if model:
                predicted_class, probability, image_tensor = predict_resnet_image(model, img, transform, disease, scan_type='xray' if disease == 'lung_cancer' else None)
                gradcam_path = generate_gradcam(model, image_tensor, transform, disease, predicted_class, 'layer4', scan_type='xray' if disease == 'lung_cancer' else None)
                message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
                return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
            else:
                message = f"<p><strong>Status:</strong> {disease.replace('_', ' ').title()} analysis is under development.</p>"

        elif scan_type == 'mri':
            model = mri_models.get(disease)
            if model:
                predicted_class, probability, image_tensor = predict_resnet_image(model, img, transform, disease)
                gradcam_path = generate_gradcam(model, image_tensor, transform, disease, predicted_class, 'layer4')
                message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
                return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
            else:
                message = f"<p><strong>Status:</strong> {disease.replace('_', ' ').title()} analysis is under development.</p>"

        elif scan_type == 'ct':
            model = ct_models.get(disease)
            if model:
                if disease == 'kidney_cancer':
                    predicted_class, probability, image_tensor = predict_kidney_image(model, img, kidney_preprocess)
                    gradcam_path = generate_kidney_gradcam(model, image_tensor, predicted_class)
                    message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
                    return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
                else:
                    transform_to_use = breast_preprocess if disease == 'breast_cancer' else transform
                    predicted_class, probability, image_tensor = predict_resnet_image(model, img, transform_to_use, disease)
                    target_layer = 'layer4'
                    gradcam_path = generate_gradcam(model, image_tensor, transform_to_use, disease, predicted_class, target_layer)
                    message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
                    return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
            else:
                message = f"<p><strong>Status:</strong> {disease.replace('_', ' ').title()} analysis is under development.</p>"

        elif scan_type == 'biopsy':
            model = biopsy_models.get(disease)
            if model:
                predicted_class, probability, image_tensor = predict_resnet_image(model, img, transform, disease, scan_type='biopsy' if disease == 'lung_cancer' else None)
                gradcam_path = generate_gradcam(model, image_tensor, transform, disease, predicted_class, 'layer4', scan_type='biopsy' if disease == 'lung_cancer' else None)
                message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
                return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
            else:
                message = f"<p><strong>Status:</strong> {disease.replace('_', ' ').title()} analysis is under development.</p>"

        elif scan_type == 'endoscopy':
            model = endoscopy_models.get(disease)
            if model:
                predicted_class, probability, image_tensor = predict_resnet_image(model, img, transform, disease)
                gradcam_path = generate_gradcam(model, image_tensor, transform, disease, predicted_class, 'layer4')
                message = f"<p><strong>Predicted Condition:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {probability:.4f}</p>"
                return jsonify({'message': message, 'gradcam_url': f"/{gradcam_path}"})
            else:
                message = f"<p><strong>Status:</strong> {disease.replace('_', ' ').title()} analysis is under development.</p>"

        else:
            message = "<p><strong>Error:</strong> Invalid scan type.</p>"

        return jsonify({'message': message})
    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        return jsonify({'message': f"<p><strong>Error:</strong> Server error: {str(e)}</p>"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 locally
    app.run(debug=True, host='0.0.0.0', port=port)
