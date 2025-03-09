import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights, vgg16, VGG16_Weights
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Define constants
VGG19_CHECKPOINT_PATH = r"models/niats_vgg19_wave_flipping.pth"
VGG16_CHECKPOINT_PATH = r"models/niats_vgg16_spiral_noaug.pth"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation for VGG19
vgg19_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Image transformation for VGG16
vgg16_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load VGG19 model
def load_vgg19_model():
    model = vgg19(weights=VGG19_Weights.DEFAULT)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES)
    )
    
    if os.path.exists(VGG19_CHECKPOINT_PATH):
        checkpoint = torch.load(VGG19_CHECKPOINT_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        else:
            model.load_state_dict(checkpoint.state_dict(), strict=False)
        print("VGG19 checkpoint loaded successfully!")
    else:
        print("VGG19 checkpoint not found. Using an untrained model.")
    
    model.to(device)
    model.eval()
    return model

# Function to load VGG16 model
def load_vgg16_model():
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, NUM_CLASSES)
    )
    
    if os.path.exists(VGG16_CHECKPOINT_PATH):
        checkpoint = torch.load(VGG16_CHECKPOINT_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        else:
            model.load_state_dict(checkpoint.state_dict(), strict=False)
        print("VGG16 checkpoint loaded successfully!")
    else:
        print("VGG16 checkpoint not found. Using an untrained model.")
    
    model.to(device)
    model.eval()
    return model

# Load models at startup
vgg19_model = load_vgg19_model()
vgg16_model = load_vgg16_model()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "VGG Classification API is running with two models"}

@app.post("/predict/vgg19/")
async def predict_vgg19(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = vgg19_transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = vgg19_model(image)
        predicted_class = torch.argmax(output, dim=1).item()

        return JSONResponse(content={"prediction": predicted_class, "model": "vgg19"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict/vgg16/")
async def predict_vgg16(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = vgg16_transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = vgg16_model(image)
        predicted_class = torch.argmax(output, dim=1).item()

        return JSONResponse(content={"prediction": predicted_class, "model": "vgg16"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/predict/randomForest/")
async def predict_random_forest(file: str = Form(...)):
    # Load dataset
    print("Hello")


    parkinsons_data = pd.read_csv('models/voice.csv')
    
    X = parkinsons_data.drop(columns=['status'], axis=1)
    Y = parkinsons_data['status']
    
    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    
    # Convert input string to numpy array
    input_data = np.array([float(i) for i in file.split(",")])
    input_data_reshaped = input_data.reshape(1, -1)
    
    # Standardize input
    std_data = scaler.transform(input_data_reshaped)
    
    # Make prediction
    prediction = rf_model.predict(std_data)
    result = "Person has Parkinson's disease" if prediction[0] == 1 else "Person is healthy"
    
    return JSONResponse(content={"prediction": result, "model": "randomForest"})
# Run with: uvicorn main:app --reload