import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights, vgg16, VGG16_Weights
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

# Define constants

CHECKPOINT_PATH = "niats_vgg19_wave_flipping.pth"
# CHECKPOINT_PATH = "niats_vgg16_spiral_noaug.pth"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 model with pre-trained weights
model = vgg19(weights=VGG19_Weights.DEFAULT )

# Load VGG16 model with pre-trained weights
# model = vgg16(weights=VGG16_Weights.DEFAULT )

# Modify classifier for binary classification
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, NUM_CLASSES)
)

# Load checkpoint if available
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    else:
        model.load_state_dict(checkpoint.state_dict(), strict=False)
    print("Checkpoint loaded successfully!")
else:
    print("Checkpoint not found. Using an untrained model.")

# Set model to evaluation mode
model.to(device)
model.eval()

# Image transformation pipeline
# For VGG-19
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# For VGG-16-No aug
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


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
    return {"message": "VGG-19 Classification API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

        return JSONResponse(content={"prediction": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run with: uvicorn main:app --reload
