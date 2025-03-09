# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision.models import vgg19, VGG19_Weights, vgg16, VGG16_Weights
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# import os

# # Define constants

# CHECKPOINT_PATH = "niats_vgg19_wave_flipping.pth"
# # CHECKPOINT_PATH = "niats_vgg16_spiral_noaug.pth"
# IMAGE_SIZE = (224, 224)
# NUM_CLASSES = 2

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load VGG19 model with pre-trained weights
# model = vgg19(weights=VGG19_Weights.DEFAULT )

# # Load VGG16 model with pre-trained weights
# # model = vgg16(weights=VGG16_Weights.DEFAULT )

# # Modify classifier for binary classification
# num_features = model.classifier[6].in_features
# model.classifier[6] = nn.Sequential(
#     nn.Dropout(0.5),
#     nn.Linear(num_features, NUM_CLASSES)
# )

# # Load checkpoint if available
# if os.path.exists(CHECKPOINT_PATH):
#     checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
#     if isinstance(checkpoint, dict):
#         model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
#     else:
#         model.load_state_dict(checkpoint.state_dict(), strict=False)
#     print("Checkpoint loaded successfully!")
# else:
#     print("Checkpoint not found. Using an untrained model.")

# # Set model to evaluation mode
# model.to(device)
# model.eval()

# # Image transformation pipeline
# # For VGG-19
# transform = transforms.Compose([
#     transforms.Resize(IMAGE_SIZE),
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# # For VGG-16-No aug
# # transform = transforms.Compose([
# #     transforms.Resize((224,224)),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])


# # Initialize FastAPI app
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# async def root():
#     return {"message": "VGG-19 Classification API is running"}

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and process the image
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         image = transform(image).unsqueeze(0).to(device)

#         # Get prediction
#         with torch.no_grad():
#             output = model(image)
#         predicted_class = torch.argmax(output, dim=1).item()

#         return JSONResponse(content={"prediction": predicted_class})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Run with: uvicorn main:app --reload




import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights, vgg16, VGG16_Weights
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import logging
from typing import Dict, Any, Union, Literal
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    VGG19_CHECKPOINT_PATH = r"models/niats_vgg19_wave_flipping.pth"
    VGG16_CHECKPOINT_PATH = r"models/niats_vgg16_spiral_noaug.pth"
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalization parameters for ImageNet
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

config = Config()

# Create transform pipelines
@lru_cache(maxsize=2)
def get_transform(model_type: Literal["vgg19", "vgg16"]) -> transforms.Compose:
    """Creates and caches the transformation pipeline based on model type"""
    if model_type == "vgg19":
        # VGG19 uses data augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224),  # Changed from RandomCrop to CenterCrop for inference
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])
    
    else:  # vgg16
        # VGG16 doesn't use augmentation according to the comment in original code
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
        ])

# Model initialization
@lru_cache(maxsize=2)
def load_model(model_type: Literal["vgg19", "vgg16"]) -> nn.Module:
    """Loads and caches the specified model type"""
    logger.info(f"Loading {model_type} model on {config.DEVICE}")
    
    # Select model architecture and checkpoint path based on model_type
    if model_type == "vgg19":
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        checkpoint_path = config.VGG19_CHECKPOINT_PATH
    else:  # vgg16
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        checkpoint_path = config.VGG16_CHECKPOINT_PATH
    
    # Modify classifier for binary classification
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, config.NUM_CLASSES)
    
    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
            else:
                model.load_state_dict(checkpoint.state_dict(), strict=False)
                
            logger.info(f"Checkpoint loaded successfully for {model_type}!")
        except Exception as e:
            logger.error(f"Error loading checkpoint for {model_type}: {str(e)}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Using pre-trained weights only.")
    
    # Set model to evaluation mode and move to device
    model = model.to(config.DEVICE)
    model.eval()
    
    # Apply torch script optimization
    try:
        model = torch.jit.script(model)
        logger.info(f"{model_type} model optimized with TorchScript")
    except Exception as e:
        logger.warning(f"Could not apply TorchScript optimization to {model_type}: {str(e)}")
    
    return model

async def process_image(file: UploadFile, model_type: Literal["vgg19", "vgg16"]) -> Dict[str, Any]:
    """Common image processing and prediction function for both model types"""
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load the model and transform (cached after first call)
        model = load_model(model_type)
        transform = get_transform(model_type)
        
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

        # Get prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities, dim=0).item()
            confidence = probabilities[predicted_class].item()
        
        return {
            "model": model_type,
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                f"class_{i}": round(float(prob) * 100, 2) 
                for i, prob in enumerate(probabilities.cpu().numpy())
            }
        }
    
    except Exception as e:
        logger.error(f"Prediction error with {model_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Dual VGG Classification API",
    description="API for image classification using fine-tuned VGG19 and VGG16 models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint to confirm API is running"""
    return {
        "message": "Dual VGG Classification API is running",
        "device": str(config.DEVICE),
        "available_models": ["vgg19", "vgg16"],
        "endpoints": {
            "vgg19_prediction": "/predict/vgg19/",
            "vgg16_prediction": "/predict/vgg16/"
        }
    }

@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict/vgg19/")
async def predict_vgg19(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict using the VGG19 model with wave flipping weights
    """
    return await process_image(file, "vgg19")

@app.post("/predict/vgg16/")
async def predict_vgg16(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict using the VGG16 model with spiral no augmentation weights
    """
    return await process_image(file, "vgg16")

@app.post("/predict/")
async def predict_default(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Default prediction endpoint using VGG19 (for backward compatibility)
    """
    return await process_image(file, "vgg19")

if __name__ == "__main__":
    # This block only executes when running the file directly, not with uvicorn
    import uvicorn
    logger.info("Starting Dual Model API server")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)