from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.responses import JSONResponse , FileResponse
import tempfile
import torchvision
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from ..model.prediction import predict_single_image
from ..configuration.config import weights
from src.utils import CommonUtils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MedViT')))

from MedViT import MedViT_large as large
from pathlib import Path
from src.utils.dicom_utils import dicom_to_image
from PIL import Image
from src.configuration.config import outputDir
from src.configuration.config import DICOM_TEMP_PATH
# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

router = APIRouter()

@router.post("/convert-dicom/")
async def convert_dicom(file: UploadFile = File(...), format: str = "png"):
    if format not in ["jpg", "png"]:
        return {"error": "Invalid format. Use 'jpg' or 'png'."}
    
    with tempfile.NamedTemporaryFile(delete=False,suffix=".dcm") as temp_dicom:
        temp_dicom.write(await file.read())
        temp_dicom_path=temp_dicom.name
    output_path = temp_dicom_path.replace(".dcm", f".{format}")

    try :
        dicom_to_image(temp_dicom_path , output_path , format=format)
        return {"converted_image_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    # finally :
    #      if os.path.exists(temp_dicom_path):
    #         os.remove(temp_dicom_path)
# Directory to save uploaded images


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = large()
model.proj_head[0]
model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=18, bias=True)
checkpoint = torch.load(weights, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
model.to(device)

# Transformation setup
transform = transforms.Compose([
              transforms.Resize((224,224)),
              transforms.Lambda(lambda image: image.convert('RGB')),
              
              transforms.ToTensor(),
              transforms.Normalize(mean=[.5], std=[.5])
              ])

# Labels setup
LABELS = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']


@router.post("/predict-from-dicom/")
async def predict_from_dicom(file: UploadFile = File(...),format:str="png"):

    """
    Endpoint to upload an image and return predictions.
    """
    try:
        dicom_conversion_response=await convert_dicom(file,format=format)

        if "converted_image_path" not in dicom_conversion_response:
            raise HTTPException(status_code=500,detail="DICOM conversion failed")
        converted_image_path=dicom_conversion_response["converted_image_path"]
        image = Image.open(converted_image_path)
        if image.mode == "I;16":
            image = image.convert("I")  # Convert to 32-bit integer mode
            image = image.point(lambda p: p * (255.0 / 65535.0))  # Normalize pixel values
            image = image.convert("L")  #
        print(f"image mode:{image.mode}")
        image = transform(image).unsqueeze(0)
        print(f"size of image : {image.size()}")
        # image = torch.from_numpy(image).float()
        image = image.to(device)
        print(f"mode of image : {image.mode}")

        

        # Call the prediction function with required arguments
        predictions = predict_single_image(
            image_path=converted_image_path,
            model=model,
            transform=transform,
            labels=LABELS,
            device=device
        )
        if os.path.exists(converted_image_path):
            os.remove(converted_image_path)
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
                return obj.tolist()
            elif isinstance(obj, np.generic):  # Convert float32, int64, etc.
                return obj.item()
            return obj 
        json_filename = "predictions.json"
        json_filepath = os.path.join(outputDir, json_filename) #checkthis
        
        with open(json_filepath, "w", encoding="utf-8") as json_file:
            json.dump(predictions, json_file, indent=4, default=convert_numpy)


        temp_dir_return = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
        shutil.move(json_filepath,temp_dir_return)
        data = {
            "file_id": os.path.basename(temp_dir_return)
        }

        return JSONResponse(content=data)
        # print(f"prediction:{predictions}")

        # return JSONResponse(content=f"Image Path: {converted_image_path}\n Predictions: {predictions}")#, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    


@router.get("/json/{temp_dir}")
async def get_json_object(temp_dir: str):
    try:
        path = os.path.join(DICOM_TEMP_PATH,temp_dir, "predictions.json")
        dir_path = os.path.join(DICOM_TEMP_PATH, temp_dir)
        if os.path.exists(path):
            file_content = ""
            with open(path) as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise e
    finally:
        # Whether we had an error or not, it's important to clean up the temp directory
        os.remove(path)
        CommonUtils.delete_if_empty(dir_path)
UPLOAD_DIR = Path("uploaded_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/predict_from_png/")
def predict(file: UploadFile = File(...)):              ### async removed for  gradio app to run faster 
# async def predict(file: UploadFile = File(...)):
    """
    Endpoint to upload an image and return predictions.
    """
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        

        # Call the prediction function with required arguments
        predictions = predict_single_image(
            image_path=str(file_path),
            model=model,
            transform=transform,
            labels=LABELS,
            device=device
        )
        # print(f"prediction:{predictions}")

        # return JSONResponse(content=f"Image Path: {str(file_path.resolve())}\n Predictions: {predictions}")#, media_type="text/plain")
        clean_predictions = {k: float(v) for k, v in predictions.items()}
        return clean_predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    

# from fastapi import APIRouter, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import tempfile
# import os
# import json
# import shutil
# import numpy as np
# from PIL import Image
# import torch


@router.post("/predictdisease/")
async def segment_dicom(file: UploadFile = File(...), format: str = "png"):
    """
    Upload a DICOM file, convert it to an image, and perform prediction.
    """
    # Validate format
    if format not in ["jpg", "png"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'jpg' or 'png'.")

    # Convert DICOM to image
    try:
        dicom_conversion_response = await convert_dicom(file, format=format)
        
        if "converted_image_path" not in dicom_conversion_response:
            raise HTTPException(status_code=500, detail="DICOM conversion failed")
        
        converted_image_path = dicom_conversion_response["converted_image_path"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during DICOM conversion: {str(e)}")

    # Load and preprocess the image
    try:
        image = Image.open(converted_image_path)
        if image.mode == "I;16":
            image = image.convert("I")  # Convert to 32-bit integer mode
            image = image.point(lambda p: p * (255.0 / 65535.0))  # Normalize pixel values
            image = image.convert("L")

        print(f"Image mode: {image.mode}")
        image = transform(image).unsqueeze(0)
        print(f"Size of image: {image.size()}")
        
        image = image.to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    # Perform prediction
    try:
        predictions = predict_single_image(
            image_path=converted_image_path,
            model=model,
            transform=transform,
            labels=LABELS,
            device=device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    # Clean up the converted image file
    if os.path.exists(converted_image_path):
        os.remove(converted_image_path)

    # Convert NumPy data to JSON serializable format
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj

    try:
        json_filename = "predictions.json"
        json_filepath = os.path.join(outputDir, json_filename)
        
        with open(json_filepath, "w", encoding="utf-8") as json_file:
            json.dump(predictions, json_file, indent=4, default=convert_numpy)

        # Move the JSON file to a temporary directory
        temp_dir_return = tempfile.mkdtemp(dir=DICOM_TEMP_PATH)
        shutil.move(json_filepath, temp_dir_return)

        return JSONResponse(content={"file_id": os.path.basename(temp_dir_return)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving predictions: {str(e)}")
    



@router.get("/json/{temp_dir}")
async def get_json_object(temp_dir: str):
    try:
        path = os.path.join(DICOM_TEMP_PATH,temp_dir, "predictions.json")
        dir_path = os.path.join(DICOM_TEMP_PATH, temp_dir)
        if os.path.exists(path):
            file_content = ""
            with open(path) as f:
                data = json.load(f)
            return data
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise e
    finally:
        # Whether we had an error or not, it's important to clean up the temp directory
        os.remove(path)
        CommonUtils.delete_if_empty(dir_path)



