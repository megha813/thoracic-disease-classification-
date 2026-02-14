import os
import sys
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
# from src.routers.api import 
# from ..utils.load_image import load_images_from_dir
import traceback


def predict_single_image(image_path, model, transform, labels, device):#, filename="prediction.csv"):
    try:
        # Load and preprocess the image
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        # image = torch.from_numpy(image).float()
        image = image.to(device)

        # Perform prediction
        model.eval()
        with torch.no_grad():
            output = model(image)
            output=torch.sigmoid(output)
        
        predictions = (output > 0.5).int()
        predictions_list = predictions.squeeze().cpu().numpy().tolist()

        # Map predictions to their labels
        prediction_results = dict(zip(labels, output[0].detach().cpu().numpy()))
        print(f"Prediction for {image_path}: {prediction_results}")

        # Save predictions to CSV
        # with open(filename, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     # Write header
        #     writer.writerow(['image_path'] + labels)
        #     # Write data
        #     writer.writerow([image_path] + predictions_list)

        # print(f"Prediction saved to {filename}")
        return prediction_results
    except Exception as e:
        # Enhanced error logging
        error_msg = f"""
        ERROR DETAILS:
        - Type: {type(e)}
        - Message: {str(e)}
        - Traceback: {traceback.format_exc()}
        """
        print(error_msg)  # This will appear in your FastAPI server logs
        return None