import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torchvision

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


from .prediction import predict_single_image
from ..configuration.config import image_path,weights
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MedViT')))

def main():
    # Load the pretrained model
    from MedViT import MedViT_large as large
    model = large()
    model.proj_head[0]
    model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=18, bias=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights, map_location=device)
    
    
    
    
    model.load_state_dict(checkpoint)
    model.to(device)

    # Define transformations for the image
    transform = transforms.Compose([
              transforms.Resize((224,224)),
              transforms.Lambda(lambda image: image.convert('RGB')),
              
              transforms.ToTensor(),
              transforms.Normalize(mean=[.5], std=[.5])
              ])

    # Provide the path to the single image
    
    
    # Define the labels
    labels = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
          'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
            'Cardiomegaly', 'Nodule', 'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity',
              'Enlarged Cardiomediastinum'
    ]

    # Predict for the single image
    prediction_results = predict_single_image(image_path, model, transform, labels, device)
    print("Prediction Results:", prediction_results)


if __name__ == "__main__":
    main()
   
       