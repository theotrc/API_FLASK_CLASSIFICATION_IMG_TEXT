
import os
import json
from google.cloud import storage
from dotenv import load_dotenv
import torch.nn as nn
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image





  
def download_model():
    load_dotenv(override=True)
    
    creds = json.loads(os.environ["GCP_SECRET"])
    creds_path = "/tmp/gcp_key.json"
    
    with open(creds_path, "w") as f:
        json.dump(creds, f)
        
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    model_folder = os.environ["MODEL_FOLDER"]
    client = storage.Client()
    bucket = client.bucket(os.environ["GCS_BUCKET"])
    

    print("BUCKET NAME:", bucket)
    bloblist = bucket.list_blobs()
    

    for blob in bloblist:
        print("BLOB NAME:", blob.name)

        local_path = os.path.join("/tmp", blob.name)
        
        # créer les dossiers parents
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        blob.download_to_filename(local_path)
        
        print(f"Downloaded {blob.name} to {local_path}")
        
    return local_path
    


class CLIPMultimodalClassifier(nn.Module):

    def __init__(self, clip_model, num_classes):
        super().__init__()

        self.clip = clip_model

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Call the CLIP model directly with both image and text inputs.
        # The CLIPModel's forward method returns a CLIPOutput object,
        # which contains the 'image_embeds' and 'text_embeds' as tensors.
        clip_outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=False # Ensure the CLIP model itself doesn't compute loss
        )

        image_features = clip_outputs.image_embeds
        text_features = clip_outputs.text_embeds

        features = torch.cat(
            [image_features, text_features],
            dim=1
        )

        logits = self.classifier(features)

        return logits
        
    
    
    
def load_model(modelpath):
    load_dotenv(override=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    num_classes = 7
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a new instance of the model class
    loaded_model = CLIPMultimodalClassifier(
        clip_model,
        num_classes
    )

    # Load the saved state dictionary
    loaded_model.load_state_dict(torch.load(modelpath, map_location=device))
    loaded_model.to(device)
        
    return loaded_model

class Model:
    def __init__(self, model):
        self.model = model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = ["Baby Care", "Beauty and Personal Care", "Computers", "Home Decor & Festive Needs", "Home Furnishing", "Kitchen & Dining", "Watches"]
        
    def predict_category(self, image_path, raw_text):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")

        # Process text and image with the CLIP processor
        inputs = self.processor(
            text=[raw_text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move inputs to the correct device
        pixel_values = inputs["pixel_values"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Make prediction
        self.model.eval() # Ensure model is in evaluation mode
        with torch.no_grad():
            outputs = self.model(
                pixel_values,
                input_ids,
                attention_mask
            )

        # Get the predicted class index
        _, predicted_class_idx = torch.max(outputs.data, 1)

        # Convert the predicted class index back to the original category name
        predicted_category = self.categories[predicted_class_idx.cpu().numpy()[0]]

        return predicted_category
    
    
    
