import json
from typing import Any

import numpy as np
import timm
import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from timm.data.transforms_factory import transforms_imagenet_eval


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.transform = transforms_imagenet_eval()

        with open("imagenet_1k.json", "r") as f:
            self.labels = list(json.loads(f.read()).values())

    # Define the arguments and types the model takes as input
    def predict(
        self,
        image: Path = Input(description="Image to classify"),
        model_name: str = "resnet18",
    ) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        img = Image.open(image).convert("RGB")
        img = self.transform(img)
        try:
            model = timm.create_model(model_name, pretrained=True)
            model.eval()
        except Exception as e:
            raise e
        # Run the prediction
        with torch.no_grad():
            labels = model(img[None, ...])
            labels = labels[0]  # we'll only do this for one image

        # top 5 preds
        topk = labels.topk(5)[1]
        output = {
            # "labels": labels.cpu().numpy(),
            "topk": [self.labels[x] for x in topk.cpu().numpy().tolist()],
        }

        return output
