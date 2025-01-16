import torch
from torchvision import models, transforms
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load a pre-trained Fast R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load and transform the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Perform object detection
def detect_objects(image_path):
    image = load_image(image_path)
    with torch.no_grad():
        predictions = model(image)
    
    return predictions[0]

# Visualize the results
def visualize_results(image_path, predictions, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        if score >= threshold:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'{label.item()} {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()

# Example usage
image_path = './real_datasets/pg_cone/imgs/1004.png'
predictions = detect_objects(image_path)
visualize_results(image_path, predictions)