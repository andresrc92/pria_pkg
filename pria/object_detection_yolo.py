import cv2
import torch
from torchvision import transforms
from PIL import Image

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define the transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert image to RGB
    img = img.convert('RGB')
    # print(img.size)
    
    # Apply transformation
    img = transform(img).unsqueeze(0)

    # Perform detection
    results = model(img)
    results.print()
    # print(results[0].shape)

    # Draw bounding boxes and labels on the frame
    # for result in results.xyxy[0]:
    #     x1, y1, x2, y2, conf, cls = result
    #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    #     label = f'{model.names[int(cls)]} {conf:.2f}'
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()