import cv2
import os

def capture_and_save_image(filename='captured_image.jpg'):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        return

    # Save the frame as a JPEG file
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")

    # Release the webcam
    cap.release()

# Call the function to capture and save the image

directory = 'dataset2'
parent = os.getcwd()

path = os.path.join(parent,directory)
os.mkdir(path)

filename = 'captured_image.jpg'
capture_and_save_image(os.path.join(path, filename))
