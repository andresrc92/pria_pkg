import cv2

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
capture_and_save_image('captured_image.jpg')
