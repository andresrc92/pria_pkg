import cv2
import os

def save_frames_from_video(video_path, output_dir, interval=1):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the frame rate of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval)  # Convert time interval to frame interval

    frame_count = 0
    saved_count = 0

    while video_capture.isOpened():
        # Read the next frame
        success, frame = video_capture.read()

        if not success:
            break

        # Save the frame if it's on the correct interval
        if frame_count % interval_frames == 0:
            frame_filename = f"frame_{saved_count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved: {frame_path}")
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print("Done processing video.")

# Example usage:
video_file_path = "your_video.webm"  # Replace with your .webm video file path
output_directory = "frames"  # Directory where the frames will be saved
save_frames_from_video(video_file_path, output_directory, interval=1)
