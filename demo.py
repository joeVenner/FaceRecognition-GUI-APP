import cv2
import os

# Open the video file for reading
video_path = 'data\WIN_20230920_07_56_11_Pro.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create a directory to save the frames (if it doesn't already exist)
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

# Loop to read and save frames from the video
frame_count = 0
while True:
    ret, frame = cap.read()  # Read a frame from the video

    if not ret:
        break  # Break the loop if no more frames are available

    # Save the frame as an image file
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f"Frames extracted and saved to '{output_dir}'.")

