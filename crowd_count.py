import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import tempfile
import io

# Load the YOLOv5 model (make sure 'yolov5s.pt' is in the right location or available)
model = YOLO('yolov5s.pt')

# Function to detect people and draw bounding boxes
def detect_and_count_people(frame):
    results = model(frame)  # Perform detection on the frame
    people_count = 0

    # Get bounding boxes, class IDs, and confidence scores
    boxes = results[0].boxes.xyxy  # Bounding box coordinates
    confs = results[0].boxes.conf  # Confidence scores
    classes = results[0].boxes.cls  # Class IDs

    # Iterate over detections
    for i in range(len(boxes)):
        class_id = int(classes[i].item())  # Get class ID
        if class_id == 0:  # Class ID 0 corresponds to 'person' in YOLOv5
            people_count += 1

            # Get bounding box coordinates 
            x1, y1, x2, y2 = map(int, boxes[i])  # Bounding box coordinates
            

            # Draw bounding box and confidence score on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for person
           

    return frame, people_count

# Streamlit application
def main():
    # Streamlit UI configuration
    st.set_page_config(page_title="Skavch Crowd Count Engine", layout="wide")

    # Add an image to the header
    st.image("bg1.jpg", use_column_width=True)  # Adjust the image path as necessary
    st.title("Skavch Crowd Count Engine")

    # File uploader to upload a video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # Store the uploaded video in a temporary file
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, 'wb') as f:
            f.write(video_file.read())

        st.text(f"Processing video...")

        # Open video file using OpenCV
        video_capture = cv2.VideoCapture(temp_video_path)

        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create an in-memory buffer to save the output video
        output_buffer = io.BytesIO()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More compatible codec for cloud environments
        temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        video_writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Process each frame in the video
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Detect people and get frame with bounding boxes
            frame, people_count = detect_and_count_people(frame)

            # Display the people count on the top left corner of the frame
            cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Write the processed frame to the output video
            video_writer.write(frame)

        # Release resources
        video_capture.release()
        video_writer.release()

        # Read the video file back into memory and store it in the buffer
        with open(temp_output_path, 'rb') as f:
            output_buffer.write(f.read())

        output_buffer.seek(0)  # Reset buffer position to the start

        st.text(f"Video processing complete!")

        # Provide a download button for the in-memory video buffer
        st.download_button('Download Processed Video', output_buffer, file_name="output_video.mp4", mime='video/mp4')

if __name__ == '__main__':
    main()