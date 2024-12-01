from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
# Load a model

# Load video
video_path = 'testing\\test2.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = 'testing\\runs_yolo\\test2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Run YOLO inference
model = YOLO("tennis_pose_estimation\\tennis_pose_estimation_m_13\\weights\\best.pt",task='pose')  # load an official model

# run the model on each frame of the video test2 and save the result video
results = model.predict("testing\\test2.mp4",stream=True, device=0,line_width=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Convert PIL image to OpenCV format
        im_bgr = np.array(im_rgb)[..., ::-1]

        # Write frame to video
        out.write(im_bgr)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved to: {output_path}")