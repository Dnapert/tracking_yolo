import torch
from sort import *
import cv2
import os

save_path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(save_path):
    os.makedirs(save_path)
output_path = os.path.join(save_path, 'output.avi')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

desired_frame_rate = 30
output_video = cv2.VideoWriter(output_path, fourcc, desired_frame_rate, (640, 480))
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FPS, desired_frame_rate)
mot_tracker = Sort()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
model.conf = 0.65  # Confidence threshold (0-1)

# Dictionary to store unique IDs and their classifications
id_classification_dict = {}

i = 0 # This is just to limit the number of frames for testing

while True:
    ret, image_show = vid.read()
    if not ret:
        break
    image_show = cv2.resize(image_show, (640, 480))
    predictions = model(image_show)
    detections = predictions.pandas().xyxy[0].to_numpy()
    if len(detections) == 0:
        continue

    # Converting detections to the format [x1, y1, x2, y2, confidence, class_idx, name]
    track_detections = []
    for detect in detections:
        x1, y1, x2, y2, conf, class_idx, name = detect
        track_detections.append([x1, y1, x2, y2, class_idx,conf])

    # Updating the tracking using SORT
    track_bbs_ids = mot_tracker.update(torch.Tensor(track_detections))

    for j in range(len(track_bbs_ids)):
        coords = track_bbs_ids[j]
        
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = "ID: " + str(name_idx) 
        class_name = model.names[int(class_idx)]  # Get the class name using the class index

# This next section is just for visualization, we may want to put this in a separate function to easily turn it on/off for testing, we will get better performance without it on
        color = (0, 255, 0)
        cv2.rectangle(image_show, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_show, f"{name}, {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('image', image_show)

        # Update the dictionary with the current ID and its classification
        id_classification_dict[name_idx] = class_name

    output_video.write(image_show)
    i += 1
    if i == 50:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

total_items = {}
for key, value in id_classification_dict.items():
    if value in total_items:
        total_items[value] += 1
    else:
        total_items[value] = 1
print("Totals: ",total_items)
vid.release()
cv2.destroyAllWindows()

# Print the number of unique IDs and their corresponding classifications
print("Number of Unique IDs:", len(id_classification_dict))
#print("ID-Classification Dictionary:", id_classification_dict)
