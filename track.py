import torch
from sort import *
import cv2
import os

save_path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(save_path):
    os.makedirs(save_path)
output_path = os.path.join(save_path, 'output_video.avi')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')


desired_frame_rate = 10
output_video = cv2.VideoWriter(output_path, fourcc, desired_frame_rate, (640, 480))
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FPS, desired_frame_rate)
mot_tracker = Sort()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

i = 0

while (True):
    ret, image_show = vid.read()
    if not ret:
        break
    image_show = cv2.resize(image_show, (640, 480))
    predictions = model(image_show)
    print(predictions.pandas().xyxy[0].drop_duplicates(subset=['name']).name.value_counts())
    detections = predictions.pandas().xyxy[0].to_numpy()

    track_bbs_ids = mot_tracker.update(detections)

    for j in range(len(track_bbs_ids.tolist())):
        coords = track_bbs_ids.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = "ID: " + str(name_idx)
        color = (0, 255, 0)
        cv2.rectangle(image_show, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_show, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('image', image_show)
    output_video.write(image_show)
    i += 1
    if i == 100:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
