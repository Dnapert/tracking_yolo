import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

img = "hc1.jpg" 

results = model(img)


results.show()
#print names and counts
print(results.pandas().xyxy[0].name.value_counts())
#print(results.pandas().xyxy[0])  # print img1 predictions (pixels) 
