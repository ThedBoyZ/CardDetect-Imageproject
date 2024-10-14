from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor # ->  from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO("card-yolo8.pt")
# list_res = []
result = model.predict(source="0", show=True) # accepts all formats - img/floder/vid.* ()
# list_res.append(result.name)
# if (len(list_res)> 10):    
#     print(max(set(list_res),key=list_res.count))
