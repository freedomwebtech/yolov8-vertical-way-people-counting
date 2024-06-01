import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
from tracker import*
from vidgear.gears import CamGear


model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)


stream = CamGear(source='https://www.youtube.com/watch?v=zkgRVV5zVpE', stream_mode = True, logging=True).start() # YouTube Video URL as input

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
cx1=456
offset=6


count = 0
cx1=435
offset=6

tracker=Tracker()
while True:    
    frame = stream.read()
    if frame is None:
        break
    count += 1
    if count % 3 != 0:
        continue
   
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        

        if cx1<(cx+offset) and cx1>(cx-offset): 
           cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
           cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
           cvzone.putTextRect(frame,f'{id}',(x4,y4),1,1)
        cv2.line(frame,(456,2),(456,499),(255,0,255),2)
        cv2.line(frame,(435,2),(435,499),(0,0,255),2)

        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
           break
stream.stop()  
cv2.destroyAllWindows()
