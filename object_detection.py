import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = [] 
file = 'labels.txt'
with open(file,'rt') as fpt :
    classLabels = fpt.read().rstrip('\n').split('\n')


#configurations as per the model config file
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) #255/2
model.setInputMean((127.5,127.5,127.5)) ## as mobilenet => takes [-1,1]
model.setInputSwapRB(True)

# Read Image
# image = cv2.imread('man_with_car.jpg')
# classIndex,confidence,bbox = model.detect(image,confThreshold = 0.5)

# font_scale = 18
# font_style = cv2.FONT_HERSHEY_PLAIN

# for classInd,conf,box in zip(classIndex.flatten(),confidence.flatten(),bbox):
#     cv2.rectangle(image,box,(255,0,2),6)
#     cv2.putText(image,classLabels[classInd-1],(box[0]+10,box[1]+40),font_style,fontScale = font_scale,color = (0,255,0),thickness = 5)

# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.show()

#Video Demo

cap = cv2.VideoCapture('sample.mp4')

#webcam
# cap = cv2.VideoCapture(1)

#check if the video is opened correctly

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('can not open video')

font_scale = 3
font_style = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    classIndex,confidence,bbox = model.detect(frame,confThreshold = 0.55)

    if (len(classIndex) != 0):
        for classInd,conf,box in zip(classIndex.flatten(),confidence.flatten(),bbox):
            if classInd<=80:
                cv2.rectangle(frame,box,(255,0,2),2)
                cv2.putText(frame,classLabels[classInd-1],(box[0]+10,box[1]+40),font_style,fontScale = font_scale,color = (0,255,0),thickness = 3)
    cv2.imshow('Object Detected Window',frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()