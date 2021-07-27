import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'ssd_mobilenet_v3_large_coco/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = [] 
file = 'labels.txt'
with open(file,'rt') as fpt :
    classLabels = fpt.read().rstrip('\n').split('\n')


#configurations
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) #255/2
model.setInputMean(127.5) ## as mobilenet => takes [-1,1]
model.setInputSwapRB(True)

# Read Image
image = cv2.imread('man_with_car.jpg')
classIndex,confidence,bbox = model.detect(image,confThreshold = 0.5)

font_scale = 18
font_style = cv2.FONT_HERSHEY_PLAIN

for classInd,conf,box in zip(classIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(image,box,(255,0,2),6)
    cv2.putText(image,classLabels[classInd-1],(box[0]+10,box[1]+40),font_style,fontScale = font_scale,color = (0,255,0),thickness = 5)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()

#Video Demo
