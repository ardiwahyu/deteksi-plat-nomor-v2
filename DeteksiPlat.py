#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from detect import run_detection
from pathlib import Path
from yolov5.models.common import DetectMultiBackend
import sys
import os
from yolov5.utils.torch_utils import select_device
import numpy as np
from tracker import *
import time
from utils.plots import save_one_box
from math import ceil

import warnings
warnings.filterwarnings("ignore")


# In[9]:


root = Tk()
root.withdraw()
filepath = askopenfilename()
print(filepath)


# In[3]:


def imshow(img,window_name='image',key=0):
    cv2.imshow(window_name,img)
    cv2.waitKey(key)
    cv2.destroyAllWindows()


# In[4]:


def prepare(jenis):
    ROOT = 'yolov5'
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
    device = select_device('')
    path_weight = 'config/'+jenis+'.pt'
    path_title = 'config/data_'+jenis+'.yaml'
    return DetectMultiBackend(path_weight, device=device, dnn=False, data=path_title, fp16=False)


# In[5]:

tracker = EuclideanDistTracker()
# model_motor = prepare('motor')
model_plat = prepare('plat_mobil')
model_mobil = prepare('mobil')


# In[6]:


def boundedFrame(img,x,y,width,height):
    cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 255), 2)
    submat = img[y:y+height, x:x+width]
    return submat, img


# In[13]:


saved_plat_ids = []
def save_plat(frame_asli, loc_roi, box_id):
    x, y, w, h, id = box_id
    is_saved = id in saved_plat_ids
    


# In[37]:


def detection(frame,x,y,width,height):
    # start_time = time.time()
    y_line = y+int(height/3)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_roi,frame = boundedFrame(frame,x,y,width,height)
    path_roi = 'data/detect/mobil_detect.jpg'
    cv2.imwrite(path_roi, frame_roi)
    arr_loc_motor, frame_roi = run_detection(model_mobil,path_roi,3)
    boxes_ids = tracker.update(listtensor_to_listxywh(arr_loc_motor))
    for box_id in boxes_ids:
        x1, y1, w, h, id = box_id
        object_motor = frame_roi[y1:y1+h, x1:x1+w]
        path_object = 'data/detect/plat_detect.jpg'
        cv2.imwrite(path_object, object_motor)
        loc_plat, object_plat = run_detection(model_plat,path_object,2)
        # for loc in loc_plat :
        #     x2, y2, w2, h2 = tensor_to_xywh(loc)
        #     if (y+y2+h2 > y_line and id not in saved_plat_ids):
        #         # save_one_box(loc, object_motor, Path('hasil/'+str(id)+'.jpg'))
        #         cv2.imwrite('hasil/'+str(id)+'.jpg', object_motor[y2:y2+h2, x2:x2+w2])
        #         saved_plat_ids.append(id)
        frame_roi[y1:y1+h, x1:x1+w]= object_plat
        cv2.rectangle(frame_roi, (x1+w-60, y1), (x1+w, y1+40), (0,0,0), -1)
        cv2.putText(frame_roi, str(id+1), (x1+w-50, y1+30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    frame[y:y+frame_roi.shape[0], x:x+frame_roi.shape[1]] = frame_roi
    # frame = cv2.line(frame, (x,y_line), (x+width,y_line), (0,255,0), 2)
    # end_time = time.time()
    # print("--- %s seconds ---" % (end_time - start_time))
    return frame

# In[23]:z

def tensor_to_xyxy(tensor):
    return int(tensor[0]), int(tensor[1]), int(tensor[2]), int(tensor[3])
    
def tensor_to_xywh(tensor):
    return int(tensor[0]), int(tensor[1]), int(tensor[2])-int(tensor[0]), int(tensor[3])-int(tensor[1])

def listtensor_to_listxywh(listtensor):
    listxywh = []
    for tensor in listtensor:
        listxywh.append(tensor_to_xywh(tensor))
    return listxywh
    
# In[38]:


cap = cv2.VideoCapture(filepath)
if not cap.isOpened():
    print("Error opening video file")
idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended or error occurred")
        break
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if (idx%int(fps/fps) == 0) :
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", ceil((frame_width/frame_height)*900), 900)
        # cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        frame = detection(frame,20,550,1400,512)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    idx += 1
cap.release()
cv2.destroyAllWindows()

