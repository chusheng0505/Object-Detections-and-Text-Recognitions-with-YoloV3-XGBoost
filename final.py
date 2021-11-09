#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.feature import hog

import pickle
from xgboost import XGBClassifier


# In[ ]:


""" ------------------ 偵測圖中具有文字的目標位置 ------------------  """

""" ## ==================== 呼叫Yolov3 ====================  ## """
def Load_YoloV3(model_weights,model_cfg):
    return cv2.dnn.readNet(model_weights,model_cfg)

""" ## ==================== 畫出BoundingBox 並將之切割出來 ====================  ## """
def Get_BoundingBox(path_InputImage,model_yolov3,scaling_ratio,threshold = 0.3):
    img = cv2.resize(cv2.imread(path_InputImage),None , fx = scaling_ratio , fy = scaling_ratio)
    height , width , channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model_yolov3.setInput(blob)
    layer_names = model_yolov3.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model_yolov3.getUnconnectedOutLayers()]
    outputs = model_yolov3.forward(output_layers)
    
    boundingBox = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boundingBox.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boundingBox, confidences, 0.5, 0.4)
    if indexes == ():
        return ()
    else:
        for i in range(len(boundingBox)):
            if i in indexes:
                x, y, w, h = boundingBox[i]
        return (x,y,w,h,img)
    
 

""" ## ====================  Image Prepocessing ====================  ## """

def Thereshold_Image(img_gray,thresh=127,max_=255):
    return cv2.threshold(img_gray,thresh,max_,cv2.THRESH_BINARY|cv2.THRESH_OTSU)    
    
def Get_CharsBoundingBox(image,model_yolov3,scaling_ratio,threshold = 0.3):
    img = cv2.resize( image ,None , fx = scaling_ratio , fy = scaling_ratio)
    height , width , channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model_yolov3.setInput(blob)
    layer_names = model_yolov3.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model_yolov3.getUnconnectedOutLayers()]
    outputs = model_yolov3.forward(output_layers)
    
    boundingBox = []
    confidences = []
    class_ids = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold:
                
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boundingBox.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boundingBox, confidences, 0.5, 0.4)
    if indexes == ():
        return ()
    else:
        boxes = []
        for i in range(len(boundingBox)):
            if i in indexes:
                x, y, w, h = boundingBox[i]
                x = math.ceil(x/scaling_ratio)
                y = math.ceil(y/scaling_ratio)
                w = math.ceil(w/scaling_ratio)
                h = math.ceil(h/scaling_ratio)
                boxes.append([x,y,w,h])
        return boxes


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', 'yolo_cfg = \'your yolov3.cfg path with format .cfg\'\nyolo_weights = \'your yolov3.model_weights path with format .weights\'\nyolo_chars_weights = \'your yolov3.model_weights path (to detect chars) with format .weights\'\noriginal_image_path = \'your images path\'\n\nscaling_ratio = 0.4\nimages_path = glob.glob(original_image_path)\nnet = Load_YoloV3(yolo_weights , yolo_cfg)\nnet_chars = Load_YoloV3(yolo_chars_cfg , yolo_weights)\n\n"""\n: params : image_unable_crop : collect the images name which are unable to be detected out\n: params : image_crop : collect those results which are detected by yolov3 \n\n# Try to reduce confidences scores in NMS(Non Maximum Suppression)\n## From 0.4 --> 0.3\n\n"""\nimage_unable_crop = [] \nimage_crop = [] \nfor k in tqdm(range(len(images_path))):\n    image_name = images_path[k]\n    bounding_box = Get_BoundingBox(images_path[k],net,scaling_ratio = scaling_ratio)\n    if bounding_box == ():\n        image_unable_crop.append(image_name)\n        print(k , image_name)\n        pass\n    else:\n        x , y , w , h , img = bounding_box\n        x = x - 25 if x-25 > 0 else 0\n        w = w + 50\n        y = y - 5 if y - 5 > 0 else 0\n        h = h + 10\n        img_crop = img[ y : y+h , x : x+w ]\n        h,w = img_crop.shape[:2]\n        h,w = math.ceil(w/scaling_ratio),math.ceil(h/scaling_ratio)\n        img_crop = cv2.resize(img_crop,(h,w))\n        image_crop.append([image_name,img_crop])')


# In[ ]:


"""
: params : Chars_ : To collect the bouding boxes index of each characters

"""

Chars_ = []
for i in tqdm(range(len(image_crop))):
    name_image = image_crop[i][0].split('\\')[-1][:-4]
    img = cv2.resize(image_crop_[i],(700,150))
    chars_boxes = sorted(Get_CharsBoundingBox(img,net_chars,scaling_ratio))
    if chars_boxes == ():
        image_unable_to_predict.append(name_image)
    else:    
        chars = []
        for k in range(len(chars_boxes)):
            x,y,w,h = chars_boxes[k]
            initial_x = x
            initial_y = y
            final_x = x + w
            final_y = y + h
            chars.append([name_image,initial_y , final_y , initial_x , final_x])
        Chars_.append(chars)


# In[ ]:


"""
# To detect the original images are vertical flipped or not

: params : Chars_Rotated_ : To collect the chars images after detection

"""

Chars_Rotated_ = []
for m in tqdm(range(len(image_crop))):
    img_ = cv2.resize(image_crop[m][1],(700,150))
    
    if Chars_[m] == []:
        Chars_Rotated_.append([])
    
    else:
        if len(Chars_[m]) < 2:
            char_rotated_ = []
            for n in range(len(Chars_[m])):
                initial_y = Chars_[m][n][1][0]
                final_y = Chars_[m][n][1][1]
                initial_x = Chars_[m][n][1][2]
                final_x = Chars_[m][n][1][3]
                char_rotated_.append(img_[initial_y:final_y,initial_x:final_x])
            Chars_Rotated_.append(char_rotated_)
        
        else:
            index_ = [0]
            distance = []
            for j in range(len(Chars_[m])-1):
                distance.append(Chars_[m][j+1][1][2] - Chars_[m][j][1][3])
            index_ = np.append(index_,np.where(np.array(distance) >= round(np.mean(distance))+5)[0])
            if len(index_) <= 4:
                num_split = [0]*3
                num_split = num_split[:len(index_)]
                for k in range(len(index_)-1):
                    num_split[k] = len(Chars_[m][ index_[k] :index_[k+1] ])
                num_split[0] = num_split[0] +1
                num_split[-1] = len(Chars_[m]) - np.sum(num_split)
            else:
                num_split = 0
    
            if num_split == 0:
                char_rotated_ = []
                for n in range(len(Chars_[m])):
                    initial_y = Chars_[m][n][1][0]
                    final_y = Chars_[m][n][1][1]
                    initial_x = Chars_[m][n][1][2]
                    final_x = Chars_[m][n][1][3]
                    char_rotated_.append(img_[initial_y:final_y,initial_x:final_y])
                Chars_Rotated_.append(char_rotated_)
            else:
                if num_split[0] < num_split[-1]:
                    char_rotated_ = []
                    for n in range(len(Chars_[m])):
                        char_rotated_.append(Chars_[m][n][1])
                    char_rotated_ = sorted(char_rotated_,key=lambda x:x[2])[::-1]
                
                    img_rotated = []
                    for n in range(len(char_rotated_)):
                        initial_y = char_rotated_[n][0]
                        final_y = char_rotated_[n][1]
                        initial_x = char_rotated_[n][2]
                        final_x = char_rotated_[n][3]
                        img_rotated.append(cv2.rotate(img_[initial_y:final_y,initial_x:final_x],cv2.ROTATE_180))
                    Chars_Rotated_.append(img_rotated)
                
                else:
                    char_rotated_ = []
                    for n in range(len(Chars_[m])):
                        initial_y = Chars_[m][n][1][0]
                        final_y = Chars_[m][n][1][1]
                        initial_x = Chars_[m][n][1][2]
                        final_x = Chars_[m][n][1][3]
                        char_rotated_.append(img_[initial_y:final_y,initial_x:final_x])
                    Chars_Rotated_.append(char_rotated_)


# In[ ]:


"""
# Final Checking the splitted chars
## If there are images with too large width --> it is possible that contains more than one chars in one images
 width_of_images > 90 : high probability that contains more chars 
 
"""
Chars_Final_Modify = []
for i in tqdm(range(len(Chars_Rotated_))):
    chars_final_modify = []
    for j in range(len(Chars_Rotated_[i])):
        if Chars_Rotated_[i] == [] :
            chars_final_modify.append(Chars_Rotated_[i])
        else:
            if Chars_Rotated_[i][j].shape ==0:
                pass
            else:
                _ , w , _ = Chars_Rotated_[i][j].shape
                if w <= 90:
                    chars_final_modify.append(Chars_Rotated_[i][j])
                else:
                    center_x = w // 2
                    for n in range(2):
                        chars_final_modify.append(Chars_Rotated_[i][j][:,center_x*i:center_x*(i+1)])
    Chars_Final_Modify.append(chars_final_modify)


# In[ ]:


"""
# HOG features
## with params : { 'orientations' : 8 , 'pixels_per_cell' : (9, 9) , 'cells_per_block' : (1, 1) }
              
"""

HOG = []
for i in tqdm(range(len(Chars_Rotated_))):
    if len(Chars_Rotated_[i]) == 0:
        HOG.append(' ')
    else:
        hog_features = []
        for j in range(len(Chars_Rotated_[i])):
            if Chars_Rotated_[i][j].size == 0:
                pass
            else:
                img_gray = cv2.cvtColor(Chars_Rotated_[i][j],cv2.COLOR_BGR2GRAY)
                img_resize = cv2.resize(img_gray,(227,227))
                _,thresh = cv2.threshold(img_resize,125,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
                hog_features.append(hog(thresh, orientations = 8 , pixels_per_cell = (9, 9),
                                        cells_per_block = (1, 1) , visualize=False))
        HOG.append(hog_features)


# In[ ]:


path_xgb = 'path of xgboost model trained'
xgb_model = pickle.load(open(path_xgb, 'rb'))

mapping = {'0': '0','1': '1','2': '2','3': '3','4': '4','5': '5',
           '6': '6','7': '7','8': '8','9': '9','10': 'A','11': 'B',
           '12': 'C','13': 'D','14': 'E','15': 'F','16': 'G',
           '17': 'H','18': 'J','19': 'K','20': 'L','21': 'M',
           '22': 'N','23': 'P','24': 'Q','25': 'R','26': 'S',
           '27': 'T','28': 'U','29': 'V','30': 'W','31': 'X',
           '32': 'Y','33': 'Z'}


# In[ ]:


"""
# Prediction
: params : Pred : Collect the final predicting results

"""

Pred = []
for i in tqdm(range(len(HOG))):
    if HOG[i] == ' ':
        Pred .append(' ')
    else:
        pred_xgb = xgb_model.predict(HOG[i])
        Pred.append(''.join([mapping[str(pred_xgb[j])] for j in range(len(pred_xgb))]))


# In[ ]:




