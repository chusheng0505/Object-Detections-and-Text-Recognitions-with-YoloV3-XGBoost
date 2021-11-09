# Object-Detections-and-Text-Recognitions (In Processing)
Objects Detection and Text Recognitions

In this projects, I am trying to use yolov3 model to detect the targets(text) and recognition it.<br>
Training of yolov3 , https://www.youtube.com/watch?v=_FNfRtXEbr4&t=1421s as my references. <br>


Steps of processing : <br>
1. YoloV3 : Find out the bounding boxes of target and characters <br>
2. HOG : To get features of Characters <br>
3. XGBoost : Model used to classify <br>
3. Join the results <br>

Problems during processing : <br>
1. Some of original images are vertical flip. <br>
1.1 We have to develop a methods to detect. <br>
1.2 By observation,we found that most of the images are in format  'XXXXX  XXX  XX' or 'XXXXX XXXX' or 'XXXXXXXXX' where X are Arabic numerals [0-9] or captital letter [A-Z but not included O and I].<br>
1.3 Detect the sides(left or right) which contains most of X.<br><b>If num(left characters) > num(right characters) :  Normal case  <br>
Else : Vertical Flip </b> <br>

2.There are some characters with less training dataset. <br>
 2.1  Create more training data for those characters by adding <b> Gaussian Noises / Random Chopping / Random Brightness Adjustment </b><br>
 2.2 Total Training Images : ~ 139k with 34 classes <br>


- Results of Detections: <br>
![1633964341928](https://user-images.githubusercontent.com/55430748/136812307-2ac3b6e4-d948-407d-86a4-904bcea64ee6.jpg)

- Results of Cropping : <br>



- Results of Detect Flipped of Images : <br>
![未命名](https://user-images.githubusercontent.com/55430748/138438302-d4299b0e-a71c-41b4-beb1-7cad65a832bf.png) <br>

- Accuracy of Classifications : <br>
**acc** = 0.79 - 0.81 <br>

- Results of Text Recognitions : <br>
### UnFlipped Case :
- ![image](https://user-images.githubusercontent.com/55430748/138632411-dd493d72-51ee-4b49-9c6b-d8ed0a89d8a4.png)

### Flipped Case :
![image](https://user-images.githubusercontent.com/55430748/138632463-f8635492-2440-4af5-b345-14b413173731.png)

