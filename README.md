# Object-Detections-and-Text-Recognitions (In Processing)
Objects Detection and Text Recognitions

In this projects, I am trying to use yolov3 model to detect the targets(text) and recognition it.<br>
Training of yolov3 , https://www.youtube.com/watch?v=_FNfRtXEbr4&t=1421s as my references. <br>


Steps of processing : <br>
1. YoloV3 : Find out the bounding boxes of target <br>
2. OpenCV : Crop it out and split into several cropped images   <br>
3. OpenCV : Detect the rotation angles of target <br>
4. CNN : Classified them by classifier <br>
5. Join the results <br>


- Results of Detections: <br>
![1633964341928](https://user-images.githubusercontent.com/55430748/136812307-2ac3b6e4-d948-407d-86a4-904bcea64ee6.jpg)

- Results of Cropping : <br>
![_1TFexxwPqzHZM9JTwJTgmt5Urphufo_10](https://user-images.githubusercontent.com/55430748/138438514-8bcd7489-0469-4c3c-97d6-261dfafe7989.png)
![_3hLk8HXQ3PKDcLhDUhIebkqnac15l_7](https://user-images.githubusercontent.com/55430748/138438525-8da6ac9a-32bb-4d82-afb1-d6b2ccf3a10a.png)
![_5a6SH4kVJenLjIPCfHJBS_z_z3Vx9g_6](https://user-images.githubusercontent.com/55430748/138438533-72a7549e-f5d9-4b7f-87a8-490422673b85.png)
![_6t7M4I9zeLNUVLEOcBVYwqslkxVxwCu_7](https://user-images.githubusercontent.com/55430748/138438543-00a0bd11-4ef0-4958-9855-fc83f040e88a.png)
![==9G3h2OPz8IzwwL8OTWLkIunpD6MvmG_1](https://user-images.githubusercontent.com/55430748/138438567-05d9bc01-f454-47fa-b47e-7238c43ae262.png)


- Results of Detect Flipped of Images : <br>
![未命名](https://user-images.githubusercontent.com/55430748/138438302-d4299b0e-a71c-41b4-beb1-7cad65a832bf.png) <br>

- Accuracy of Classifications : <br>
**acc** = 0.79 - 0.81 <br>

- Results of Text Recognitions : <br>
### UnFlipped Case :
- ![image](https://user-images.githubusercontent.com/55430748/138632411-dd493d72-51ee-4b49-9c6b-d8ed0a89d8a4.png)

### Flipped Case :
![image](https://user-images.githubusercontent.com/55430748/138632463-f8635492-2440-4af5-b345-14b413173731.png)

