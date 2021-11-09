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
![__eN=NSp5V3VuC9dHV9664GdpOgFf3Q_6_0](https://user-images.githubusercontent.com/55430748/140849510-7623db13-603e-4251-b711-8be7654cea13.png)
![_1UDp_d33Sykrs6V2ZkovqVvwdtTbW=7_5_1](https://user-images.githubusercontent.com/55430748/140849520-dccf9c07-e1e7-4f5b-ae0b-ca98b852d383.png)
![_cupNTdDY9geb4ohRfCmowssGw2H7PF_8_2](https://user-images.githubusercontent.com/55430748/140849528-1e712b4a-b446-4a8d-bb18-607df82bd2b1.png)
![_93es8OLoCyvfH5mkC7io4sZhZ5flMW_5_4](https://user-images.githubusercontent.com/55430748/140849541-9af4dbb9-c42a-49ac-a8dc-275c982dfcf3.png)
6244df-3044-4b58-bf5d-a9cbae3191b4.png)
![_aEUhsBCvi5XSdQuS=Rf2zZlmgidqWvF_5_5](https://user-images.githubusercontent.com/55430748/140849551-744aca54-20e4-48a5-b77a-0b98ecf94e01.png)
![=YhmQm46MFBzLuVx7_1t9vmwkXqCLxfY_5_6](https://user-images.githubusercontent.com/55430748/140849557-74f83539-ae2e-4d70-a1de-a74d4470dd3f.png)
![=8aD28_bpgyEZCTgBoFOrJB1HYahnRno_4_7](https://user-images.githubusercontent.com/55430748/140849561-b69f5e92-81fe-4edc-9956-d86a95fc199f.png)
![3hm1YTfpFH7JT8iqFKvDZnvPPiRC_3_8](https://user-images.githubusercontent.com/55430748/140849571-a3a0a1a3-eb8e-4ac4-b020-4163adc6bcb8.png)
![1=XUFgMHYhPGey1bWnT2sN4pJWQf6Dr_3_9](https://user-images.githubusercontent.com/55430748/140849575-1f94de9e-4baa-451c-9afd-8a8a5172a4a2.png)
![_etUJHOxLVHj8FBapTEXz7gV6qxEYN_9_A](https://user-images.githubusercontent.com/55430748/140849800-5e70fab5-41d3-483a-9b52-deb895a13c1a.png)
![5scu1b_XOpymrPzoyTuK84mHo96ogkk_2_B](https://user-images.githubusercontent.com/55430748/140849804-934b3a65-176f-4951-ae2b-88d6a8589bb5.png)
![YtXIKISljpquEP1OzgveEeyCwKz4sFr9_2_C](https://user-images.githubusercontent.com/55430748/140849822-ee88b34a-2689-46c1-be1c-26ecf11132ec.png)
![O=JWF5uF_PHlpnCXh6r6x_1jRofR=6k_2_D](https://user-images.githubusercontent.com/55430748/140849832-f34bb76d-e605-4c38-8478-3e3e73d6062b.png)
![1LQi1BY8CHsuhsTfIZhncOTl6jx9CwQ_8_E](https://user-images.githubusercontent.com/55430748/140849839-6a14727e-807e-40be-a1a0-d20ab50328ac.png)
![=FXufjC2QtDLPp2Jk9o=V=9ecTgy1ICV_1_F](https://user-images.githubusercontent.com/55430748/140849844-bf8539c6-43c0-4651-a57d-c7bd36c4facd.png)
![2wXfFGac_NQBcslRhwQ_M=aipcp66xq_1_G](https://user-images.githubusercontent.com/55430748/140849849-ba45447d-54e0-4173-ab63-af7a270a8a8c.png)
![_LVuHIIIVCo5n7qG3=7zPg4R4DPH9D_2_H](https://user-images.githubusercontent.com/55430748/140849854-59e120bd-cd99-490a-a96a-7c73af4275a5.png)
![9gNqlGteIKwUPIlxlon4Np8IMYNcT8su_2_J](https://user-images.githubusercontent.com/55430748/140849859-40418dc7-3fbb-4bd9-aee1-b2afc633cc1b.png)
![_i29RZW9mO_p5JHpbgZ8oKLif_U3guxJ_2_K](https://user-images.githubusercontent.com/55430748/140849862-39ba697a-c8d5-4566-ae4c-5632aea854ad.png)
![_93es8OLoCyvfH5mkC7io4sZhZ5flMW_1_L](https://user-images.githubusercontent.com/55430748/140849871-fbd3e73c-a13d-4740-917a-7516867ebca6.png)
![=bR3=mEeRPo2yHYufvB2zg7KOo5DJ_y_2_M](https://user-images.githubusercontent.com/55430748/140849876-6d767f61-78bc-4f83-9da7-ec3e6f06f6fa.png)
![1f9rVQC3ctThWEC3U6y7xUvpqsTYjg_1_N](https://user-images.githubusercontent.com/55430748/140849882-fe6e1559-c7d1-42d0-954e-091903605c64.png)
![2DJO92cuezluigFtCJza6_5axDGGu3H1_2_P](https://user-images.githubusercontent.com/55430748/140849890-4b55eeb8-4230-41ba-926d-1867bcec992f.png)
![IFEzUfTbDt8WJ6WWKQyqEY6JYxqjdGmL_7_1](https://user-images.githubusercontent.com/55430748/140849898-05d9a96a-f90c-4779-9cfe-9b6746a1ddb1.png)
![1KRMM1ryeY_3TcmCbqI_j9SH7Ki6JEB_2_R](https://user-images.githubusercontent.com/55430748/140849903-71425c13-1206-424c-aae0-0eed4b7122a5.png)
![=IqWiIw4rMdojOkXl_ZJa79R7Xp9GbH_2_S](https://user-images.githubusercontent.com/55430748/140849912-27a953a6-f27e-41e8-97d6-ca8a41497d5d.png)
![_nXmR9SaD7j_POvFmQncfE4EkVeskdb_2_T](https://user-images.githubusercontent.com/55430748/140849917-514cf276-e771-4ad1-a542-5e96b2fb9f6c.png)
![_vqUmNM49fDFMROb6qhry4X463=7oGZ_2_U](https://user-images.githubusercontent.com/55430748/140849926-189fdce4-9bb8-4601-9f49-631733130c2b.png)
![2MLmsvHUXP5N__e7uPMvXiuP6ciTO6I_2_V](https://user-images.githubusercontent.com/55430748/140849930-6da96b57-5b24-479c-989f-5ac275e2a1c2.png)
![=Fl=aWSz37Wi_tE8=3Zj_PnvFH5dWS36_1_W](https://user-images.githubusercontent.com/55430748/140849937-f0842c8e-f596-4b21-aebb-95b64aff13b9.png)
![=8zXrXmdjgnyGrDW7W5Cn5ITSS7MzERF_2_X](https://user-images.githubusercontent.com/55430748/140849949-3739775d-cbaa-4b0d-9af2-90b78c86b53b.png)
![1k2VhvPBKWkWtkHsZK85Ud2yEjVOVSoU_9_Y](https://user-images.githubusercontent.com/55430748/140849956-4672cac1-4a74-4b2d-894d-d54a1ef77aa5.png)
![=WNo2paaxUIGTSeVDEVwFq1D28Hlp2o__2_Z](https://user-images.githubusercontent.com/55430748/140849964-a742df02-53b0-4c67-a2e5-1c62c5c981ff.png)


- Accuracy of Classifications : <br>
**acc** = 0.9602 - 0.984 <br>
- Confusion Matirx : <br>
![confusion_matrix](https://user-images.githubusercontent.com/55430748/140850117-9adf2088-7106-4273-b498-8b9ec7c6a0f8.png =250x250) <br>
![non_diagonal_element_error](https://user-images.githubusercontent.com/55430748/140850119-d39cc0dc-307d-4126-b2d5-c26f8d85de68.png =250x250) <br>


- Results of Text Recognitions : <br>
### UnFlipped Case :
- ![image](https://user-images.githubusercontent.com/55430748/138632411-dd493d72-51ee-4b49-9c6b-d8ed0a89d8a4.png)

### Flipped Case :
![image](https://user-images.githubusercontent.com/55430748/138632463-f8635492-2440-4af5-b345-14b413173731.png)

