---
layout: post
title: Pose Estimation
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Pose Estimation includes Applications, Human Pose Estimation, Head Pose Estimation & VTuber, Hand Pose Estimation , Object Pose Estimation.

---
## Pose Estimation Applications

### [健身鏡](https://johnsonfitnesslive.com/?action=mirror_pro_intro)
![](https://johnsonfitnesslive.com/images/mirrorPro-parallax-bg2-img03.gif)

---
### 運動裁判 (Sport Referee)
<iframe width="1163" height="654" src="https://www.youtube.com/embed/VZgXUBi_wkM" title="Building an Advanced AI Basketball Referee" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<iframe width="1163" height="654" src="https://www.youtube.com/embed/NnJYQxWoUyo" title="【桌球行動裁判】專題作品Demo - AI應用工程師養成班" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---
### [馬術治療](https://www.inside.com.tw/article/21711-aigo-interview-aifly)
![](https://bucket-image.inkmaginecms.com/version/desktop/cabinet/files/consoles/1/teams/1/2022/10/iVZJfECzMvwKhiXAfNxdK309wlkBdNx7IKNGZmGV.png)

---
### [跌倒偵測](https://www.chinatimes.com/realtimenews/20201203005307-260418?chdtv)
<table>
  <tr>
  <td><img src="https://images.chinatimes.com/newsphoto/2020-12-03/1024/20201203005495.jpg"></td>  
  <td><img src="https://matching.org.tw/website/uploads_product/website_1/P0000100000044_4_123.jpg"></td>
  </tr>
</table>

### [產線SOP](https://www.inside.com.tw/article/21716-aigo-interview-beseye-alpha)
* 距離：人臉辨識技術，可辨識距離大約僅兩公尺以內，而人體骨幹技術不受距離限制，15 公尺外遠距離的人體也能精準偵測，進而分析人體外觀、四肢與臉部的特徵，以達到偵測需求。
* 角度：在正臉的情況下，臉部辨識的精準度非常高，目前僅有極少數相貌非常相似的同卵雙胞胎能夠騙過臉部辨識技術；但在非正臉的時候，臉部辨識的精準度就會急遽下降。在很多實際使用的場景，並不能要求每個參與者都靠近鏡頭、花幾秒鐘以正臉掃描辨識，在這種情況下，人體骨幹分析就能充分派上用場，它的精準度甚至比「非正臉的人臉辨識」高出 30% 以上。

---
### [Pose-controlled Lights](https://github.com/burningion/dab-and-tpose-controlled-lights)
![](https://github.com/burningion/dab-and-tpose-controlled-lights/raw/master/images/dab-tpose.gif?raw=True)

---
## Human Pose Estimation

**Benchmark:** [https://paperswithcode.com/task/pose-estimation](https://paperswithcode.com/task/pose-estimation)<br>

<table>
<tr><th>Model Name</th><th>Paper</th><th>Code</th></tr>
<tr>
<td>PCT</td>
<td>https://arxiv.org/abs/2303.11638</td>
<td>https://github.com/gengzigang/pct</td>
</tr>
<tr>
<td>ViTPose</td>
<td>https://arxiv.org/abs/2204.12484v3</td>
<td>https://github.com/vitae-transformer/vitpose</td>
</tr>
</table>
	
### PoseNet
**Paper:**  [arxiv.org/abs/1505.07427](https://arxiv.org/abs/1505.07427)<br>
**Code:** [rwightman/posenet-pytorch](https://github.com/rwightman/posenet-pytorch)<br>
**Kaggle:** [PoseNet Pytorch](https://www.kaggle.com/code/rkuo2000/posenet-pytorch)<br>

![](https://debuggercafe.com/wp-content/uploads/2020/10/keypoint_exmp.jpg)
![](https://www.researchgate.net/profile/Soroush-Seifi/publication/335989945/figure/fig2/AS:806499555233793@1569295886946/The-Posenet-architecture-Yellow-modules-are-shared-with-GoogleNet-while-green-modules.ppm)
![](https://i1.wp.com/parleylabs.com/wp-content/uploads/2020/01/image-1.png?resize=1024%2C420&ssl=1)

---
### OpenPose
**Paper:** [arxiv.org/abs/1812.08008](https://arxiv.org/abs/1812.08008)<br>
**Code:** [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)<br>
**Kaggle:** [OpenPose Pytorch](https://www.kaggle.com/code/rkuo2000/openpose-pytorch)<br>

<p><img src="https://viso.ai/wp-content/uploads/2021/01/Keypoints-Detected-by-OpenPose-on-the-COCO-Dataset.jpg"><img src="https://media.arxiv-vanity.com/render-output/5509832/x2.png"></p>

---
### DensePose
**Paper:** [arxiv.org/abs/1802.00434](https://arxiv.org/abs/1802.00434)<br>
**Code:** [facebookresearch/DensePose](https://github.com/facebookresearch/Densepose)<br>

---
### Multi-Person Part Segmentation
**Paper:** [arxiv.org/abs/1907.05193](https://arxiv.org/abs/1907.05193)<br>
**Code:** [kevinlin311tw/CDCL-human-part-segmentation](https://github.com/kevinlin311tw/CDCL-human-part-segmentation)

![](https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/cdcl_teaser.jpg?raw=true)

---
### YOLO-Pose
**Paper:** [YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss](https://arxiv.org/abs/2204.06806)<br>

---
### ViTPose
**Paper:** [ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://arxiv.org/abs/2204.12484)<br>
**Paper:** [ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation](https://arxiv.org/abs/2212.04246)<br>
**Paper:** [ViTPose++: Vision Transformer for Generic Body Pose Estimation](https://arxiv.org/abs/2212.04246)<br>
**Code:** [https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose)<br>

<div><img src="https://media.giphy.com/media/13oe6zo6b2B7CdsOac/giphy.gif"><img src="https://media.giphy.com/media/4JLERHxOEgH0tt5DZO/giphy.gif"></div>

![](https://cdn.supervisely.com/blog/vitpose-state-of-the-art-pose-estimation-model-in-supervisely/architecture.png?width=800)

---
### ED-Pose
**Paper:** [Explicit Box Detection Unifies End-to-End Multi-Person Pose Estimation](https://arxiv.org/abs/2302.01593)<br>
**Code:** [https://github.com/IDEA-Research/ED-Pose](https://github.com/IDEA-Research/ED-Pose)<br>
<table>
<tr>
<td><img src="https://github.com/IDEA-Research/ED-Pose/blob/master/figs/crowd%20scene.gif?raw=true"></td>
<td><img src="https://github.com/IDEA-Research/ED-Pose/blob/master/figs/fast_speed.gif?raw=true"></td>
</tr>
</table>

---
### OSX-UBody
**Paper:** [One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer](https://arxiv.org/abs/2303.16160)<br>
**Code:** [https://github.com/IDEA-Research/OSX](https://github.com/IDEA-Research/OSX)<br>
![](https://github.com/IDEA-Research/OSX/blob/main/assets/grouned_sam_osx_demo.gif?raw=true)

---
### Human Pose as Compositional Tokens
**Paper:** [https://arxiv.org/abs/2303.11638](https://arxiv.org/abs/2303.11638)<br>
**Code:** [https://github.com/gengzigang/pct](https://github.com/gengzigang/pct)<br>

<div><img src="https://github.com/Gengzigang/PCT/raw/main/demo/demo_0.gif"><img src="https://github.com/Gengzigang/PCT/raw/main/demo/demo_1.gif"></div>

---
### DWPose
**Paper:** [Effective Whole-body Pose Estimation with Two-stages Distillation](https://arxiv.org/abs/2307.15880)<br>
**Code:** [https://github.com/IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose)<br>
![](https://github.com/IDEA-Research/DWPose/blob/onnx/resources/lalaland.gif?raw=true)

---
### Group Pose
**Paper:** [Group Pose: A Simple Baseline for End-to-End Multi-person Pose Estimation](https://arxiv.org/abs/2308.07313)<br>
**Code:** [https://github.com/Michel-liu/GroupPose](https://github.com/Michel-liu/GroupPose)<br>
![](https://github.com/Michel-liu/GroupPose/blob/main/.github/overview.jpg?raw=true)

---
### [YOLOv8 Pose](https://docs.ultralytics.com/tasks/pose/)
**Kaggle:** [https://www.kaggle.com/rkuo2000/yolov8-pose](https://www.kaggle.com/code/rkuo2000/yolov8-pose)<br>

---
### MMPose
[Algorithms](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html)<br>
**Code:** [https://github.com/open-mmlab/mmpose](https://github.com/open-mmlab/mmpose)<br>
* support two new datasets: UBody, 300W-LP
* support for four new algorithms: MotionBERT, DWPose, EDPose, Uniformer

<iframe width="920" height="520" src="https://user-images.githubusercontent.com/15977946/124654387-0fd3c500-ded1-11eb-84f6-24eeddbf4d91.mp4" title="MMPos Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## 3D Human Pose Estimation
**Benchmark:** [https://paperswithcode.com/task/3d-human-pose-estimation](https://paperswithcode.com/task/3d-human-pose-estimation)<br>

### MotionBERT
**Paper:** [MotionBERT: A Unified Perspective on Learning Human Motion Representations](https://arxiv.org/abs/2210.06551)<br>
**Code:** [https://github.com/Walter0807/MotionBERT](https://github.com/Walter0807/MotionBERT)<br>
![](https://camo.githubusercontent.com/dbdf38903ae38c5839f38b672a22373cede7667521a75d02b7f9f2ac700d6863/68747470733a2f2f6d6f74696f6e626572742e6769746875622e696f2f6173736574732f7465617365722e676966)

---
### BCP+VHA R152 384x384
**Paper:** [Representation learning of vertex heatmaps for 3D human mesh reconstruction from multi-view images](https://arxiv.org/abs/2306.16615v1)<br>

---
## Face Datasets

###  300-W: [300 Faces In-the-Wild](https://ibug.doc.ic.ac.uk/resources/300-W/)
![](https://ibug.doc.ic.ac.uk/media/uploads/images/300-w/figure_1_68.jpg)

---
### 300W-LPA: [300W-LPA Database](https://sites.google.com/view/300w-lpa-database)
![](https://github.com/rkuo2000/AI-course/blob/main/images/300W-LPA.png?raw=true)

---
### LFPW: [Labeled Face Parts in the Wild (LFPW) Dataset](https://neerajkumar.org/databases/lfpw/)
![](https://neerajkumar.org/databases/lfpw/index_files/image002.png)

---
### HELEN: [Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/)
![](https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-01-28_at_9.49.56_PM.png)

---
### AFW: [Annotated Faces in the Wild](https://datasets.activeloop.ai/docs/ml/datasets/afw-dataset/)
AFW (Annotated Faces in the Wild) is a face detection dataset that contains 205 images with 468 faces. Each face image is labeled with at most 6 landmarks with visibility labels, as well as a bounding box.
![](https://datasets.activeloop.ai/wp-content/uploads/2022/09/afw-dataset-Activeloop-Platform-visualization-image-1024x466.webp)

---
### IBUG: [https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
![](https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/problems.jpg)

---
### Head Pose Estimation
**Code:** [yinguobing/head-pose-estimation](https://github.com/yinguobing/head-pose-estimation)<br>
**Kaggle:** [https://www.kaggle.com/rkuo2000/head-pose-estimation](https://www.kaggle.com/rkuo2000/head-pose-estimation)

<table>
  <tr>
    <td><img src="https://github.com/rkuo2000/head-pose-estimation/blob/master/doc/demo.gif?raw=true"></td>
    <td><img src="https://github.com/rkuo2000/head-pose-estimation/blob/master/doc/demo1.gif?raw=true"></td>
  </tr>
</table>

### HopeNet
**Paper:** [Fine-Grained Head Pose Estimation Without Keypoints](https://arxiv.org/abs/1710.00925)<br>
![](https://miro.medium.com/max/700/1*KFeS0YYzAOM1XFokGf3bZA.png)

**Code:** [Hopenet](https://github.com/axinc-ai/ailia-models/tree/master/face_recognition/hopenet)<br>
**Code:** [https://github.com/natanielruiz/deep-head-pose](https://github.com/natanielruiz/deep-head-pose)<br>
![](https://github.com/natanielruiz/deep-head-pose/blob/master/conan-cruise.gif?raw=true)

**Blog:** [HOPE-Net : A Machine Learning Model for Estimating Face Orientation](https://medium.com/axinc-ai/hope-net-a-machine-learning-model-for-estimating-face-orientation-83d5af26a513)

---
## VTuber
[Vtuber總數突破16000人，增速不緩一年增加3000人](https://www.4gamers.com.tw/news/detail/50500/virtual-youtuber-surpasses-16000)
依據日本數據調查分析公司 User Local 的報告，在該社最新的 [User Local VTuber](https://virtual-youtuber.userlocal.jp/document/ranking) 排行榜上，有紀錄的 Vtuber 正式突破了 16,000 人。

1位 [Gawr Gura(がうるぐら サメちゃん)](https://virtual-youtuber.userlocal.jp/user/0DCB37A5BB880687_c8ea33) Gawr Gura Ch. hololive-EN
<iframe width="730" height="411" src="https://www.youtube.com/embed/VDE9T4iSL5Y" title="【3RD ANNIVERSARY】3 YEARS OF SHARK #Gura3 #mocopi" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

2位 [キズナアイ](https://virtual-youtuber.userlocal.jp/user/D780B63C2DEBA9A2_fa95ae) [Kizuna AI](https://kizunaai.com/)
<iframe width="730" height="411" src="https://www.youtube.com/embed/8QOAT1dLf1w" title="【非公開ライブ映像】「LINX」 from &quot;Kizuna AI hello, world 2022&quot;" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

**VTuber-Unity = Head-Pose-Estimation + Face-Alignment + GazeTracking**<br>

---
### [VRoid Studio](https://vroid.com/studio)
![](https://images.ctfassets.net/bg7jm93753li/7BCJCKvgD7tMwD5R3m2ZSK/58bb7b585e2e5bb79f752969c3f857d9/keyvisual_renewal_en.png?fm=webp&w=768)

---
### [VTuber_Unity](https://github.com/kwea123/VTuber_Unity)

<p align="center"><img src="https://github.com/kwea123/VTuber_Unity/blob/master/images/debug_gpu.gif?raw=true"></p>

---
### [OpenVtuber](https://github.com/1996scarlet/OpenVtuber)

<p align="center"><img src="https://github.com/1996scarlet/OpenVtuber/blob/master/docs/images/one.gif?raw=true"></p>

---
## Hand Pose Estimation

### Hand3D
**Paper:** [arxiv.org/abs/1705.01389](https://arxiv.org/abs/1705.01389)<br>
**Code:** [lmb-freiburg/hand3d](https://github.com/lmb-freiburg/hand3d)<br>
![](https://github.com/lmb-freiburg/hand3d/blob/master/teaser.png?raw=true)

---
### InterHand2.6M
**Paper:** [https://arxiv.org/abs/2008.09309](https://arxiv.org/abs/2008.09309)<br>
**Code:** [https://github.com/facebookresearch/InterHand2.6M](https://github.com/facebookresearch/InterHand2.6M)<br>
**Dataset:** [Re:InterHand Dataset](https://mks0601.github.io/ReInterHand/)<br>
![](https://github.com/facebookresearch/InterHand2.6M/blob/main/assets/teaser.gif?raw=true)

---
### InterWild
**Paper:** [Bringing Inputs to Shared Domains for 3D Interacting Hands Recovery in the Wild](https://arxiv.org/abs/2303.13652)<br>
**Code:** [https://github.com/facebookresearch/InterWild/tree/main#test](https://github.com/facebookresearch/InterWild)<br>
![](https://github.com/facebookresearch/InterWild/blob/main/assets/teaser.png?raw=true)
<table>
  <tr>
  <td><img src="https://github.com/facebookresearch/InterWild/blob/main/assets/demo1.png?raw=true"></td>
  <td><img src="https://github.com/facebookresearch/InterWild/blob/main/assets/demo2.png?raw=true"></td>
  </tr>
  <tr>
  <td><img src="https://github.com/facebookresearch/InterWild/blob/main/assets/demo3.png?raw=true"></td>
  <td><img src="https://github.com/facebookresearch/InterWild/blob/main/assets/demo4.png?raw=true"></td>
  </tr>
  <tr>
  <td><img src="https://github.com/facebookresearch/InterWild/blob/main/assets/demo5.png?raw=true"></td>
  <td><img src="https://github.com/facebookresearch/InterWild/blob/main/assets/demo6.png?raw=true"></td>
  </tr>  
</table>

### RenderIH
**Paper:** [RenderIH: A Large-scale Synthetic Dataset for 3D Interacting Hand Pose Estimation](https://arxiv.org/abs/2309.09301)<br>
**Code:** [https://github.com/adwardlee/RenderIH](https://github.com/adwardlee/RenderIH)<br>
[![Watch the video](https://youtu.be/vt5fpE0bzSY)](https://youtu.be/vt5fpE0bzSY)
<video width="400" controls><source src="https://youtu.be/vt5fpE0bzSY"></video>

**TransHand** - transformer-based pose estimation network<br>
![](https://adwardlee.github.io/view_renderih/static/images/network.png)

---
## Exercises of Pose Estimation

### PoseNet
**Kaggle:** [https://www.kaggle.com/rkuo2000/posenet-pytorch](https://www.kaggle.com/rkuo2000/posenet-pytorch)<br>
**Kaggle:** [https://www.kaggle.com/rkuo2000/posenet-human-pose](https://www.kaggle.com/rkuo2000/posenet-human-pose)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/PoseNet_bodylines.png?raw=true)

---
### OpenPose
**Kaggle:** [https://github.com/rkuo2000/openpose-pytorch](https://github.com/rkuo2000/openpose-pytorch)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/OpenPose_pytorch_racers.png?raw=true)
![](https://github.com/rkuo2000/AI-course/blob/main/images/OpenPose_pytorch_fall1.png?raw=true)

---
### MMPose
**Kaggle:**[https://www.kaggle.com/rkuo2000/mmpose](https://www.kaggle.com/rkuo2000/mmpose) <br>
#### 2D Human Pose
![](https://github.com/open-mmlab/mmpose/blob/master/demo/resources/demo_coco.gif?raw=true)
#### 2D Human Whole-Body
![](https://user-images.githubusercontent.com/9464825/95552839-00a61080-0a40-11eb-818c-b8dad7307217.gif)
#### 2D Hand Pose
![](https://user-images.githubusercontent.com/11788150/109098558-8c54db00-775c-11eb-8966-85df96b23dc5.gif)
#### 2D Face Keypoints
![](https://user-images.githubusercontent.com/11788150/109144943-ccd44900-779c-11eb-9e9d-8682e7629654.gif)
#### 3D Human Pose
![](https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif)
#### 2D Pose Tracking
![](https://user-images.githubusercontent.com/11788150/109099201-a93dde00-775d-11eb-9624-f9676fc0e478.gif)
#### 2D Animal Pose
![](https://user-images.githubusercontent.com/11788150/114201893-4446ec00-9989-11eb-808b-5718c47c7b23.gif)
#### 3D Hand Pose
![](https://user-images.githubusercontent.com/28900607/121288285-b8fcbf00-c915-11eb-98e4-ba846de12987.gif)
#### WebCam Effect
![](https://user-images.githubusercontent.com/15977946/124059525-ce20c580-da5d-11eb-8e4a-2d96cd31fe9f.gif)

---
### Basketball Referee
**Code:** [AI Basketball Referee](https://github.com/ayushpai/AI-Basketball-Referee)<br>
該AI主要追蹤兩個東西：球的運動軌跡和人的步數, using YOLOv8 Pose<br>
<iframe width="1166" height="656" src="https://www.youtube.com/embed/VZgXUBi_wkM" title="Building an Advanced AI Basketball Referee" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---
### Head Pose Estimation
**Kaggle:** [https://kaggle.com/rkuo2000/head-pose-estimation](https://kaggle.com/rkuo2000/head-pose-estimation)<br>
<iframe width="652" height="489" src="https://www.youtube.com/embed/BHwHmCUHRyQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### VTuber-Unity 
**Head-Pose-Estimation + Face-Alignment + GazeTracking**<br>

<u>Build-up Steps</u>:
1. Create a character: **[VRoid Studio](https://vroid.com/studio)**
2. Synchronize the face: **[VTuber_Unity](https://github.com/kwea123/VTuber_Unity)**
3. Take video: **[OBS Studio](https://obsproject.com/download)**
4. Post-processing:
 - Auto-subtitle: **[Autosub](https://github.com/kwea123/autosub)**
 - Auto-subtitle in live stream: **[Unity_live_caption](https://github.com/kwea123/Unity_live_caption)**
 - Encode the subtitle into video: **[小丸工具箱](https://maruko.appinn.me/)**
5. Upload: YouTube
6. [Optional] Install CUDA & CuDNN to enable GPU acceleration
7. To Run <br>
`$git clone https://github.com/kwea123/VTuber_Unity` <br>
`$python demo.py --debug --cpu` <br>

<p align="center"><img src="https://github.com/kwea123/VTuber_Unity/blob/master/images/debug_gpu.gif?raw=true"></p>

---
### OpenVtuber
<u>Build-up Steps</u>:
* Repro [Github](https://github.com/1996scarlet/OpenVtuber)<br>
`$git clone https://github.com/1996scarlet/OpenVtuber`<br>
`$cd OpenVtuber`<br>
`$pip3 install –r requirements.txt`<br>
* Install node.js for Windows <br>
* run Socket-IO Server <br>
`$cd NodeServer` <br>
`$npm install express socket.io` <br>
`$node. index.js` <br>
* Open a browser at  http://127.0.0.1:6789/kizuna <br>
* PythonClient with Webcam <br>
`$cd ../PythonClient` <br>
`$python3 vtuber_link_start.py` <br>

<p align="center"><img src="https://camo.githubusercontent.com/83ad3e28fa8a9b51d5e30cdf745324b09ac97650aea38742c8e4806f9526bc91/68747470733a2f2f73332e617831782e636f6d2f323032302f31322f31322f72564f33464f2e676966"></p>


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*
