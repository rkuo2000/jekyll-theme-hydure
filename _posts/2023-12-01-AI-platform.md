---
layout: post
title: AI Platforms
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

AI chips, Hardware, ML Benchrmark, Framework, Open platforms

---
## AI chips
### Neuromorphic Computing
![](https://www.ece.jhu.edu/dfan/wp-content/uploads/2019/09/brain.jpg)

---
**Paper:** [An overview of brain-like computing: Architecture, applications, and future trends](https://www.frontiersin.org/articles/10.3389/fnbot.2022.1041108/full)<br>
![](https://www.frontiersin.org/files/Articles/1041108/fnbot-16-1041108-HTML-r1/image_m/fnbot-16-1041108-g010.jpg)
![](https://www.frontiersin.org/files/Articles/1041108/fnbot-16-1041108-HTML-r1/image_m/fnbot-16-1041108-g012.jpg)

---
### [Top 10 AI Chip Makers of 2023: In-depth Guide](https://research.aimultiple.com/ai-chip-makers/)

### Groq
#### [The Groq LPU™ Inference Engine](https://wow.groq.com/lpu-inference-engine/)
**Paper:** [A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3470496.3527405)
![](https://imageio.forbes.com/specials-images/imageserve/6037f9807842cbd489fa3801/The-Groq-processor-is-unique--acting-as-a-single-fast-core-with-on-die-memory-/960x0.png?format=png&width=1440)
![](https://imageio.forbes.com/specials-images/imageserve/6037f9e640b8dd29ddc0607d/The-Groq-node-has-four-chips-per-card--similar-to-most-AI-startups-/960x0.png?format=png&width=1440)

---
### [Rain AI](https://rain.ai/approach)
#### Digital In-Memory Compute
![](https://images.squarespace-cdn.com/content/v1/65343802e6fb0c67c8ecb591/9b225d68-dd82-4597-8678-060fb97e4af6/Screen+Shot+2023-12-01+at+10.00.51+AM.png?format=1000w)
#### Numerics
![](https://images.squarespace-cdn.com/content/v1/65343802e6fb0c67c8ecb591/d596c0d7-5516-4573-a6bd-b7a41155fe71/numerics.png?format=1000w)

---
### [Cerebras](https://www.cerebras.net/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw/sw-co-design-for-deep-learning)
![](https://www.cerebras.net/wp-content/uploads/2022/08/datapath-dataflow.jpg)
![](https://www.cerebras.net/wp-content/uploads/2022/08/fabric-uai-1032x696.jpg)

---
#### [Tesla AI](https://www.tesla.com/AI)
[Enter Dojo: Tesla Reveals Design for Modular Supercomputer & D1 Chip](https://www.hpcwire.com/2021/08/20/enter-dojo-tesla-reveals-design-for-modular-supercomputer-d1-chip/)<br>

[Teslas will be 'more intelligent' than HUMANS by 2033 as their microchips already have 36% the capacity of the brain, study reveals](https://www.dailymail.co.uk/sciencetech/article-11206325/Teslas-smarter-humans-2033-microchips-handle-362T-operations-second.html)<br>

---
## AI Hardware

### Google TPU Cloud
[Google’s Cloud TPU v4 provides exaFLOPS-scale ML with industry-leading efficiency](https://cloud.google.com/blog/topics/systems/tpu-v4-enables-performance-energy-and-co2e-efficiency-gains)
![](https://storage.googleapis.com/gweb-cloudblog-publish/images/1_Cloud_TPU_v4.max-1100x1100.jpg)
One eighth of a TPU v4 pod from Google's world’s largest publicly available ML cluster located in Oklahoma, which runs on ~90% carbon-free energy.<br>
<br>
<p><img src="https://storage.googleapis.com/gweb-cloudblog-publish/images/2_Cloud_TPU_v4.max-1400x1400.jpg" width="50%" height="50%"></p>
TPU v4 is the first supercomputer to deploy a reconfigurable OCS. OCSes dynamically reconfigure their interconnect topology
Much cheaper, lower power, and faster than Infiniband, OCSes and underlying optical components are <5% of TPU v4’s system cost and <5% of system power.

---
### TAIDE cloud
**Blog:** [【LLM關鍵基礎建設：算力】因應大模型訓練需求，國網中心算力明年大擴充](https://www.ithome.com.tw/news/160091)<br>
國網中心臺灣杉2號，不論是對7B模型進行預訓練（搭配1,400億個Token訓練資料）還是對13B模型預訓練（搭配2,400億個Token資料量）的需求，都可以勝任。<br>
Meta從無到有訓練Llama 2時，需要上千甚至上萬片A100 GPU，所需時間大約為6個月，<br>
而臺灣杉2號採用相對低階的V100 GPU，效能約為1：3。若以臺灣杉2號進行70B模型預訓練，可能得花上9個月至1年。<br>
![](https://s4.itho.me/sites/default/files/styles/picture_size_large/public/field/image/feng_mian_-guo_ke_hui_-960.jpg?itok=fu9S5jYh)

---
## AMD Instinct GPUs
### [MI300](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
304 GPU CUs, 192GB HBM3 memory, 5.3 TB peark theoretical memory bandwidth<br>
![](https://www.techspot.com/images2/news/bigimage/2023/06/2023-06-14-image-31.jpg)

---
### [MI200](https://www.amd.com/en/products/accelerators/instinct/mi200.html)
220 CUs, 128GB HBM2e memory, 3.2TB/s Peak Memory Bandwidth, 400GB/s Peark aggregate Infinity Fabric<br>
![](https://images.anandtech.com/doci/17054/AMD%20ACP%20Press%20Deck_30.jpg)

![](https://assets-global.website-files.com/63ebd7a58848e8a8f651aad0/6511c7f9ba82103ccfa55d0c_Datacenter-1-p-1080.png)
![](https://magnifier.cmoney.tw/wp-content/uploads/2023/06/mi300-h100-comp.jpg)

---
## Nvidia

### CUDA & CuDNN
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
* [CuDNN](https://developer.nvidia.com/cudnn)
![](https://developer.nvidia.com/sites/default/files/akamai/cudnn/cudnn_chart.png)

---
### AI SuperComputer
**DGX GH200**<br>
![](https://www.storagereview.com/wp-content/uploads/2023/06/storagereview-nvidia-dgx-gh200-1.jpg)
![](https://www.storagereview.com/wp-content/uploads/2023/06/Screenshot-2023-06-26-at-11.28.32-AM.png)

### AI Data Center
**HGX H200**<br>
<p><img src="https://cdn.videocardz.com/1/2023/11/NVIDIA-H200-Overview-1536x747.jpg"></p>

### AI Workstatione/Server (for Enterprise)
**DGX H100**<br>
系統每個 H100 Tensor Core GPU 性能平均比以前 GPU 高約 6 倍，搭載 8 個 GPU，每個 GPU 都有一個 Transformer Engine，加速生成式 AI 模型。8 個 H100 GPU 透過 NVIDIA NVLink 連接，形成巨大 GPU，也可擴展 DGXH100 系統，使用 400 Gbps 超低延遲 NVIDIA Quantum InfiniBand 將數百個 DGX H100 節點連線一台 AI 超級電腦，速度是之前網路的兩倍。
![](https://static.gigabyte.com/StaticFile/Image/tw/dbdec6b0925ad592a9dc2ecfe09d90f9/ModelSectionChildItem/5832/webp/1920)

---
### AI HPC
**HGX A100**<br>
[搭配HGX A100模組，華碩發表首款搭配SXM形式GPU伺服器](https://www.ithome.com.tw/review/147911)<br>
<p><img src="https://s4.itho.me/sites/default/files/images/ESC%20N4A-E11_2D%20top%20open.jpg"></p>

---
### GPU
[GeForce RTX-4090](https://www.techpowerup.com/gpu-specs/geforce-rtx-4090.c3889)
<p><img src="https://www.pcworld.com/wp-content/uploads/2023/04/geforce-rtx-4090-jensen.jpg" width="50%" height="50%"></p>

---
### AI PC/Notebook
**NPU**: [三款AI PC筆電搶先看！英特爾首度在臺公開展示整合NPU的Core Ultra筆電，具備有支援70億參數Llama 2模型的推論能力](https://www.ithome.com.tw/news/159673)<br>
![](https://s4.itho.me/sites/default/files/images/PXL_20231107_003902969.jpg)
宏碁在現場展示用Core Ultra筆電執行圖像生成模型，可以在筆電桌面螢幕中自動生成動態立體的太空人桌布，還可以利用筆電前置鏡頭來追蹤使用者的臉部輪廓，讓桌布可以朝著使用者視角移動。此外，還可以利用工具將2D平面圖像轉為3D裸眼立體圖。

---
### Edge AI
![](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-agx-orin-family-4c25-p@2x.jpg)

![](https://www.seeedstudio.com/blog/wp-content/uploads/2022/07/NVIDIA-Jetson-comparison_00.png)

![](https://global.discourse-cdn.com/nvidia/original/3X/0/b/0bfc1897e20cca7f4eac2966f2ad5829d412cbeb.jpeg)
![](https://global.discourse-cdn.com/nvidia/optimized/3X/5/a/5af686ee3f4ad71bc44f22e4a9323fe68ed94ba8_2_690x248.jpeg)

---
### [Collections of MPU for Edge AI applications](https://makerpro.cc/2023/08/collections-of-mcu-for-edge-ai-applications/)
#### [天璣 9300](https://www.mediatek.tw/products/smartphones-2/mediatek-dimensity-9300)
* 單核性能提升超過 15%
* 多核性能提升超過 40%
* 4 個 Cortex-X4 CPU 主頻最高可達 3.25GHz
* 4 個 Cortex-A720 CPU 主頻為 2.0GHz
* 內置 18MB 超大容量緩存組合，三級緩存（L3）+ 系統緩存（SLC）容量較上一代提升 29%

#### [天璣 8300](https://www.mediatek.tw/products/smartphones-2/mediatek-dimensity-8300)
* 八核 CPU 包括 4 個 Cortex-A715 大核和 4 個 Cortex-A510 能效核心
* Mali-G615 GPU
* 支援 LPDDR5X 8533Mbps 記憶體
* 支援 UFS 4.0 + 多循環隊列技術（Multi-Circular Queue，MCQ）
* 高能效 4nm 製程

#### ADI MAX78000
![](https://i0.wp.com/makerpro.cc/wp-content/uploads/2023/08/EdgeAI_MCU_P1.jpg?resize=1024%2C414&ssl=1)

#### TI MPU: AM62A、AM68A、AM69A
![](https://i0.wp.com/makerpro.cc/wp-content/uploads/2023/08/1691657090157.jpg?resize=768%2C607&ssl=1)

---
### Kneron 耐能智慧
* KNEO300 EdgeGPT
<p><img src="https://image-cdn.learnin.tw/bnextmedia/image/album/2023-11/img-1701333658-39165.jpg" width="50%" height="50%"></p>

* KL530
![](https://www.kneron.com/tw/_upload/image/solution/large/938617699868711f.jpg)
  - 基於ARM Cortex M4 CPU内核的低功耗性能和高能效設計。
  - 算力達1 TOPS INT 4，在同等硬件條件下比INT 8的處理效率提升高達70%。
  - 支持CNN,Transformer，RNN Hybrid等多種AI模型。
  - 智能ISP可基於AI優化圖像質量，強力Codec實現高效率多媒體壓縮。
  - 冷啟動時間低於500ms，平均功耗低於500mW。
！[](https://github.com/rkuo2000/EdgeAI-course/blob/main/images/Kneron-KL520-devkit.jpg?raw=true！)

* KL720 (算力可達0.9 TOPS/W)

---
### Realtek AmebaPro2
[AMB82-MINI](https://www.amebaiot.com/en/amebapro2/#rtk_amb82_mini)<br>
<p><img src="https://www.amebaiot.com/wp-content/uploads/2023/03/amb82_mini.png" width="50%" height="50%"></p>
* MCU
  - Part Number: RTL8735B
  - 32-bit Arm v8M, up to 500MHz
* MEMORY
  - 768KB ROM
  - 512KB RAM
  - 16MB Flash
  - Supports MCM embedded DDR2/DDR3L memory up to 128MB
* KEY FEATURES
  - Integrated 802.11 a/b/g/n Wi-Fi, 2.4GHz/5GHz
  - Bluetooth Low Energy (BLE) 5.1
  - Integrated Intelligent Engine @ 0.4 TOPS
<iframe width="580" height="327" src="https://www.youtube.com/embed/_Kzqh6JXndo" title="AIoT: AmebaPro2 vs ESP32" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
## [mlplatform.org](https://www.mlplatform.org/)
The machine learning platform is part of the Linaro Artificial Intelligence Initiative and is the home for Arm NN and Compute Library – open-source software libraries that optimise the execution of machine learning (ML) workloads on Arm-based processors.
![](https://www.mlplatform.org/assets/images/assets/images/content/NN-frameworks20190814-800-1d11a6.webp)
<table>
  <tr><td>Project</td><td>Repository</td></tr>
  <tr><td>Arm NN</td><td>[https://github.com/ARM-software/armnn](https://github.com/ARM-software/armnn)</td></tr>
  <tr><td>Compute Library</td><td>[https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary](https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary)</td></tr>
  <tr><td>Arm Android NN Driver</td><td>https://github.com/ARM-software/android-nn-driver</td></tr>
</table>

---
### [ARM NN SDK](https://www.arm.com/zh-TW/products/silicon-ip-cpu/ethos/arm-nn)
免費提供的 Arm NN (類神經網路) SDK，是一組開放原始碼的 Linux 軟體工具，可在節能裝置上實現機器學習工作負載。這項推論引擎可做為橋樑，連接現有神經網路框架與節能的 Arm Cortex-A CPU、Arm Mali 繪圖處理器及 Ethos NPU。<br>

**[ARM NN](https://github.com/ARM-software/armnn)**<br>
Arm NN is the most performant machine learning (ML) inference engine for Android and Linux, accelerating ML on Arm Cortex-A CPUs and Arm Mali GPUs.

---
## Benchmark
### [MLPerf](https://mlcommons.org/en/)

### [MLPerf™ Inference Benchmark Suite](https://github.com/mlcommons/inference)
MLPerf Inference v3.1 (submission 04/08/2023)

| model | reference app | framework | dataset |
| ---- | ---- | ---- | ---- |
| resnet50-v1.5 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | tensorflow, pytorch, onnx | imagenet2012 |
| retinanet 800x800 | [vision/classification_and_detection](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) | pytorch, onnx | openimages resized to 800x800|
| bert | [language/bert](https://github.com/mlcommons/inference/tree/master/language/bert) | tensorflow, pytorch, onnx | squad-1.1 |
| dlrm-v2 | [recommendation/dlrm](https://github.com/mlcommons/inference/tree/master/recommendation/dlrm_v2/pytorch) | pytorch | Multihot Criteo Terabyte |
| 3d-unet | [vision/medical_imaging/3d-unet-kits19](https://github.com/mlcommons/inference/tree/master/vision/medical_imaging/3d-unet-kits19) | pytorch, tensorflow, onnx | KiTS19 |
| rnnt | [speech_recognition/rnnt](https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt) | pytorch | OpenSLR LibriSpeech Corpus |
| gpt-j | [language/gpt-j](https://github.com/mlcommons/inference/tree/master/language/gpt-j)| pytorch | CNN-Daily Mail |

---
### [NVIDIA’s MLPerf Benchmark Results](https://www.nvidia.com/en-us/data-center/resources/mlperf-benchmarks/)
**NVIDIA H100 Tensor Core GPU**<br>

| Benchmark                                  | Per-Accelerator Records |
|--------------------------------------------|-------------------------|
| Large Language Model (LLM)                 | 548 hours (23 days) |
| Natural Language Processing (BERT)         | 0.71 hours |
| Recommendation (DLRM-dcnv2)                | 0.56 hours |
| Speech Recognition (RNN-T)                 | 2.2 hours | 
| Image Classification (ResNet-50 v1.5)      | 1.8 hours |
| Object Detection, Heavyweight (Mask R-CNN) | 2.6 hours |
| Object Detection, Lightweight (RetinaNet)  | 4.9 hours |
| Image Segmentation (3D U-Net)              | 1.6 hours |

---
## Frameworks

### [PyTorch](https://pytorch.org)

### [Tensorflow](https://www.tensorflow.org)

### [Keras 3.0](https://keras.io/keras_3/)
![](https://s3.amazonaws.com/keras.io/img/keras_3/cross_framework_keras_3.jpg)

---
### [MLX](https://github.com/ml-explore/mlx)
MLX is an array framework for machine learning on Apple silicon, brought to you by Apple machine learning research.<br>
[MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html)<br>

---
### TinyML
[EloquentTinyML](https://github.com/eloquentarduino/EloquentTinyML)

### [Tensorflow.js](https://www.tensorflow.org/js/demos)

### [MediaPipe](https://google.github.io/mediapipe/)


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

