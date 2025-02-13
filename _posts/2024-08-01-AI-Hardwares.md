---
layout: post
title: AI Hardwares
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

AI chips, AI hardwares, Edge-AI, Benchrmarks, Frameworks

---
## AI chips
![](https://github.com/rkuo2000/AI-course/blob/main/images/All_GPUs_crowd_3nm_at_TSMC.jpg?raw=true)

---
### [Etched AI](https://www.etched.com/announcing-etched)
![](https://cdn.prod.website-files.com/6570a6bdf377183fb173431e/667a6b7e59b498f452015850_sohu_cropped-p-1600.png)

---
### Neuromorphic Computing
![](https://www.ece.jhu.edu/dfan/wp-content/uploads/2019/09/brain.jpg)

---
**Paper:** [An overview of brain-like computing: Architecture, applications, and future trends](https://www.frontiersin.org/articles/10.3389/fnbot.2022.1041108/full)<br>
<p><img width="50%" height="50%" src="https://www.frontiersin.org/files/Articles/1041108/fnbot-16-1041108-HTML-r1/image_m/fnbot-16-1041108-g010.jpg"></p>

---
### [Top 10 AI Chip Makers of 2023: In-depth Guide](https://research.aimultiple.com/ai-chip-makers/)

### Groq
#### [The Groq LPU™ Inference Engine](https://wow.groq.com/lpu-inference-engine/)
**Paper:** [A Software-defined Tensor Streaming Multiprocessor for Large-scale Machine Learning](https://dl.acm.org/doi/pdf/10.1145/3470496.3527405)
<p><img width="80%" height="80%" src="https://imageio.forbes.com/specials-images/imageserve/6037f9807842cbd489fa3801/The-Groq-processor-is-unique--acting-as-a-single-fast-core-with-on-die-memory-/960x0.png"></p>
<p><img width="80%" height="80%" src="https://imageio.forbes.com/specials-images/imageserve/6037f9e640b8dd29ddc0607d/The-Groq-node-has-four-chips-per-card--similar-to-most-AI-startups-/960x0.png"></p>

---
### [Rain AI](https://rain.ai/approach)
#### Digital In-Memory Compute
<p><img width="50%" height="50%" src="https://images.squarespace-cdn.com/content/v1/65343802e6fb0c67c8ecb591/9b225d68-dd82-4597-8678-060fb97e4af6/Screen+Shot+2023-12-01+at+10.00.51+AM.png?format=1000"></p>

#### Numerics
<p><img width="50%" height="50%" src="https://images.squarespace-cdn.com/content/v1/65343802e6fb0c67c8ecb591/d596c0d7-5516-4573-a6bd-b7a41155fe71/numerics.png?format=1000w"></p>

---
### [Cerebras](https://www.cerebras.net/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw/sw-co-design-for-deep-learning)
<p><img width="50%" height="50%" src="https://www.cerebras.net/wp-content/uploads/2022/08/fabric-uai-1032x696.jpg"></p>

---
### [Tesla](https://www.tesla.com/AI)
**FSD Chip**<br>
![](https://digitalassets.tesla.com/tesla-contents/image/upload/f_auto,q_auto/hardware.jpg)

---
## AI Hardwares

### Google TPU Cloud
[提供2倍以上單位成本效能，Google Cloud第五代TPU登場](https://www.ithome.com.tw/review/162164)<br>
![](https://s4.itho.me/sites/default/files/images/Google%20TPU%20v5e-2023-12-2.jpg)

---
### TAIDE cloud
**Blog:** [【LLM關鍵基礎建設：算力】因應大模型訓練需求，國網中心算力明年大擴充](https://www.ithome.com.tw/news/160091)<br>
國網中心臺灣杉2號，不論是對7B模型進行預訓練（搭配1,400億個Token訓練資料）還是對13B模型預訓練（搭配2,400億個Token資料量）的需求，都可以勝任。<br>
Meta從無到有訓練Llama 2時，需要上千甚至上萬片A100 GPU，所需時間大約為6個月，<br>
而臺灣杉2號採用相對低階的V100 GPU，效能約為1：3。若以臺灣杉2號進行70B模型預訓練，可能得花上9個月至1年。<br>
![](https://s4.itho.me/sites/default/files/styles/picture_size_large/public/field/image/feng_mian_-guo_ke_hui_-960.jpg?itok=fu9S5jYh)

---
## Nvidia
![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*gPajGtDQ-yPGgyFj.gif)

### CUDA & CuDNN
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
* [CuDNN](https://developer.nvidia.com/cudnn)
![](https://developer.nvidia.com/sites/default/files/akamai/cudnn/cudnn_chart.png)

---
### AI SuperComputer
<iframe width="630" height="384" src="https://www.youtube.com/embed/FQ_nDdXQ5TA" title="2024 Nvidia GTC大會 八分鐘帶你看精華" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

**[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)**<br>
![](https://www.storagereview.com/wp-content/uploads/2024/03/Storagereview-GTC-Networking-Switches.jpg)

**DGX GH200**<br>
![](https://www.storagereview.com/wp-content/uploads/2023/06/storagereview-nvidia-dgx-gh200-1.jpg)
![](https://www.storagereview.com/wp-content/uploads/2023/06/Screenshot-2023-06-26-at-11.28.32-AM.png)

---
### AI Data Center
**DGX SuperPOD with DGX GB200 Systems**<br>
![](https://www.nvidia.com/content/nvidiaGDC/us/en_US/data-center/dgx-platform/_jcr_content/root/responsivegrid/nv_container_728664875/nv_container/nv_teaser_copy.coreimg.100.410.jpeg/1710779437424/nvidia-dgx-platform.jpeg)

**HGX H200**<br>
<p><img src="https://cdn.videocardz.com/1/2023/11/NVIDIA-H200-Overview-1536x747.jpg"></p>

### AI Workstatione/Server (for Enterprise)
**DGX H100**<br>
![](https://static.gigabyte.com/StaticFile/Image/tw/dbdec6b0925ad592a9dc2ecfe09d90f9/ModelSectionChildItem/5832/webp/1920)

---
### AI HPC
**HGX A100**<br>
[搭配HGX A100模組，華碩發表首款搭配SXM形式GPU伺服器](https://www.ithome.com.tw/review/147911)<br>
<p><img width="50%" height="50%" src="https://s4.itho.me/sites/default/files/images/ESC%20N4A-E11_2D%20top%20open.jpg"></p>

---
### GPU
[GeForce RTX-5090](https://www.nvidia.com/zh-tw/geforce/graphics-cards/50-series/)<br>
[NVIDIA GeForce RTX 5090評測1/24解禁，RTX 5080評測1/30解禁連同RTX 5090開賣](https://www.4gamers.com.tw/news/detail/69559/nvidia-geforce-rtx-5090-reviews-go-live-january-24-rtx-5080-on-january-30)<br>
![](https://www.zotac.com/download/files/styles/w1024/public/product_gallery/graphics_cards/zt-b50900j-10p-image01.jpg?itok=xfxYvtfG)

---
## AMD Instinct GPUs
### [MI300](https://www.gigabyte.com/tw/Industry-Solutions/amd-instinct-mi300)
304 GPU CUs, 192GB HBM3 memory, 5.3 TB peark theoretical memory bandwidth<br>
![](https://www.gigabyte.com/FileUpload/global/Other/31/Photo/amd-instinct-mi300x.png)

---
#### [MI200](https://www.amd.com/en/products/accelerators/instinct/mi200.html)
220 CUs, 128GB HBM2e memory, 3.2TB/s Peak Memory Bandwidth, 400GB/s Peark aggregate Infinity Fabric<br>
![](https://images.anandtech.com/doci/17054/AMD%20ACP%20Press%20Deck_30.jpg)
![](https://assets-global.website-files.com/63ebd7a58848e8a8f651aad0/6511c7f9ba82103ccfa55d0c_Datacenter-1-p-1080.png)

---
### Intel
![](https://habana.ai/wp-content/uploads/2024/04/reference-board.webp)

#### Gaudi3
Intel® Gaudi® 3 accelerator with L2 cache for every 2 MME and 16 TPC unit<br>
![](https://habana.ai/wp-content/uploads/2024/04/gaudi3-processor-architecture.webp)

---
### AI PC/Notebook
**NPU**: [三款AI PC筆電搶先看！英特爾首度在臺公開展示整合NPU的Core Ultra筆電，具備有支援70億參數Llama 2模型的推論能力](https://www.ithome.com.tw/news/159673)<br>
![](https://s4.itho.me/sites/default/files/images/PXL_20231107_003902969.jpg)
宏碁在現場展示用Core Ultra筆電執行圖像生成模型，可以在筆電桌面螢幕中自動生成動態立體的太空人桌布，還可以利用筆電前置鏡頭來追蹤使用者的臉部輪廓，讓桌布可以朝著使用者視角移動。此外，還可以利用工具將2D平面圖像轉為3D裸眼立體圖。

---
## Edge AI

### Nvidia Jetson
![](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-agx-orin-family-4c25-p@2x.jpg)

![](https://global.discourse-cdn.com/nvidia/original/3X/0/b/0bfc1897e20cca7f4eac2966f2ad5829d412cbeb.jpeg)
![](https://global.discourse-cdn.com/nvidia/optimized/3X/5/a/5af686ee3f4ad71bc44f22e4a9323fe68ed94ba8_2_690x248.jpeg)

![](https://www.seeedstudio.com/blog/wp-content/uploads/2022/07/NVIDIA-Jetson-comparison_00.png)
![](https://robotkingdom.com.tw/wp-content/uploads/2024/12/jetson-orin-nano-super-dev-kit-ari.jpeg)

---
### MediaTek
![](https://i.mediatek.com/hs-fs/hubfs/AI%20Landing%20Page/Images/AI%20Gen%20new_all%20Gen%20NPU.png?width=969&height=305&name=AI%20Gen%20new_all%20Gen%20NPU.png)

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

---
### Kneron

#### KNEO300 EdgeGPT
<p><img src="https://image-cdn.learnin.tw/bnextmedia/image/album/2023-11/img-1701333658-39165.jpg" width="50%" height="50%"></p>

#### KL720
![](https://no.mouser.com/images/marketingid/2022/microsites/118401037/Overview%20Image.png)
L730晶片集成了最先進第三代KDP系列可重構NPU架構，可提供高達8TOPS的有效算力。<br>
- Quad ARM® Cortex™ A55 CPU。
- 內建DSP，可以加速AI模型後處理，語音處理。
- Linux和RTOS、TSMC 12 納米工藝。
- 高達4K@60fps解析度，與主流感測器的無縫 RGB Bayer 接口，多達4通道影像接口。
- 高達3.6eTOPS@int8 / 7.2eTops@int4。
- 支持Cafee、Tensorflow、Tensorflowlite、Pytorch、Keras、ONNX框架。
- 并兼容CNN、Transformer、RNN Hybrid等多種AI模型， 有更高的處理效率和精度。

#### KLM553 AI SoC
KLM5S3採用全新的NPU架構，在行業率先商用支持幾乎無損的INT4精度和Transformer，相比其他晶片具有更高的運算效率及更低功耗，可在多個複雜場景使用，例如ADAS和AIoT等場景應用。<br>
- 基於ARM® Cortex™ A5 CPU。
- 耐能第三代可重構NPU，算力高達0.5eTOPS@int8/1eTOPS@int4，實際運算效率遠高於其他同等硬體規格晶片。
- 支持Caffe、Tensorflow、Tensorflowlite、Pytorch、Keras、ONNX等多種AI框架。
- 優秀的ISP性能，同時支持高達5M@30FPS，120db寬動態、星光級低照、電子防抖及硬體全景魚眼校正等功能。
- 廣泛應用於城市安防、智能駕駛、終端設備、各類廣角鏡頭處理場景等諸多領域。

#### KL530
![](https://www.kneron.com/tw/_upload/image/solution/large/938617699868711f.jpg)
KL530是耐能新一代異構AI晶片，采用全新的NPU架構，在行業率先商用支持INT4精度和Transformer，相比其它晶片具有更高的運算能力及更低功耗，具備強大的圖像處理能力和豐富的接口，將進一步促進邊緣AI晶片在ADAS、AIoT等場景應用。<br>
- 基於ARM Cortex M4 CPU内核的低功耗性能和高能效設計。
- 算力達1 TOPS INT 4，在同等硬件條件下比INT 8的處理效率提升高達70%。
- 支持CNN,Transformer，RNN Hybrid等多種AI模型。
- 智能ISP可基於AI優化圖像質量，強力Codec實現高效率多媒體壓縮。
- 冷啟動時間低於500ms，平均功耗低於500mW。
![](https://www.everfocus.com/upload/catalog_m/b098f7b0f63a26477cd2cb5d08eb12c0.png)

---
### Raspberry Pi 5
![](https://github.com/rkuo2000/AI-course/blob/main/images/RPi-5.jpg?raw=true)
樹莓派 5 搭載一顆64位元四核心 Arm Cortex-A76 處理器，運行速度達2.4GHz，相對於樹莓派 4，CPU效能提升了2-3倍。搭配800MHz的VideoCore VII GPU，提供顯著提升的圖形效能；支援雙4Kp60 HDMI顯示輸出<br>
Broadcom BCM2712 2.4GHz quad-core 64-bit Arm Cortex-A76 CPU, with Cryptographic Extension, 512KB per-core L2 caches, and a 2MB shared L3 cache<br>


#### 套件：[樹莓派 8GB主板+外殼+原廠5A電源器+64GB記憶卡](https://www.ruten.com.tw/item/show?21407267192911)
![](https://gcs.rimg.com.tw/g1/5/48/db/21407287949531_158.jpg)

---
### Realtek

#### [AMB82-MINI](https://www.amebaiot.com/en/amebapro2/#rtk_amb82_mini)
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

### ARM NN

#### [MLPerf](https://mlcommons.org/en/)

#### [MLPerf™ Inference Benchmark Suite](https://github.com/mlcommons/inference)

---
### [NVIDIA’s MLPerf Benchmark Results](https://www.nvidia.com/en-us/data-center/resources/mlperf-benchmarks/)
![](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/resources/mlperf/hpc-charts-mlperf-training-sdchart-june.svg)

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

