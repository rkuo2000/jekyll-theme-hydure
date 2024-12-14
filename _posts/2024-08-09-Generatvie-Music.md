---
layout: post
title: Generative Music
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Music Seperationm, Music Generation, etc.

---
## Music Seperation
### Spleeter
**Paper:** [Spleeter: A FAST AND STATE-OF-THE ART MUSIC SOURCE
SEPARATION TOOL WITH PRE-TRAINED MODELS](https://archives.ismir.net/ismir2019/latebreaking/000036.pdf)<br>
**Code:** [deezer/spleeter](https://github.com/deezer/spleeter)<br>

---
### Wave-U-Net
**Paper:** [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185)<br>
**Code:** [f90/Wave-U-Net](https://github.com/f90/Wave-U-Net)<br>

![](https://github.com/f90/Wave-U-Net/blob/master/waveunet.png?raw=true)

---
### Hyper Wave-U-Net
**Paper:** [Improving singing voice separation with the Wave-U-Net using Minimum Hyperspherical Energy](https://arxiv.org/abs/1910.10071)<br>
**Code:** [jperezlapillo/hyper-wave-u-net](https://github.com/jperezlapillo/hyper-wave-u-net)<br>
**MHE regularisation:**<br>
![](https://github.com/jperezlapillo/Hyper-Wave-U-Net/blob/master/diagram_v2.JPG?raw=true)

---
### Demucs
**Paper:** [Music Source Separation in the Waveform Domain](https://arxiv.org/abs/1911.13254)<br>
**Code:** [facebookresearch/demucs](https://github.com/facebookresearch/demucs)<br>

![](https://github.com/facebookresearch/demucs/blob/main/demucs.png?raw=true)

---
### RVC vs SoftVC
"Retrieval-based Voice Conversion" 和 "SoftVC VITS Singing Voice Conversion" 是兩種聲音轉換技術的不同變種。以下是它們之間的一些區別：<br>

1.方法原理：<br>
Retrieval-based Voice Conversion：這種方法通常涉及使用大規模的語音資料庫或語音庫，從中檢索與輸入語音相似的聲音樣本，並將輸入語音轉換成與檢索到的聲音樣本相似的聲音。它使用檢索到的聲音作為目標來進行聲音轉換。<br>
SoftVC VITS Singing Voice Conversion：這是一種基於神經網路的聲音轉換方法，通常使用變分自動編碼器（Variational Autoencoder，VAE）或其他神經網路架構。專注於歌聲轉換，它的目標是將輸入歌聲樣本轉換成具有不同特徵的歌聲，例如性別、音調等。<br>

2.應用領域：<br>
Retrieval-based Voice Conversion 通常用於語音轉換任務，例如將一個人的語音轉換成另一個人的語音。它也可以用於歌聲轉換，但在歌聲轉換方面通常不如專門設計的方法表現出色。<br>
SoftVC VITS Singing Voice Conversion 主要用於歌聲轉換任務，特別是針對歌手之間的音樂聲音特徵轉換，例如將男性歌手的聲音轉換成女性歌手的聲音，或者改變歌曲的音調和音樂特徵。<br>

3.技術複雜性：<br>
Retrieval-based Voice Conversion 的實現通常較為簡單，因為它主要依賴於聲音樣本的檢索和聲音特徵的映射。<br>
SoftVC VITS Singing Voice Conversion 更複雜，因為它需要訓練深度神經網路模型，可能需要大量的數據和計算資源。<br>

---
### Retrieval-based Voice Conversion
**Blog:** [RVC-WebUI開源專案教學](https://gogoplus.net/%E7%BF%BB%E5%94%B1%E6%9C%80%E5%A5%BD%E7%94%A8%E7%9A%84%E9%96%8B%E6%BA%90%E7%A8%8B%E5%BC%8F-rvc-webui-%E5%85%8B%E9%9A%86%E4%BD%A0%E7%9A%84%E8%81%B2%E9%9F%B3/)<br>
**Code:** [https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)<br>

<iframe width="994" height="491" src="https://www.youtube.com/embed/9nHbw0eUJeE" title="AI 用你的聲音創建歌詞曲 五音不全的人也可以靠AI實現當歌手的夢想 SunoAI + RVC WebUI + ChatGPT" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

---
### GPT-SoVITS
**Blog:** [GPT-SoVITS 用 AI 快速複製你的聲音，搭配 Colab 免費入門](https://medium.com/dean-lin/gpt-sovits-%E7%94%A8-ai-%E5%BF%AB%E9%80%9F%E8%A4%87%E8%A3%BD%E4%BD%A0%E7%9A%84%E8%81%B2%E9%9F%B3-%E6%90%AD%E9%85%8D-colab-%E5%85%8D%E8%B2%BB%E5%85%A5%E9%96%80-f6a620cf7fc6)<br>
**Code:** [https://github.com/RVC-Boss/GPT-SoVITS/](https://github.com/RVC-Boss/GPT-SoVITS/)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/so-vits-svc-5-0](https://www.kaggle.com/code/rkuo2000/so-vits-svc-5-0)<br>

---
## Music Generation

### [OpenAI Jukebox](https://jukebox.openai.com/)
**Blog:** [Jukebox](https://openai.com/blog/jukebox/)<br>
model modified from **VQ-VAE-2**
**Paper:** [Jukebox: A Generative Model for Music](https://arxiv.org/abs/2005.00341)<br>
**Colab:** [Interacting with Jukebox](https://colab.research.google.com/github/openai/jukebox/blob/master/jukebox/Interacting_with_Jukebox.ipynb)<br>

---
### DeepSinger
**Blog:** [Microsoft’s AI generates voices that sing in Chinese and English](https://venturebeat.com/2020/07/13/microsofts-ai-generates-voices-that-sing-in-chinese-and-english/)<br>
**Paper:** [DeepSinger: Singing Voice Synthesis with Data Mined From the Web](https://arxiv.org/abs/2007.04590)<br>
**Demo:** [DeepSinger: Singing Voice Synthesis with Data Mined From the Web](https://speechresearch.github.io/deepsinger/)<br>

![](https://lfs.aminer.cn/upload/pdf_image/5ecf/e0e/5ecfae0e9e795eb20a615049img-002.png)
<p align="center">The alignment model based on the architecture of automatic speech recognition</p>

![](https://lfs.aminer.cn/upload/pdf_image/5ecf/e0e/5ecfae0e9e795eb20a615049img-004.png)
<p align="center">The architecture of the singing model</p>

![](https://lfs.aminer.cn/upload/pdf_image/5ecf/e0e/5ecfae0e9e795eb20a615049img-005.png)
<p align="center">The inference process of singing voice synthesis</p>

---
### [MusicGen](https://ai.honu.io/papers/musicgen/)
**Paper:** [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)<br>
**Code:** [https://github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)<br>

---
### Tiny Audio Diffusion
**Code:** [https://github.com/crlandsc/tiny-audio-diffusion](https://github.com/crlandsc/tiny-audio-diffusion)<br>
<iframe width="750" height="422" src="https://www.youtube.com/embed/m6Eh2srtTro" title="Generate Sounds With AI Using Tiny Audio Diffusion" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*


