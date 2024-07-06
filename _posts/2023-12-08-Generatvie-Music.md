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


