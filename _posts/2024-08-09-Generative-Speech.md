---
layout: post
title: Generative Speech 
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

This introduction includes Speech Datasets, Text-To-Speech, Voice Cloning.

---
## Text-To-Speech (using TTS)

### WaveNet
**Paper:** [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)<br>
**Code:** [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)<br>
With a pre-trained model provided here, you can synthesize waveform given a mel spectrogram, not raw text.<br>
You will need mel-spectrogram prediction model (such as Tacotron2) to use the pre-trained models for TTS.<br>
**Demo:** [An open source implementation of WaveNet vocoder](https://r9y9.github.io/wavenet_vocoder/)<br>
**Blog:** [WaveNet: A generative model for raw audio](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio)<br>
![](https://lh3.googleusercontent.com/Zy5xK_i2F8sNH5tFtRa0SjbLp_CU7QwzS2iB5nf2ijIf_OYm-Q5D0SgoW9SmfbDF97tNEF7CmxaL-o6oLC8sGIrJ5HxWNk79dL1r7Rc=w1440-rw-v1)

---
### Tacotron-2
**Paper:** [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)<br>
**Code:** [Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)<br>
![](https://camo.githubusercontent.com/d6c3e238b30a49a31c947dd0c5b344c452b53ab5eb735dc79675b67c92a2cf96/68747470733a2f2f707265766965772e6962622e636f2f625538734c532f5461636f74726f6e5f325f4172636869746563747572652e706e67)

**Code:** [Tacotron 2 (without wavenet)](https://github.com/NVIDIA/tacotron2)<br>
![](https://github.com/NVIDIA/tacotron2/blob/master/tensorboard.png?raw=true)

---
### Forward Tacotron
**Blog:** [利用 ForwardTacotron 創造穩健的神經語言合成](https://blogs.nvidia.com.tw/2021/03/31/creating-robust-neural-speech-synthesis-with-forwardtacotron/)<br>
**Code:** [https://github.com/as-ideas/ForwardTacotron](https://github.com/as-ideas/ForwardTacotron)<br>
![](https://github.com/as-ideas/ForwardTacotron/blob/master/assets/model.png?raw=true)

---
### Few-shot Transformer TTS
**Paper:** [Multilingual Byte2Speech Models for Scalable Low-resource Speech Synthesis](https://arxiv.org/abs/2103.03541)<br>
**Code:** [https://github.com/mutiann/few-shot-transformer-tts](https://github.com/mutiann/few-shot-transformer-tts)<br>

---
### MetaAudio
**Paper:** [MetaAudio: A Few-Shot Audio Classification Benchmark](https://arxiv.org/abs/2204.02121)<br>
**Code:** [MetaAudio-A-Few-Shot-Audio-Classification-Benchmark](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark)<br>
**Dataset:** ESC-50, NSynth, FSDKaggle18, BirdClef2020, VoxCeleb1

---
### SeamlessM4T
**Paper:** [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation](https://arxiv.org/abs/2308.11596)<br>
**Code:** [https://github.com/facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication)<br>
**Colab:** [seamless_m4t_colab](https://github.com/camenduru/seamless-m4t-colab)<br>
[app.py](https://huggingface.co/spaces/facebook/seamless_m4t/blob/main/app.py)<br>

---
### Audiobox
**Paper:** [Audiobox: Generating audio from voice and natural language prompts](https://ai.meta.com/blog/audiobox-generating-audio-voice-natural-language-prompts/)<br>
![](https://s4.itho.me/sites/default/files/styles/picture_size_large/public/field/image/1204-audiobox_meta-960.jpg?itok=JttF4Z4F)

---
### coqui TTS
**Code:** [https://www.kaggle.com/code/rkuo2000/coqui-tts](https://github.com/coqui-ai/TTS)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/coqui-tts](https://www.kaggle.com/code/rkuo2000/coqui-tts)<br>

---
## Speech Seperation

### Looking to Listen
**Paper:** [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation](https://arxiv.org/abs/1804.03619)<br>
**Blog:** [Looking to Listen: Audio-Visual Speech Separation](https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)<br>

![](https://3.bp.blogspot.com/-i8yGQmRfu6k/Ws03pWxgp2I/AAAAAAAACiM/3KgklbbHIvsYo4Tyw3N1TKa7Eywagr4eACLcBGAs/s640/image6.jpg)
<iframe width="640" height="360" src="https://www.youtube.com/embed/Z_ogAiVoE1g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="640" height="360" src="https://www.youtube.com/embed/uKwUL7vt03M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="640" height="360" src="https://www.youtube.com/embed/_7aMiqXubWo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### VoiceFilter
**Paper:** [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826)<br>
**Code:** [mindslab-ai/voicefilter](https://github.com/mindslab-ai/voicefilter)<br>
Training took about 20 hours on AWS p3.2xlarge(NVIDIA V100)<br>
**Code:** [jain-abhinav02/VoiceFilter](https://github.com/jain-abhinav02/VoiceFilter)<br>
The model was trained on Google Colab for 30 epochs. Training took about 37 hours on NVIDIA Tesla P100 GPU.<br>

![](https://github.com/jain-abhinav02/VoiceFilter/raw/master/assets/images/model_workflow.PNG)
<iframe width="600" height="338" src="https://www.youtube.com/embed/2BF_1X7bmds" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### VoiceFilter-Lite
**Paper:** [VoiceFilter-Lite: Streaming Targeted Voice Separation for On-Device Speech Recognition](https://arxiv.org/abs/2009.04323)<br>
**Blog:** [](https://google.github.io/speaker-id/publications/VoiceFilter-Lite/)<br>

![](https://google.github.io/speaker-id/publications/VoiceFilter-Lite/resources/architecture.png)
<iframe width="800" height="450" src="https://www.youtube.com/embed/BiWMZdnHuVs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Voice Filter
**Paper:** [Voice Filter: Few-shot text-to-speech speaker adaptation using voice conversion as a post-processing module](https://arxiv.org/abs/2202.08164)<br>

---
## Voice Conversion
**Paper:** [An Overview of Voice Conversion and its Challenges: From Statistical Modeling to Deep Learning](https://arxiv.org/abs/2008.03648)<br>
![](https://d3i71xaburhd42.cloudfront.net/4e1f36855442b761729dad4507513e23ca66206c/7-Figure2-1.png)

**Blog:** [Voice Cloning Using Deep Learning](https://medium.com/the-research-nest/voice-cloning-using-deep-learning-166f1b8d8595)<br>

---
### Voice Cloning
**Paper:** [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558.abs)<br>
**Kaggle:** [https://github.com/CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)<br>
![](https://camo.githubusercontent.com/52ad48c214ea4126fc823420629940a0cccbf638ff386709056de790edf8bd1b/68747470733a2f2f692e696d6775722e636f6d2f386c46556c677a2e706e67)<br>

---
### SV2TTS
**Blog:** [Voice Cloning: Corentin's Improvisation On SV2TTS](https://www.datasciencecentral.com/profiles/blogs/voice-cloning-corentin-s-improvisation-on-sv2tts)<br>
**Paper:** [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)<br>
**Code:** [CorentinJ/Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)<br>

![](https://storage.ning.com/topology/rest/1.0/file/get/8569870256)

**Synthesizer** : The synthesizer is Tacotron2 without Wavenet<br>
![](https://storage.ning.com/topology/rest/1.0/file/get/8569881687)

**SV2TTS Toolbox**<br>
<iframe width="1148" height="646" src="https://www.youtube.com/embed/-O_hYhToKoA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---
### Review of Deep Learning Voice Conversion 
**Paper:** [Overview of Voice Conversion Methods Based on Deep Learning](https://www.mdpi.com/2076-3417/13/5/3100)<br>
**Paper:** [Reimagining Speech: A Scoping Review of Deep Learning-Powered Voice Conversion](https://arxiv.org/abs/2311.08104)<br>

---
### Personalized Lightweight TTS
**Paper:** [Personalized Lightweight Text-to-Speech: Voice Cloning with Adaptive Structured Pruning](https://arxiv.org/abs/2303.11816)<br>

---
### [NeMO](https://github.com/NVIDIA/NeMo)
[Nemo ASR](https://catalog.ngc.nvidia.com/orgs/nvidia/collections/nemo_asr)<br>
[Automatic Speech Recognition (ASR)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html)<br>

---
### [Speech Synthesis](https://github.com/AI-FREE-Team/TTS-AIGO)
<iframe width="1115" height="627" src="https://www.youtube.com/embed/Q6DhZMei43M" title="語音合成Speech Synthesis Part I 開源專案簡介" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<iframe width="1132" height="637" src="https://www.youtube.com/embed/dTS7WkBMyuw" title="語音合成Speech Synthesis Part II Nemo 簡介" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<iframe width="1115" height="627" src="https://www.youtube.com/embed/zBbdYIyu-WI" title="語音合成Speech Synthesis Part III 程式碼介紹" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---
### [PolyVoice](https://speechtranslation.github.io/polyvoice/)
**Paper:**[PolyVoice: Language Models for Speech to Speech Translation](https://arxiv.org/abs/2306.02982)<br>
PolyVoice 是一种基于语言模型的 S2ST 框架，能够处理书写和非书写语言。<br>
PolyVoice 使用通过自监督训练方法获得的离散单元作为源语音和目标语音之间的中间表示。PolyVoice 由两部分组成:<br>
Speech-to-Unit（S2UT）翻译模块，将源语言语音的离散单元转换为目标语言语音的离散单元；<br><br>
Unit-to-Speech（U2S）合成模块， 在保留源语言语音说话人风格的同时合成目标语言语音。<br>
![](https://speechtranslation.github.io/polyvoice/pics/overview.png)

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
### LSLM
**Paper:** [Language Model Can Listen While Speaking](https://arxiv.org/abs/2408.02622)<br>
![](https://ziyang.tech/LSLM/pic/duplex.png)
![](https://ziyang.tech/LSLM/pic/model-model.png)

---
## Automatic Speech Recognition (ASR)

### Whisper
**Paper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)<br>
![](https://www.researchgate.net/profile/Zhanibek_Kozhirbayev/publication/376675691/figure/fig2/AS:11431281213642924@1703085158954/Whisper-architecture-representation.png)
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/whisper](https://www.kaggle.com/code/rkuo2000/whisper)<br>

---
### [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models.<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/faster-whisper](https://www.kaggle.com/code/rkuo2000/faster-whisper)<br>

---
### [Open Whisper-style Speech Models (OWSM)](https://www.wavlab.org/activities/2024/owsm/)
**Paper:** [OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer](https://arxiv.org/abs/2401.16658)<br>

---
### Qwen-Audio
**Github:** [https://github.com/QwenLM/Qwen-Audio](https://github.com/QwenLM/Qwen-Audio)<br>
![](https://github.com/QwenLM/Qwen-Audio/raw/main/assets/framework.png)

---
### Canary
**model:** [nvidia/canary-1b](https://huggingface.co/nvidia/canary-1b)<br>
**Paper:** [Less is More: Accurate Speech Recognition & Translation without Web-Scale Data](https://arxiv.org/abs/2406.19674)<br>

---
### [Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3)
**model:** openai/whisper-large-v3<br>
**model:** openai/whisper-large-v3-turbo<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/whisper-large-v3-turbo](https://www.kaggle.com/code/rkuo2000/whisper-large-v3-turbo)<br>

---
### Meta SpiritLM
**Blog:** [Meta開源首個多模態語言模型Meta Spirit LM](https://www.ithome.com.tw/news/165587)<br>
**Code:** [https://github.com/facebookresearch/spiritlm](https://github.com/facebookresearch/spiritlm)<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

