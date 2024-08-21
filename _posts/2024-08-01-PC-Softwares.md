---
layout: post
title: PC softwares
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Colab, Notepad++, Git-for-Windows, Python3-for-Windows, GPU libraries (CUDA & CuDNN), etc.

---
## 雲端平台

### [Google Colab 新手的入門教學](https://tw.alphacamp.co/blog/google-colab)

### [Kaggle 使用介紹](https://rkuo2000.github.io/AI-course/lecture/2023/12/01/Kaggle-Intro.html)

---
## 程式語言

### [Python Programming](https://www.programiz.com/python-programming)

### [Python Tutorials](https://www.w3schools.com/python/python_intro.asp)

---
## 程式編輯器

### For Windows
* Download [Notepad++ v8.6.9](https://notepad-plus-plus.org/downloads/)

### For Ubuntu / MacOS
* **nano** (for Ubuntu / MacOS)
* **vim** (for Ubuntu / MacOS)

---
## Linux作業系統模擬器

### [Git-for-Windows](https://gitforwindows.org/)

---
### [Linux Command 命令列指令與基本操作入門教學](https://blog.techbridge.cc/2017/12/23/linux-commnd-line-tutorial/)
* `ls -l` (列出目錄檔案)<br>
* `cd ~` (換目錄)<br>
* `mkdir new` (產生新檔案夾)<br>
* `rm file_name` (移除檔案)<br>
* `rm –rf directory_name` (移除檔案夾)<br>
* `df .` (顯示SSD/硬碟的用量)<br>
* `df -a` (顯示所有儲存裝置的用量)<br>
* `du –sh directory` (查看某檔案夾之儲存用量)<br>
* `free` (檢查動態記憶體用量)<br>
* `ps –a`   (列出正在執行的程序)<br>
* `kill -9 567`  (移除程序 id=567)<br>
* `cat /etc/os-release` (列出顯示檔案內容，此檔案是作業系統版本)<br>
* `vi file_name` (編輯檔案)<br>
* `nano file_name` (編輯檔案)<br>
* `clear` (清除螢幕顯示)<br>
* `history` (列出操作記錄)<br>

### [GNU / Linux 各種壓縮與解壓縮指令](https://note.drx.tw/2008/04/command.html)

---
### GPU 工具箱與函式庫
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
  - [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
* [CuDNN](https://developer.nvidia.com/cudnn)
  - [cuDNN installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

---
## Python 解譯器與軟體包

### [Python3 for Windows](https://www.python.org/downloads/windows/) ~  Python 3.11.9 (windows installer 64-bit)

### [Windows 如何安裝Python3.11](https://ailog.tw/lifelog/2023/01/30/win-python311/#google_vignette)

---
### Ubuntu OS
`$ python3 -V`<br>

**Ubuntu 20.04 LTS**<br>
Python 3.8.10<br>

**Ubuntu 22.04 LTS**<br>
Python 3.10.12<br>

**Ubuntu 24.04 LTS**<br>
Python 3.12.3

---
### Install Python packages 
啟動 GitBash (啟動 Linux終端機)<br>

`python3 -V`<br>
`python3 –m pip install --upgrade pip`<br>
`pip -V`<br>
`pip install jupyter`<br>
`pip install pandas`<br>
`pip install matplotlib pillow imutils`<br>
`pip install opencv-python`<br>
`pip install scikit-learn`<br>
`git clone https://github.com/rkuo2000/cv2`<br>


---
### PyTorch
#### [Get-Started](https://pytorch.org/get-started/locally/)<br>
#### [PyTorch Tutorials](https://pytorch.org/tutorials/)
`pip install torch torchvision torchaudio`<br>

---
### Tensorflow
#### [Tensorflow Turorials](https://www.tensorflow.org/tutorials)
`pip install tensorflow`<br>

---
## LLM 安裝

### Ollama

#### [download Ollama on Windows](https://ollama.com/download/windows)
**Blog** [How to run Ollama on Windows](https://medium.com/@researchgraph/how-to-run-ollama-on-windows-8a1622525ada)<br>

#### [download Ollama on Mac](https://ollama.com/download/mac)

#### [download Ollama on Linux](https://ollama.com/download/linux)
`curl -fsSL https://ollama.com/install.sh | sh`<br>

```
ollama -h
ollama -v
ollama pull tinyllama
ollama list
ollama run tinyllama
```

**[ollama library](https://ollama.com/library)**<br>

---
### [LM Studio](https://lmstudio.ai/)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

