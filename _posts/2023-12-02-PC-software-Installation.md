---
layout: post
title: PC Software Installation
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

PC Software : Colab, Notepad++, Git-for-Windows, Python3-for-Windows, GPU libraries (CUDA & CuDNN), etc.

---
## 雲端平台
### Google Colab 教學
[Google Colab 新手的入門教學](https://tw.alphacamp.co/blog/google-colab)<br>

---
## Python 

### [Python Programming](https://www.programiz.com/python-programming)

### [Python Tutorials](https://www.w3schools.com/python/python_intro.asp)

---
## 程式編輯器

### For Windows: install [Notepad++](https://notepad-plus-plus.org/downloads/)

### For Ubuntu / MacOS: no intallation needed, use built-in editors
* **nano** (for Ubuntu / MacOS)<br>
* **vim** (for Ubuntu / MacOS)<br>

---
## Linux作業系統模擬器

### [Git-for-Windows](https://gitforwindows.org/)
**[Download](https://github.com/git-for-windows/git/releases/latest)**<br>

---
### [Linux Command 命令列指令與基本操作入門教學](https://blog.techbridge.cc/2017/12/23/linux-commnd-line-tutorial/)
* `ls -l` (列出目錄檔案)<br>
* `cd ~` (換目錄)<br>
* `mkdir new` (產生新檔案夾)<br>
* `rm file_name` (移除檔案)<br>
* `rm –rf directory_name` (移除檔案夾)<br>
* `df .` (顯示SD卡已用量)<br>
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
## Python解譯器

### [Python3 for Windows](https://www.python.org/downloads/windows/)
Python 3.11.9 - April 2, 2024<br>
Download [Windows installer (64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)<br>

### [Windows如何安裝Python3.11](https://ailog.tw/lifelog/2023/01/30/win-python311/#google_vignette)

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
* 啟動 GitBash (啟動 Linux終端機)

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
### GPU 函式庫安裝
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 
  - [CUDA installation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
* [CuDNN](https://developer.nvidia.com/cudnn)
  - [cuDNN installation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

---
### [PyTorch get-started](https://pytorch.org/get-started/locally/)<br>
#### [PyTorch Tutorials](https://pytorch.org/tutorials/)
`pip install torch torchvision torchaudio`<br>

---
### Tensorflow
#### [Tensorflow Turorials](https://www.tensorflow.org/tutorials)
`pip install tensorflow`<br>

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

