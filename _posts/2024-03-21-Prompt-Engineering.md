---
layout: post
title: Prompt Engineering
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Prompt Engineering (提示工程)

---
## 初級入門（免費課程）
* [Prompt Engineering with Llama 2](https://www.deeplearning.ai/short-courses/prompt-engineering-with-llama-2/)<br>
* [Deep Learning Generative AI short courses](https://www.deeplearning.ai/short-courses/)<br>

### [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)<br>
aka **In-Context Prompting**<br>
* Basic Prompting
  - Zero-shot
  - Few-shot
* Instruction Prompting
* Self-Consistency Sampling
* Chain-of-Thought (CoT)
* Automatic Prompt Design
* [Augmented Language Models](https://arxiv.org/abs/2302.07842)

---
## 生成式AI導論

### 【生成式AI導論 2024】第3講：訓練不了人工智慧？你可以訓練你自己 (上) — 神奇咒語與提供更多資訊
<iframe width="863" height="486" src="https://www.youtube.com/embed/A3Yx35KrSN0?list=PLJV_el3uVTsPz6CTopeRp2L2t4aL_KgiI" title="【生成式AI導論 2024】第3講：訓練不了人工智慧？你可以訓練你自己 (上) — 神奇咒語與提供更多資訊" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---
### 【生成式AI導論 2024】第4講：訓練不了人工智慧？你可以訓練你自己 (中) — 拆解問題與使用工具
<iframe width="863" height="486" src="https://www.youtube.com/embed/lwe3_x50_uw?list=PLJV_el3uVTsPz6CTopeRp2L2t4aL_KgiI" title="【生成式AI導論 2024】第4講：訓練不了人工智慧？你可以訓練你自己 (中) — 拆解問題與使用工具" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---
### 【生成式AI導論 2024】第5講：訓練不了人工智慧？你可以訓練你自己 (下) — 讓語言彼此合作，把一個人活成一個團隊
<iframe width="1271" height="716" src="https://www.youtube.com/embed/inebiWdQW-4" title="【生成式AI導論 2024】第5講：訓練不了人工智慧？你可以訓練你自己 (下) — 讓語言彼此合作，把一個人活成一個團隊 (開頭有芙莉蓮雷，慎入)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---
## 神奇的咒語
不是對所有的模型都有用<br>

### 1. 要語言模型思考: "Think about it Step-by-Step"
**Paper:** [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)<br>
![](https://ar5iv.labs.arxiv.org/html/2201.11903/assets/x1.png)

---
### 2. 要語言模型解釋: "Explain it"
**Paper:** [Can Large Language Models Be an Alternative to Human Evaluation?](https://arxiv.org/abs/2305.01937)<br>
![](https://aisholar.s3.ap-northeast-1.amazonaws.com/posts/July2023/Can_Large_Language_Models_Be_an_Alternative_to_Human_Evaluation_fig1.png)

---
### 3. 對語言模型情緒勒索: "This is very important to my career"
**Paper:** [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)<br>
![](https://digitalrosh.com/wp-content/uploads/2024/01/EmotionPrompt.jpg)

---
### 4. 對語言模型下原則性指導 （Principled Instructions）
**Paper:** [Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4](https://arxiv.org/abs/2312.16171)<br>
![](https://arxiv.org/html/2312.16171v2/x1.png)
1. 對它說話不用客氣, 對它有什麼要求直說無妨
2. 給它明確的指示, 直說要它做甚麼, 不用跟它說不做什麼
3. 跟它說如果做得好,會給它小費$xxx 
4. 跟它說做不好, 它會得到處罰
5. 跟它說要保證它的答案是沒有偏見的, 而且要避免使用任何刻板印象
...

---
### 5.直接問語言模型給更強大的咒語 (Forward Generation Template)
* `Let's think step by step`
* `Take a deep breath then work on this problem step by step`

---
### 6. 輸入額外範例 
**In-Context Learning**<br>

**Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)<br>
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*1PJi06R7QMTGBh8CdIsW8w.png)

---
### 7. 拆解任務 （Recursive Reprompting and Revision)
**Paper:** [Re3: Generating Longer Stories With Recursive Reprompting and Revision](https://arxiv.org/abs/2210.06774)<br>
**Paper:** [Joint Prompt Optimization of Stacked LLMs using Variational Inference](https://arxiv.org/abs/2306.12509)<br>

---
### 8. 要語言模型去使用工具
**Paper:** [AnyTool: Self-Reflective, Hierarchical Agents for Large-Scale API Calls](https://arxiv.org/abs/2402.04253)<br>
![](https://github.com/rkuo2000/AI-course/blob/main/images/Prompting_AnyTool.png?raw=true)

---

---
### 9. 自動提示比手工編寫的提示好
**Paper:** [The Unreasonable Effectiveness of Eccentric Automatic Prompts](https://arxiv.org/html/2402.10949v2)<br>
**Blog:** [要求AI模仿《星際爭霸戰》能提升效能，但還是用「自動優化」以魔法打敗魔法最有效
](https://www.techbang.com/posts/113796-asking-the-ai-to-answer-in-the-way-of-a-starcraft-character)<br>

---
### 10. 要語言模型摘要文章/論文後寫出簡報/文章
**收集, 啟發, 輸出**<br>
* 一段一段分析文章中提供了什麼解決方法
* 如何解決我提出的需求與問題？
* 根據問題, 詳列摘要重點
* 改寫成精簡但精采的簡報大綱

<iframe width="988" height="556" src="https://www.youtube.com/embed/iBWJ5rJ_a6I" title="善用ChatGPT，讓專業的你更亮眼 ! | 生成式AI主題論壇" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>**Paper:** [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)<br>


<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

