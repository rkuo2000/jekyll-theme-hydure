---
layout: post
title: LLM Fine-Tuning
author: [Richard Kuo]
category: [Lecture]
tags: [jekyll, ai]
---

Introduction to LLM Fine-Tuning

---
## LLM FineTuning

### [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)
![](https://eugeneyan.com/assets/lora.jpg)

---
### [FlashAttention](https://pypi.org/project/flash-attn/)
**Paper:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)<br>
**Blog:** [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)<br>
![](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*i4tDdwgvGtXuTIyJpFUn8A.png)

---
### Few-Shot PEFT
**Paper:** [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)<br>

---
### PEFT (Parameter-Efficient Fine-Tuning)
**Paper:** [Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment](https://arxiv.org/abs/2312.12148)<br>
**Code:** [https://github.com/huggingface/peft](https://github.com/huggingface/peft)<br>

---
### QLoRA
**Paper:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)<br>
**Code:** [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)<br>

---
### AWQ
**Paper:** [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)<br>
**Code:** [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)<br>
![](https://github.com/mit-han-lab/llm-awq/raw/main/figures/overview.png)

---
### Soft-prompt Tuning
**Paper:** [Soft-prompt Tuning for Large Language Models to Evaluate Bias](https://arxiv.org/abs/2306.04735)<br>

---
### [Platypus](https://platypus-llm.github.io/)
**Paper:** [Platypus: Quick, Cheap, and Powerful Refinement of LLMs](https://arxiv.org/abs/2308.07317)<br>
**Code:** [https://github.com/arielnlee/Platypus/](https://github.com/arielnlee/Platypus/)<br>

---
### LLM Lingua
**Paper: [LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models](https://arxiv.org/abs/2310.05736)<br>
**Code: [https://github.com/microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/llm-lingua](https://www.kaggle.com/code/rkuo2000/llm-lingua)<br>
![](https://github.com/microsoft/LLMLingua/raw/main/images/LLMLingua.png)

---
### LongLoRA
**Code:** [https://github.com/dvlab-research/LongLoRA](https://github.com/dvlab-research/LongLoRA)<br>
[2023.11.19] We release a new version of LongAlpaca models, LongAlpaca-7B-16k, LongAlpaca-7B-16k, and LongAlpaca-7B-16k. <br>
![](https://github.com/dvlab-research/LongLoRA/raw/main/imgs/LongAlpaca.png)

---
### [GPTCache](https://github.com/zilliztech/GPTCache)
![](https://eugeneyan.com/assets/gptcache.jpg)

---
### localLLM 
**Code:** [https://github.com/ykhli/local-ai-stack](https://github.com/ykhli/local-ai-stack)<br>
🦙 Inference: Ollama<br>
💻 VectorDB: Supabase pgvector<br>
🧠 LLM Orchestration: Langchain.js<br>
🖼️ App logic: Next.js<br>
🧮 Embeddings generation: Transformer.js and all-MiniLM-L6-v2<br>

---
### AirLLM
**Blog:** [Unbelievable! Run 70B LLM Inference on a Single 4GB GPU with This NEW Technique](https://ai.gopubby.com/unbelievable-run-70b-llm-inference-on-a-single-4gb-gpu-with-this-new-technique-93e2057c7eeb)<br>
**Code:** [https://github.com/lyogavin/Anima/tree/main/air_llm](https://github.com/lyogavin/Anima/tree/main/air_llm)<br>

---
### PowerInfer
**Paper:** [PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU](https://arxiv.org/abs/2312.12456)<br>
**Code:** [https://github.com/SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)<br>
**Blog:** [2080 Ti就能跑70B大模型，上交大新框架让LLM推理增速11倍](https://mp.weixin.qq.com/s/GnEK3xE5EhR5N9Mzs3tOtA)<br>

https://github.com/SJTU-IPADS/PowerInfer/assets/34213478/fe441a42-5fce-448b-a3e5-ea4abb43ba23
PowerInfer v.s. llama.cpp on a single RTX 4090(24G) running Falcon(ReLU)-40B-FP16 with a 11x speedup!<br>
Both PowerInfer and llama.cpp were running on the same hardware and fully utilized VRAM on RTX 4090.

---
### Eagle
**Paper:** [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)<br>
**Code:** [https://github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)<br>
**Kaggle:** [https://www.kaggle.com/code/rkuo2000/eagle-llm](https://www.kaggle.com/code/rkuo2000/eagle-llm)<br>

---
### Era of 1-bit LLMs
**Paper:** [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)<br>
**Blog:** [No more Floating Points, The Era of 1.58-bit Large Language Models](https://medium.com/ai-insights-cobet/no-more-floating-points-the-era-of-1-58-bit-large-language-models-b9805879ac0a)<br>
![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*Skb5lzSS3JUlxnw2jmANSg.png)
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*jt9iF_OXLhqJQhfwJCArBg.png)

---
### GaLore
**Paper:** [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)<br>
**Code:** [https://github.com/jiaweizzhao/galore](https://github.com/jiaweizzhao/galore)<br>
To train a 7B model with a single GPU such as NVIDIA RTX 4090, all you need to do is to specify --optimizer=galore_adamw8bit_per_layer, which enables GaLoreAdamW8bit with per-layer weight updates. With activation checkpointing, you can maintain a batch size of 16 tested on NVIDIA RTX 4090.

---
### LlamaFactory
**Paper:** [LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models
](https://arxiv.org/abs/2403.13372)<br>
**Code:** [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)<br>

---
### PEFT for Vision Model
**Paper:** [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242)
![](https://arxiv.org/html/2402.02242v1/x1.png)
![](https://arxiv.org/html/2402.02242v1/x2.png)

---
### ReFT
representation fine-tuning (ReFT) library, a Powerful, Parameter-Efficient, and Interpretable fine-tuning method<br>
**Paper:** [ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592)<br>
**Code:** [https://github.com/stanfordnlp/pyreft](https://github.com/stanfordnlp/pyreft)<br>

---
### ORPO
**model:** [kaist-ai/mistral-orpo-beta](https://huggingface.co/kaist-ai/mistral-orpo-beta)<br>
**Paper:** [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)<br>
**Code:** [https://github.com/xfactlab/orpo](https://github.com/xfactlab/orpo)<br>
**Blog:** [Fine-tune Llama 3 with ORPO](https://huggingface.co/blog/mlabonne/orpo-llama-3)<br>

---
### FlagEmbedding
**model:** [namespace-Pt/Llama-3-8B-Instruct-80K-QLoRA](https://huggingface.co/namespace-Pt/Llama-3-8B-Instruct-80K-QLoRA)<br>
**Paper:** [Extending Llama-3's Context Ten-Fold Overnight](https://arxiv.org/abs/2404.19553)<br>
**Code:** [https://github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)<br>

---
### Memory-Efficient Inference: Smaller KV Cache with Cross-Layer Attention
**Paper:** [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981)<br>
![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*T4NveajaY-nxdvVj.png)

---
## Federated Learning
**Blog:** [聯盟式學習 (Federated Learning)](https://medium.com/sherry-ai/%E8%81%AF%E7%9B%9F%E5%BC%8F%E5%AD%B8%E7%BF%92-federated-learning-b4cc5af7a9c0)<br>
**Paper:** [Federated Machine Learning: Concept and Applications](https://arxiv.org/abs/1902.04885)<br>
Architecture for a horizontal federated learning system<br>
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*IEZFpz4MWZmKboZSGb9OmQ.png)
Architecture for a vertical federated learning system<br>
![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*IEZFpz4MWZmKboZSGb9OmQ.png)

### FATE-LLM
FATE-LLM is a framework to support federated learning for large language models(LLMs) and small language models(SLMs).<br>
**Code:** [https://github.com/FederatedAI/FATE-LLM](https://github.com/FederatedAI/FATE-LLM)<br>
![](https://github.com/FederatedAI/FATE-LLM/raw/main/doc/images/fate-llm-show.png)

### OpenFedLLM
**Paper:** [OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning](https://arxiv.org/abs/2402.06954)<br>
**Code:** [https://github.com/rui-ye/openfedllm](https://github.com/rui-ye/openfedllm)<br>
![](https://github.com/rui-ye/OpenFedLLM/blob/main/doc/assets/openfedllm-intro.png?raw=true)

<br>
<br>

*This site was last updated {{ site.time | date: "%B %d, %Y" }}.*

