# Finetuning-Mistral-7B-for-Cooking-Assistant-with-RAG
# 🍳 Fine-Tuning Mistral-7B for a Cooking Assistant with RAG

A domain-specific **AI cooking assistant** built by fine-tuning **Mistral-7B-Instruct** using **QLoRA** and enhanced with **Retrieval-Augmented Generation (RAG)**. The system answers practical cooking questions with grounded, step-by-step explanations.

---

## 🚀 Project Highlights
- Fine-tuned **Mistral-7B-Instruct v0.2** using **LoRA (PEFT)** on **4-bit quantized weights**
- Built under limited GPU resources while training only a small subset of parameters
- Integrated **RAG (Sentence-BERT + FAISS)** for context-aware responses
- Achieved strong performance gains on a small, curated dataset

---

## 📊 Results
- **Validation Perplexity:** reduced from **~49 → ~2.36**
- Improved factual accuracy and reduced hallucinations
- RAG further enhanced answers for less-common cooking queries

---

## 🧠 Dataset
- **97 curated cooking knowledge paragraphs**
- Topics include:
  - Cooking techniques
  - Food safety
  - Baking science
  - Frying & sautéing methods
- Formatted using **instruction–completion** style prompts

---

## 🛠️ Technical Stack
- **Model:** Mistral-7B-Instruct v0.2  
- **Fine-Tuning:** QLoRA (NF4, 4-bit quantization)  
- **Frameworks:** PyTorch, Hugging Face Transformers, PEFT  
- **Retrieval:** Sentence-BERT, FAISS  
- **Optimization:** AdamW with Linear Warmup Scheduler  

---

## 🔄 Pipeline Overview
1. Dataset preprocessing and instruction formatting  
2. LoRA fine-tuning on quantized Mistral-7B  
3. Validation using loss and perplexity  
4. RAG pipeline for retrieval-based inference  
5. Interactive chatbot-style generation  

---

## 🧪 Evaluation
- **Quantitative:** Validation loss and perplexity  
- **Qualitative:** Human evaluation using realistic cooking queries  
- Compared base model vs fine-tuned vs RAG-enhanced outputs  

---

## 💬 Example Queries
- *How can I fix risotto that turned out too mushy?*  
- *What is the safest way to reheat leftover rice?*  
- *How do I get chewier chocolate chip cookies?*  

---

## 🌱 Future Improvements
- Expand dataset size
- Generate synthetic Q&A pairs
- Improve multi-turn conversation handling
- Add web or image-based retrieval

---

## 📚 References
- Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**
- Dettmers et al., **QLoRA**
- Reimers & Gurevych, **Sentence-BERT**
- Johnson et al., **FAISS**
- **Mistral-7B Model Card**

---

✨ *Built as part of an academic project exploring efficient fine-tuning and retrieval-augmented LLMs.*

