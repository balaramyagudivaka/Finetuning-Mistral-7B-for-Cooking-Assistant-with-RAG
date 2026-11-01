# 🍳 Fine-Tuning Mistral-7B with QLoRA + RAG: A Cooking Assistant

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-blueviolet)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)

A domain-specific AI assistant that answers practical cooking questions. It fine-tunes **Mistral-7B-Instruct v0.2** with **QLoRA** (4-bit quantization + LoRA adapters) on a single free **Tesla T4 GPU**, then adds **Retrieval-Augmented Generation (RAG)** so answers are grounded in a curated knowledge base.

## ✨ Highlights

- **Fine-tuned a 7-billion-parameter model on a 16 GB GPU** by loading it in 4-bit NF4 and training LoRA adapters: only **13.6M parameters (0.19%)** were trained.
- **Held-out perplexity dropped from 49.05 to 19.76 (−59.7%)** after fine-tuning, measured with the same evaluation function before and after training.
- **RAG pipeline**: `all-MiniLM-L6-v2` sentence embeddings (384-dim) + a **FAISS** index retrieve the top-3 most relevant passages, which are inserted into the prompt before generation.
- **Side-by-side comparison** of base model, fine-tuned model and fine-tuned + RAG answers, plus retrieval-distance diagnostics.
- **Hand-written training loop** (AdamW, linear warmup, gradient clipping, per-epoch validation) alongside an optional Hugging Face `SFTTrainer` path.

---

## 🏗️ How It Works

```
                   ┌──────────────────────────┐
  User question ──▶│ Embed (MiniLM-L6-v2)     │
                   └────────────┬─────────────┘
                                ▼
                   ┌──────────────────────────┐
                   │ FAISS search (top-3)     │◀── 97 cooking knowledge passages
                   └────────────┬─────────────┘
                                ▼
                   ┌──────────────────────────┐
                   │ Prompt = question +      │
                   │ retrieved passages       │
                   └────────────┬─────────────┘
                                ▼
                   ┌──────────────────────────┐
                   │ Mistral-7B (4-bit)       │
                   │ + fine-tuned LoRA        │──▶ Grounded answer
                   └──────────────────────────┘
```

---

## 📊 Results

| Metric (15 held-out passages) | Base Mistral-7B | Fine-tuned |
|---|---|---|
| Perplexity | 49.05 | **19.76** |
| Negative log-likelihood | 3.89 | **2.98** |

Retrieval on unseen test questions: mean L2 distance 0.95 to the top-3 passages (lower is closer).

**Limitations, honestly stated:** the dataset is small (82 training / 15 validation passages), and training ran for 100 epochs, so training loss fell far below validation loss, which is a sign of overfitting. Answer quality was judged by reading sample outputs rather than with an automated benchmark.

---

## ⚙️ Training Setup

| Setting | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Quantization | 4-bit NF4, double quantization, fp16 compute (bitsandbytes) |
| LoRA | rank 16, alpha 32, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optimizer | AdamW, learning rate 2e-4, 50 warmup steps, grad-norm clip 0.3 |
| Batch size / max length | 4 / 512 tokens |
| Hardware | 1 × NVIDIA Tesla T4 (Google Colab) |

---

## 🧠 Dataset

`cooking_qa.json`: **97 curated cooking knowledge passages** covering techniques, food safety, baking science and frying/sautéing, wrapped in Mistral's `[INST] … [/INST]` instruction format. Split 85% / 15% into training and validation.

---

## 🚀 Run It

The notebook is designed for **Google Colab** with a T4 GPU (free tier works).

1. Open `mistral7b_qlora_rag_cooking_assistant.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Upload `cooking_qa.json` to the Colab file panel (the notebook reads `/content/cooking_qa.json`).
3. Run all cells. To run locally instead: `pip install -r requirements.txt` (requires an NVIDIA GPU with ~16 GB memory).

---

## 💬 Example Questions

- *How do I make a perfect risotto?*
- *What's the best way to caramelize onions?*
- *How do I prevent my scrambled eggs from being watery?*

---

## 🌱 Next Steps

- Expand the dataset with synthetic Q&A pairs and add early stopping to reduce overfitting
- Add an automated evaluation (held-out question set, RAG faithfulness scores)
- Deploy a Gradio demo on Hugging Face Spaces
- Move notebook code into reusable Python modules

---

## 📁 Files

```
├── mistral7b_qlora_rag_cooking_assistant.ipynb  # Full pipeline: data → QLoRA training → evaluation → RAG chat
├── cooking_qa.json                              # Knowledge base / training data (97 passages)
└── requirements.txt                             # Python dependencies
```

## 📚 References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)
- Reimers & Gurevych, *Sentence-BERT* (2019)
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS, 2017)
- Mistral AI, *Mistral-7B-Instruct-v0.2* model card

---

Built by [Bala Ramya Gudivaka](https://www.linkedin.com/in/bala-ramya-gudivaka-54731424b/) · Data Science MS, Stony Brook University
