# Finetuning-Mistral-7B-for-Cooking-Assistant-with-RAG
Fine-tuned Mistral-7B-Instruct v0.2 as a domain-specific cooking assistant using LoRA (PEFT) on 4-bit quantized weights to meet limited GPU resource constraints.

Curated and preprocessed a specialized cooking_qa dataset (97 paragraphs) covering cooking techniques, food safety, and baking science, formatted in instruction–completion style.

Implemented QLoRA with NF4 quantization and trained only attention projection layers (q_proj, k_proj, v_proj, o_proj) to reduce trainable parameters.

Built a custom PyTorch training loop with AdamW optimizer, linear warmup scheduler, and validation tracking using loss and perplexity metrics.

Achieved a significant perplexity reduction from ~49 to ~2.36, demonstrating effective domain adaptation on a small dataset.

Conducted human evaluation using realistic cooking queries, showing improved factual accuracy, step-by-step guidance, and reduced hallucinations compared to the base model.

Designed and integrated a Retrieval-Augmented Generation (RAG) pipeline using Sentence-BERT embeddings and FAISS for context-aware responses.

Compared standard fine-tuned generation vs RAG-enhanced inference, showing improved answer grounding for less-common cooking questions.

Developed an interactive CLI-based cooking chatbot, supporting multi-turn conversations with detailed procedural instructions.
