# NALAPRO · Text Classification on 20 Newsgroups

Master's NLP project comparing five modeling paradigms on the
20-Newsgroups text-classification benchmark:

1. **Task 1** — 2-layer ReLU NN on Word2Vec & TF-IDF embeddings + a custom experiment
2. **Task 2** — Fine-tuned BERT-base
3. **Task 3** — BERT with masked-language-modeling domain-adaptation, then classification fine-tuning
4. **Task 4** — Llama-3-8B-Instruct, zero-shot and few-shot (k = 1, 3, 5)
5. **Bonus** — Llama-3 fine-tuned with QLoRA (4-bit NF4 + LoRA on every linear layer)

## Experiment tracking


A flat list of every individual run URL lives in [`wandb_links.md`](wandb_links.md).

## Tools used

- **PyTorch** — tensor library and CUDA backend
- **Hugging Face Transformers** — BERT and Llama-3 model classes, tokenizers, training APIs
- **PEFT** — LoRA adapter wiring
- **bitsandbytes** — 4-bit NF4 quantization and paged AdamW optimizer
- **TRL** — `SFTTrainer` for supervised fine-tuning of causal LMs
- **scikit-learn** — `fetch_20newsgroups`, classification metrics
- **gensim** — Word2Vec training
- **NLTK** — tokenization / stopword removal for the classical preprocessing branch
- **Weights & Biases** — experiment tracking (required by project rubric §5.b)
- **matplotlib / seaborn / pandas** — plots and tables

### AI assistance disclosure (project rubric §5.c)

Parts of this codebase — primarily boilerplate, plotting code, and docstrings —
were drafted with help from an AI assistant (Anthropic Claude). All modeling
choices, hyperparameter settings, and result interpretations are mine, and every
line of the code is annotated so I can defend it.

### Source citation

The QLoRA fine-tuning pipeline (Task 4 bonus) is adapted from the DataCamp
tutorial *"Fine-Tuning Llama 3.1 for Text Classification"* by Abid Ali Awan
(https://www.datacamp.com/tutorial/fine-tuning-llama-3-1). The same 4-bit-NF4 +
LoRA-on-all-linears + SFT recipe is reused for a different dataset (20 Newsgroups
instead of customer-complaint text) and integrated with the rest of the project's
evaluation framework.

## Setup

```bash
git clone https://github.com/<you>/nalapro-project.git
cd nalapro-project
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Secrets needed at runtime

- `HF_TOKEN` — Hugging Face Hub token with **"Read public gated repos"** scope, for Llama-3-Instruct
- `WANDB_API_KEY` — from <https://wandb.ai/authorize>

Set them as environment variables, Kaggle Secrets, or Colab Secrets (the
notebook auto-detects). Llama-3-8B-Instruct also requires accepting Meta's
license at <https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct>.

## Reproducing the experiments

The notebooks are independent — you can run any one of them on its own.
Expected wall-clock on a single T4 GPU (Colab Pro / Kaggle):

| Notebook | Time |
|---|---|
| Task 1 | 10–15 min |
| Task 2 | 25–40 min |
| Task 3 | 50–70 min |
| Task 4 + Bonus | ~90 min |



## Dataset

20 Newsgroups is fetched at runtime via `sklearn.datasets.fetch_20newsgroups`
and is **not committed to this repository** (per rubric instruction #3).
