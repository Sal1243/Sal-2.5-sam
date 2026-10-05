# SAL-2.5-SAM

A coding-assistant fine-tuning pipeline built on **Qwen2.5-Coder-1.5B** with **QLoRA**, a **hard daily token budget of 262,144 tokens**, and an **OpenAI-style FastAPI server** that returns `HTTP 429` once the budget is used up.

Everything is driven from one Colab notebook: [`Sal_2_5_sam.ipynb`](Sal_2_5_sam.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sal1243/Sal-2.5-sam/blob/main/Sal_2_5_sam.ipynb)

---

## 1. What is in this repo

| Path | What it is |
|---|---|
| `Sal_2_5_sam.ipynb` | The full pipeline. Its cells write, run, and test every script below. |
| `README.md` | This file. |

The notebook **generates** these files in its Colab workspace:

| File | Role |
|---|---|
| `budget.py` | Daily token budget: reads/writes `budget.json`, resets each calendar day, `TokenBudgetCallback` stops training when the cap is hit. |
| `prep_data.py` | Loads a dataset, de-duplicates, length-filters, converts to ChatML, writes `prepared_dataset.jsonl`. |
| `train.py` | QLoRA (4-bit NF4, LoRA r=8, all attention + MLP projections) with `SFTTrainer`, packing, and the budget callback. Saves the adapter to `./qwen2.5-coder-qlora-adapter`. |
| `eval.py` | Runs 20 Python coding prompts, writes `eval_report.md` (time, tokens, tokens/sec, outputs). |
| `merge.py` | Merges the LoRA adapter into the base model (safetensors), then converts to GGUF with llama.cpp. |
| `app.py` | FastAPI server: `POST /v1/chat/completions`, OpenAI-shaped request and response, budget check, 429 on exhaustion. |
| `test_suite.py` | Checks a normal 200 response and a 429 when the budget is exhausted. |
| `Dockerfile`, `requirements.txt`, `MODEL_CARD.md` | Packaging and docs. |

---

## 2. Setup

**Requirements:** a GPU runtime (Colab T4 is enough for the 1.5B model), a Hugging Face token, and a GitHub token if you want to push.

1. Open the notebook with the Colab badge above.
2. **Runtime → Change runtime type → T4 GPU.**
3. In Colab **Secrets** (key icon), add `HF_TOKEN` and `GITHUB_TOKEN`.
4. Run cells top to bottom. **Skip the `microsoft/layoutlmv3-base` cell** (it is a document-layout model and is not used by this pipeline).
5. If imports break, pin the versions the notebook settled on:

```bash
pip install -q "transformers==4.49.0" "protobuf==5.29.3" "tokenizers>=0.21.0" peft trl bitsandbytes datasets accelerate fastapi uvicorn pydantic httpx jinja2
pip uninstall -y torchao   # avoids a PEFT version clash seen during eval
```

## 3. Run the pipeline

Once the scripts exist in your working folder:

```bash
python prep_data.py     # build prepared_dataset.jsonl
python train.py         # QLoRA fine-tune, stops at the daily cap
python eval.py          # writes eval_report.md
python merge.py         # merged model + GGUF
python test_suite.py    # API + budget tests
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 4. Use the model

**curl**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "SAL-2.5-SAM",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "Write a Python function to check if a number is prime."}
    ],
    "max_tokens": 256,
    "temperature": 0.2
  }'
```

**Python (OpenAI client)**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="SAL-2.5-SAM",
    messages=[{"role": "user", "content": "Reverse a linked list in Python."}],
    max_tokens=256,
)
print(r.choices[0].message.content, r.usage)
```

Defaults: `max_tokens=128`, `temperature=0.7`. Each response includes `usage` (prompt, completion, total tokens).

**GGUF (local)**: load `qwen2.5-coder-merged.gguf` in llama.cpp, Ollama, or LM Studio.

## 5. The daily token budget

- Limit: **262,144 tokens/day**, stored in `budget.json` as `{"date": "YYYY-MM-DD", "consumed_tokens": N}`.
- The count resets automatically when the date changes.
- **Training** raises `TokenBudgetExceededException` when the cap is crossed. **Serving** returns `429` with `Daily token budget of 262144 has been exhausted.`
- Training and serving share the same file, so they draw from one pool.
- Check usage: `cat budget.json`
- Docker: mount a volume so the count survives restarts.

```bash
docker build -t sal-2.5-sam .
docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data sal-2.5-sam
```

---

## 6. Current status and fixes needed

The pipeline runs end to end, but it is currently a **smoke test**, not a finished coding model. Fix these before relying on it:

| # | Issue | Fix |
|---|---|---|
| 1 | Only `README.md` and the notebook are on GitHub. The scripts exist only inside Colab (the push step failed). | Commit the scripts (see section 7). |
| 2 | Training runs **5 steps on 200 rows of `tatsu-lab/alpaca`**, which is general chat data, not code. | Use code data (e.g. `ise-uiuc/Magicoder-OSS-Instruct-75K`, `sahil2801/CodeAlpaca-20k`), raise `max_steps`/epochs, and spread training across days. |
| 3 | `prep_data.py` writes `<|im_end|` without the closing `>`. | Change to `<|im_end|>` in the ChatML template. |
| 4 | `app.py` loads the **base** `Qwen/Qwen2.5-Coder-1.5B`, not your merged model, so outputs are base-model text (some test replies were garbage). | Set `path_to_load = "./qwen2.5-coder-merged"` (or your Hugging Face repo). Consider the `-Instruct` base for chat. |
| 5 | The budget callback cannot see the batch in `on_step_end`, so it falls back to `batch_size × 512` (an estimate). | Count real tokens from the dataloader/collator (non-pad `input_ids`). |
| 6 | No checkpoint is saved when the cap stops training, so there is nothing to resume tomorrow. | Set `save_strategy="steps"` and resume with `resume_from_checkpoint=True`. |
| 7 | `test_suite.py` hardcodes the date `2026-10-04`, so the 429 test fails on any other day. | Use `str(date.today())`. |
| 8 | API reports the model as `qwen2.5-coder`. | Rename to `SAL-2.5-SAM` in `app.py`. |

## 7. Push the scripts to GitHub

Run in Colab (token comes from Colab Secrets and is not written into `.git/config`):

```python
from google.colab import userdata
import subprocess
token = userdata.get("GITHUB_TOKEN")
cmds = """
cd Sal-2.5-sam
printf "budget.json\nprepared_dataset.jsonl\nresults/\n*.gguf\nqwen2.5-coder-merged/\nqwen2.5-coder-qlora-adapter/\nllama.cpp/\n" > .gitignore
git add .
git commit -m "feat: SAL-2.5-SAM pipeline scripts"
"""
subprocess.run(cmds, shell=True, check=True)
subprocess.run(f"cd Sal-2.5-sam && git push https://{token}@github.com/Sal1243/Sal-2.5-sam.git main", shell=True, check=True)
```

Upload model weights to Hugging Face (not Git): `huggingface-cli upload Sal1243/SAL-2.5-SAM ./qwen2.5-coder-merged`.

**Security:** if you ever pasted a GitHub token into a notebook cell or remote URL, revoke it and create a new one.

## 8. License and base model

Base model: `Qwen/Qwen2.5-Coder-1.5B` (check its license on Hugging Face). Add your own license file before publishing.
