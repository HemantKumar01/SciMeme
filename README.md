# 🧠 SciMemeX: Automatic Generation of Scientific Memes from Research Articles

> **Paper**: *Breaking Bored? SciMemeX: Towards Automatic Generation of Scientific Memes from Research Articles*  
> **Framework**: Modular Multi-Agent System for Contrastive Meme Generation  
> **Key Components**: LLM-based Summarization, Template Selection, Meme Generation, and Evaluation

---

## 🌟 Overview

**SciMemeX** introduces a novel paradigm for **science communication** by automatically generating **scientific memes** from research papers.  
It leverages a **multi-agent framework** combining **idea extraction**, **template selection**, and **feedback-guided contrastive refinement** to produce memes that are:

- **Faithful** to the scientific content (scientific fidelity)  
- **Clear** in their communicative intent (clarity & interpretability)  
- **Engaging** for broader scientific audiences (engagement potential)

The system is designed to **translate dense academic insights into accessible, humorous, and memorable visual narratives**, fostering wider engagement beyond academia.

---

## 🧩 Core Components

| Module | Description |
|--------|--------------|
| **Concisio Agent** | Extracts comparative insights: *prior work vs. new contributions* (~100-word summary). |
| **Template Selector Agent (TSA)** | Chooses optimal meme templates for given paper ideas. |
| **Generator Agent** | Produces creative meme captions conditioned on paper content and selected templates. |
| **Contrastive Generator** | Refines memes iteratively using feedback from prior best/worst generations. |
| **Evaluation Agents** | Three LLM-as-judge evaluators for *Fidelity*, *Clarity*, and *Engagement*. |

The web implementation also generates normalized text-box coordinates while producing each caption, then renders the winning meme locally with Pillow.

---

## 🧠 Evaluation Dimensions

SciMemeX employs three complementary automatic evaluation metrics implemented in `eval_metrics.py`:

| Metric | Scale | Definition |
|---------|--------|------------|
| **Scientific Fidelity** | 1–5 | Faithfulness to the paper’s core contribution. |
| **Clarity (FRI Score)** | 1–3 | How easily a graduate-level audience grasps the meme’s message. |
| **Engagement Potential** | 1–5 | Humor, shareability, and non-offensiveness. |

---

### 2️⃣ Set Up Virtual Environment
```
python3 -m venv scimemex_env
source scimemex_env/bin/activate```


### Set Up Virtual Environment
```pip install -r requirements.txt```


### 🔧 Configuration
Edit config.yaml as follows:

api_keys:
  openai: "YOUR_OPENAI_API_KEY"

models:
  default: gpt-4o

temperatures:
  fidelity: 0.2
  clarity: 0.5
  engagement: 0.2

data:
  path: "./papers_json/"
  amount: 1000

---
### 🚀 Usage

```python eval/eval_run.py```

Run Full Multi-Agent Meme Generation

```python SciMemeX.py --config_path config.yaml --output results/output.json```

### Web application

The web pipeline implements the new exploration–exploitation search directly from an uploaded PDF:

1. PDF extraction, Innovative Reflections, and Concisio.
2. Initial template selection, multimodal caption generation, and parallel fidelity/clarity/engagement evaluation.
3. Up to six contrastive iterations. Every iteration reselects templates and uses the previous best/worst candidates plus critic feedback.
4. All-time-best selection, final evaluation, and coordinate-aware local PNG rendering.

Each search round uses one generation request for all selected images. Caption text and normalized `0..1000` placement coordinates are produced together. The API key is request-scoped and is never written to the JSON output.

The web interface uses three server-configured open-source models through Amazon Bedrock. Put the shared Bedrock API key in a root-level `.env` file (this file is ignored by Git):

```dotenv
OPEN_SOURCE_API_KEY=your-api-key
BEDROCK_REGION=us-east-1
```

Available models are GLM 5 (`zai.glm-5`), Qwen3 VL 235B A22B (`qwen.qwen3-vl-235b-a22b-instruct`), and Llama 3 70B Instruct (`meta.llama3-70b-instruct-v1:0`). GLM 5 is the default for pipeline stages, while Qwen3 VL is the default critic. `BEDROCK_REGION` is optional and defaults to `us-east-1`. Users can instead select the **OpenAI API key** tab and supply their own key; that key remains request-scoped and is not stored.

For a hosted deployment, configure `OPEN_SOURCE_API_KEY` as a service environment variable or secret instead of baking `.env` into the container image.

```bash
python3 -m venv .venv-web
source .venv-web/bin/activate
pip install -r requirements-web.txt
uvicorn webapp.app:app --reload
```

Open `http://127.0.0.1:8000`. The interface shows the current stage, progress, and final meme. Intermediate outputs remain collapsed but can be inspected or downloaded as one JSON file. Six iterations reproduce the paper's search depth; fewer iterations reduce API usage.


### 🧾 Example Output

{
  "paper_id": "2305.12345",
  "final_best_meme": "Using recurrence for sequence modeling → Using self-attention for everything.",
  "final_best_score": 10.82,
  "iteration_scores": [
    {"iteration": 1, "fidelity": 4.5, "clarity": 2.1, "engagement": 4.0, "average": 10.6}
  ]
}
