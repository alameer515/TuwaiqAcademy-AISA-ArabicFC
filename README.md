---
license: apache-2.0
task_categories:
  - text-generation
  - question-answering
language:
  - ar
tags:
  - function-calling
  - tool-use
  - agentic
  - arabic
  - reasoning
  - shared-task
  - arabicnlp2026
  - emnlp2026
  - dialect
  - msa
  - gulf
  - egyptian
  - levantine
  - maghrebi
  - aisa
pretty_name: AISA-ArabicFC (ArabicNLP 2026 Shared Task)
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
      - split: dev
        path: data/dev-*.parquet
---

<div align="center">

# AISA-ArabicFC

### Arabic Function Calling for Agentic AI Systems

*The first open benchmark for tool-use in Arabic — across five dialects, eight real-world domains, and 27 structured tools.*

<br/>

[![ArabicNLP 2026](https://img.shields.io/badge/ArabicNLP-2026-4f29b7?style=for-the-badge&logoColor=white)](https://arabicnlp2026.sigarab.org/)
[![EMNLP 2026](https://img.shields.io/badge/co--located%20with-EMNLP%202026-57e3d8?style=for-the-badge)](https://2026.emnlp.org/)
[![Shared Task Page](https://img.shields.io/badge/🤗-Shared%20Task%20Page-f4a664?style=for-the-badge)](https://huggingface.co/spaces/Omartificial-Intelligence-Space/AISA-ArabicFC-Shared-Task)
[![License](https://img.shields.io/badge/License-Apache%202.0-80d883?style=for-the-badge)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tuwaiq Academy](https://img.shields.io/badge/Built%20by-Tuwaiq%20Academy-262626?style=for-the-badge)](https://tuwaiq.edu.sa)

<br/>

**12,125 queries** · **5 dialects** · **8 domains** · **27 tools** · **12K reasoning traces**

📅 **Test set releases July 20, 2026** · 🏛️ **Budapest · Oct 24–29, 2026**

</div>

---

## 🆕 Update — Data **v1.2** & fair scoring (June 2026)

> **Argument scoring is now robust to surface form.** A correct answer written in natural Arabic is no longer marked wrong over formatting. The evaluator normalizes **both** your prediction and the gold before comparing — it equates *different writings of the same value*, never different values:
>
> - **Numbers** — `5000` = `5000.0` = `٥٠٠٠`
> - **Arabic orthography** — `الإمارات` = `الامارات` (diacritics · alef · ya · ta-marbuta)
> - **Number words** — `٤ بيتزا` = `اربع بيتزا`
> - **Lists** (`items`, `country`) — order-independent: `مصر و الإمارات` = `مصر، الإمارات`
> - **Aliases** — `الأردن` = `Jordan` · `ريال سعودي` = `SAR` · `الإنجليزية` = `en`
>
> The **train + dev gold has been canonicalized** to one convention (numbers → int, countries/cities → Arabic, currencies/languages → ISO, enums → canonical token). **You may submit values in Arabic _or_ the canonical form — both score identically.**
>
> **v1.2** additionally repairs rows whose gold call was missing its arguments — those are now populated (recovered from each example's reasoning trace) across train, dev, and test. If you downloaded or submitted on an earlier copy, please **re-download and re-submit** to get your fair score. Full rules live in the leaderboard's **How it works** tab.

---

## 🎯 The challenge

> A user in Cairo says *عايز أحجز دكتور*.
>
> A user in Riyadh says *أبي أحجز موعد عند الدكتور*.
>
> Same intent. Same tool. **The model has to know.**

Function calling — the bridge between language models and the real world — has exploded for English. Models book flights, query databases, chain tools into agents. For **Arabic**, with its rich dialectal variation and morphological complexity, this capability barely exists.

**AISA-ArabicFC closes that gap.** Given an Arabic query in any of five dialects and a candidate set of tool definitions, your system must decide whether a tool call is needed, select the correct function, and extract structured JSON arguments — optionally producing an Arabic reasoning trace.

---

## ⚡ At a glance

| | |
|---|---|
| 🗂️&nbsp;**Splits** | `train` **10,550** · `dev` **545** · `test` (blind, July 20) |
| 🗣️&nbsp;**Dialects** | MSA · Gulf · Egyptian · Levantine · Maghrebi |
| 🏛️&nbsp;**Domains** | Healthcare · Banking · Government · Islamic · Travel · Weather · E-commerce · Utilities |
| 🛠️&nbsp;**Tools** | 27 total (20 called + 7 distractors), 4 candidates per query |
| 🧠&nbsp;**Reasoning** | 12,000 Arabic `<think>` traces for Track B |
| ❌&nbsp;**Negatives** | No-call cases included → hallucination is in-scope |
| 🎯&nbsp;**Headline metric** | **Argument Exact Match (ArgEM)** — pilot SOTA: **0.541** |

---

## 🚀 Quick start

```python
from datasets import load_dataset

ds = load_dataset("TuwaiqAcademy/AISA-ArabicFC")

# DatasetDict({
#   train: Dataset(num_rows=10550, features=[...])
#   dev:   Dataset(num_rows=545,   features=[...])
# })

ex = ds["train"][0]
print(ex["text"])              # full prompt with tool declarations
print(ex["tool_called"])       # gold tool name (or 'none' for negatives)
print(ex["tools_sampled"])     # the 4 candidate tools shown
print(ex["dialect"])           # msa / gulf / egyptian / levantine / maghrebi
```

> 💡 **Tip** — `ex["text"]` is the ready-to-use formatted prompt. For custom prompting, use `ex["messages"]` (structured) and `ex["tools_sampled"]` (the 4 candidates).

---

## 📋 What's released

<div align="center">

| Split | Rows | Positive (call) | Negative (no-call) | Status |
|:---|:---:|:---:|:---:|:---:|
| `train` | **10,550** | 10,500 | 50 | ✅ Public |
| `dev` | **545** | 500 | 45 | ✅ Public |
| `test` | — | — | — | 🔒 Released **July 20, 2026** |

</div>

### Dialect distribution

| Dialect | Train | Dev | Share |
|:---|---:|---:|---:|
| 🟢 Modern Standard Arabic (MSA) | 6,151 | 323 | **58.3%** |
| 🟣 Levantine | 1,784 | 75 | 16.9% |
| 🟠 Egyptian | 1,283 | 66 | 12.2% |
| 🔵 Gulf | 1,190 | 75 | 11.3% |
| 🟡 Maghrebi | 142 | 6 | 1.3% |

### Domains × tools

| Domain | Tools |
|:---|:---|
| 🏥 **Healthcare** | `book_doctor_appointment` · `search_medications` · `check_insurance_coverage` |
| 🏦 **Banking & Finance** | `transfer_money` · `convert_currency` · `calculate_customs` |
| 🏛️ **Government** | `check_visa_status` · `check_iqama_status` · `check_traffic_violations` |
| 🕌 **Islamic Services** | `get_qibla_direction` · `calculate_zakat` · `search_quran` · `calculate_inheritance` |
| ✈️ **Travel** | `search_hotels` · `search_umrah_packages` |
| 🌤️ **Weather & Environment** | `get_weather` · `get_air_quality` |
| 🛒 **E-commerce** | `compare_prices` · `order_food` |
| 🔧 **Utilities** | `translate_text` · `calculate_end_of_service` |

---

## 🧬 Schema

Each row is a complete function-calling example:

| Field | Type | Description |
|:---|:---|:---|
| `text` | `string` | Full formatted prompt — system instructions, current time, tool declarations, user message |
| `requires_function` | `bool` | Whether the query needs a tool call (`true`) or is a no-call negative (`false`) |
| `tool_called` | `string` | Gold function name — `"none"` for negatives |
| `messages` | `list[dict]` | Structured conversation: `developer` / `user` / `assistant` roles, with optional `think` for Track B reasoning |
| `tools` | `list[dict]` | All 27 tool schemas (the global tool registry) |
| `tools_sampled` | `list[dict]` | The 4 candidate tools shown for this query |
| `negative_category` | `string \| null` | Sub-category for negatives (chitchat / ambiguous / out-of-scope) |
| `dialect` | `string` | `msa`, `gulf`, `egyptian`, `levantine`, or `maghrebi` |

---

## 🏁 Tracks & scoring

<div align="center">

| Track | Goal | Scoring |
|:---:|:---|:---|
| **A** | Core function calling | `0.40 · FnAcc + 0.60 · ArgEM` |
| **B** | Reasoning-augmented (`<think>` traces) | `0.30 · FnAcc + 0.50 · ArgEM + 0.20 · ThinkRate` |
| **C** | Cross-dialect robustness | Per-dialect FnAcc + ArgEM · gap (max − min) — diagnostic |

</div>

**FnAcc** — exact match of the function name. `"none"` for negatives, so this metric folds in hallucination too.
**ArgEM** — exact match of all predicted argument key-value pairs, under the **normalization rules** above (Arabic / ISO / number forms treated as equal). **The headline.**
**ThinkRate** *(Track B)* — did the system emit an Arabic `<think>` trace before the call?

---

## 🥇 Pilot baselines — the bar is wide open

| System | FnAcc | **ArgEM ★** | Overall (A) | Overall (B) |
|:---|:---:|:---:|:---:|:---:|
| 🥇 **AISA-Think** · Gemma 3 (270M) + LoRA · reasoning-augmented | **0.982** | **0.541** | **0.717** | **0.739** |
| GPT-4o · zero-shot | 0.927 | 0.070 | 0.413 | 0.313 |
| GPT-4o · 3-shot | 0.854 | 0.122 | 0.415 | 0.317 |
| Random | 0.047 | 0.033 | 0.039 | 0.031 |

*ThinkRate (Track B): AISA-Think **0.868**, others **0.000**.*

> 💡 **Two findings worth your attention**
> 1. **Argument extraction is the wall.** Best system: 0.541 ArgEM. GPT-4o: 0.070. There's enormous room to improve.
> 2. **A 270M Arabic-fine-tuned model beats GPT-4o** on every metric. Task-specific training > raw scale.

Baseline model: 👉 [**TuwaiqAcademy/AISA-AR-FunctionCall-Think**](https://huggingface.co/TuwaiqAcademy/AISA-AR-FunctionCall-Think)

---

## 📅 Timeline

| Date | Milestone |
|:---|:---|
| **May 16, 2026** | 🚀 Task launch · website live · registration opens |
| **June 1, 2026** | 📦 Train + dev data · baseline code · evaluation scripts released |
| **July 20, 2026** | 🔒 Registration deadline · blind test data released |
| **July 30, 2026** | 🏆 Final results released |
| **August 22, 2026** | 📄 Camera-ready system description papers due |
| **September 1, 2026** | 📚 Shared task overview paper due |
| **September 10, 2026** | ✍️ Conference camera-ready deadline |
| **October 24–29, 2026** | 🎤 ArabicNLP 2026 / EMNLP 2026 · Budapest, Hungary |

---

## 📚 Required citations

> System description papers using this dataset **must cite all three** works.

<details open>
<summary><strong>📌 1 · Shared Task</strong> — Najar et al. (2026)</summary>

```bibtex
@inproceedings{najar2026aisaarabicfc,
  title     = {{AISA-ArabicFC}: Arabic Function Calling for Agentic AI Systems},
  author    = {Najar, Omar and Al Khalifa, Mohammed and Alzaharani, Saeed},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing Conference (ArabicNLP 2026)},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics}
}
```
</details>

<details>
<summary><strong>📌 2 · AISA Architecture</strong> — Nacar, Deema &amp; Mohammed (2026)</summary>

```bibtex
@misc{nacar2026aisa,
  title     = {{AISA}: A Unified Architecture for Agentic AI Systems},
  author    = {Nacar, Omer and Deema, A. and Mohammed, A.},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18161880},
  url       = {https://doi.org/10.5281/zenodo.18161880}
}
```
</details>

<details>
<summary><strong>📌 3 · Methodology</strong> — Nacar et al. (2026), arXiv</summary>

```bibtex
@article{nacar2026language,
  title   = {From Language to Action in Arabic: Reliable Structured Tool Calling via Data-Centric Fine-Tuning},
  author  = {Nacar, Omer and Alquffari, Deema and Alsharideh, Saleh and AlOtaibi, Adeem and Alabdulkarim, Abdulaziz and Alhazmi, Leen and Alomar, Nada and Alzubaidi, Wareef and Alsultan, Nada and Alrabghi, Ahmed and others},
  journal = {arXiv preprint arXiv:2603.16901},
  year    = {2026}
}
```
</details>

---

## 👥 Organizers

<div align="center">

| | | |
|:---:|:---:|:---:|
| **Omer Nacar** | **Mohammed Al Khalifa** | **Saeed Alzaharani** |
| Tuwaiq Academy | Tuwaiq Academy | Tuwaiq Academy |

📧 **Get in touch** → [`trdc@tuwaiq.edu.sa`](mailto:trdc@tuwaiq.edu.sa)
🌐 **Shared task page** → [Hugging Face Space](https://huggingface.co/spaces/Omartificial-Intelligence-Space/AISA-ArabicFC-Shared-Task)

</div>

---

## ⚖️ License

Released under **Apache 2.0**. Use it, fine-tune on it, ship it — just cite the works above in any derived research.

---

<div align="center" dir="rtl" lang="ar">

## 🌍 العربية

### مهمة مشتركة لاستدعاء الدوال العربية في أنظمة الذكاء الاصطناعي التوكيلي

**AISA-ArabicFC** هو أول معيار عربي مفتوح لاستدعاء الدوال (Function Calling) ضمن المنظومات الذكية التوكيلية، عبر **خمس لهجات** (الفصحى، الخليجية، المصرية، الشامية، المغاربية) و**ثمانية مجالات خدمية حقيقية** (الصحة، التمويل، الحكومة، الخدمات الإسلامية، السفر، الطقس، التجارة، الخدمات العامة).

تضم البيانات **١٠٬٥٥٠ عينة تدريب** و**٥٤٥ عينة تطوير**، مع آثار استدلال عربية لمسار التفكير قبل الاستدعاء، إضافةً إلى مجموعة اختبار مغلقة تُطلَق في **٢٠ يوليو ٢٠٢٦**.

أُطلقت هذه المهمة كجزء من **المؤتمر الرابع لمعالجة اللغة العربية (ArabicNLP 2026)** المُنعقد ضمن **EMNLP 2026** في بودابست، بتنظيم من **أكاديمية طويق**.

</div>

---

<div align="center">

**Built with ♥ for the Arabic NLP community.**

</div>
