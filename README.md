# NLP Analysis of Australian Health Practitioner Disciplinary Tribunal Decisions

This repository contains the analysis code and configuration files for a natural language processing (NLP) study of 3,586 Australian health practitioner disciplinary tribunal decisions published between 1999 and 2025.

**Author:** Hayden Farquhar MBBS MPHTM
**ORCID:** [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)
**Contact:** hayden.farquhar@icloud.com

## Overview

This study applied NLP techniques to publicly available tribunal decisions sourced from the Australasian Legal Information Institute (AustLII) and the Open Australian Legal Corpus. The pipeline includes:

- **Data acquisition:** Web scraping from AustLII (with rate limiting), the Open Australian Legal Corpus (HuggingFace), and the Internet Archive Wayback Machine
- **Text preprocessing:** HTML/RTF/PDF extraction, section segmentation, metadata enrichment
- **Annotation:** Dual-coder protocol (human + LLM) with adjudication for a 300-decision gold standard
- **Classification:** TF-IDF + logistic regression multi-label classifier for 7 misconduct categories
- **Outcome extraction:** Regex-based extraction of 9 disciplinary outcome types
- **Named entity recognition:** Rule-based extraction of medications, medical conditions, and regulatory boards using spaCy
- **Topic modelling:** BERTopic with MiniLM sentence embeddings
- **Statistical analysis:** Chi-square tests, Fisher's exact tests, Mann-Kendall trend analysis, bootstrap confidence intervals

## Key Findings

- Professional conduct breaches were the most prevalent misconduct type (32.7%), followed by boundary violations (30.1%), clinical competence failures (25.6%), and prescribing misconduct (25.4%)
- Significant temporal increases were identified for boundary violations, professional conduct breaches, dishonesty/fraud, and communication failures
- Boundary violations were associated with higher odds of registration cancellation (OR = 1.47, p < 0.001)
- 64% of prescribing misconduct decisions involved opioid medications
- Significant variation in misconduct types and outcomes was observed across professions and jurisdictions

## Repository Structure

```
├── src/                        # Source code
│   ├── scraping/               # AustLII web scraping and corpus construction
│   ├── preprocessing/          # Text cleaning, annotation, metadata enrichment
│   ├── classification/         # Misconduct classifier and outcome extraction
│   ├── ner/                    # Named entity recognition (medications, conditions, boards)
│   ├── topic_modelling/        # BERTopic analysis
│   └── analysis/               # Statistical analysis and visualisation
├── config/                     # Configuration files
│   ├── taxonomy.json           # Misconduct, outcome, finding, and profession taxonomies
│   ├── tribunals.json          # Registry of 24 tribunals across 8 jurisdictions
│   └── label_studio_config.xml # Annotation interface configuration
├── notebooks/                  # Jupyter notebooks
│   └── legal_bert_training.ipynb  # Legal-BERT fine-tuning notebook (for Colab)
└── requirements/               # Python dependencies
    ├── requirements-scraping.txt   # Data acquisition dependencies
    └── requirements-ml.txt         # ML/NLP pipeline dependencies
```

## Reproducing the Analysis

### Prerequisites

- Python 3.12+ (3.12 recommended for ML dependencies)
- pandoc 3.x (optional, for PDF generation)

### Setup

Two virtual environments are used to avoid dependency conflicts:

```bash
# Scraping and preprocessing
python3.12 -m venv .venv-scraping
source .venv-scraping/bin/activate
pip install -r requirements/requirements-scraping.txt

# ML/NLP pipeline
python3.12 -m venv .venv-ml
source .venv-ml/bin/activate
pip install -r requirements/requirements-ml.txt
python -m spacy download en_core_web_sm
```

### Pipeline Steps

The pipeline is executed as a series of Python modules. Each step produces output files consumed by subsequent steps.

#### 1. Data Acquisition

```bash
# Scrape decisions from AustLII (requires ~3 hours due to rate limiting)
.venv-scraping/bin/python -m src.scraping.run_pipeline

# Merge all sources into a single corpus
.venv-scraping/bin/python -m src.scraping.merge_corpus
```

**Output:** `data/extracted/all_decisions.jsonl` (3,586 decisions)

#### 2. Preprocessing and Annotation

```bash
# Enrich metadata (profession, jurisdiction, dates)
.venv-scraping/bin/python -m src.preprocessing.metadata_enrichment

# Select stratified sample for annotation
.venv-scraping/bin/python -m src.preprocessing.sample_selection

# Run LLM annotation (requires Anthropic API key)
export ANTHROPIC_API_KEY=your_key_here
.venv-scraping/bin/python -m src.preprocessing.llm_annotation
```

#### 3. Classification

```bash
# Prepare train/val/test splits
.venv-ml/bin/python -m src.classification.data_preparation

# Train misconduct classifier
.venv-ml/bin/python -m src.classification.sklearn_classifier

# Extract outcomes using regex
.venv-ml/bin/python -m src.classification.outcome_extraction

# Apply classifier to full corpus
.venv-ml/bin/python -m src.classification.predict_full_corpus
```

**Output:** `data/extracted/misconduct_predictions_sklearn.jsonl`

#### 4. NER and Topic Modelling

```bash
# Extract medications, conditions, and boards
.venv-ml/bin/python -m src.ner.entity_extraction

# Run BERTopic
.venv-ml/bin/python -m src.topic_modelling.bertopic_analysis
```

#### 5. Statistical Analysis

```bash
# Generate all tables, figures, and statistical tests
.venv-ml/bin/python -m src.analysis.statistical_analysis

# Prescribing-specific analysis
.venv-ml/bin/python -m src.analysis.prescribing_analysis

# Validation and sensitivity analyses
.venv-ml/bin/python -m src.analysis.validation_analysis
```

**Output:** `outputs/tables/` (26 CSV tables), `outputs/figures/` (16 PNG figures), `outputs/statistics/` (statistical test results as JSON)

## Data Availability

The raw tribunal decisions are publicly available through [AustLII](https://www.austlii.edu.au/au/special/healthprac/) and the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus). The scraped corpus and annotation files are not included in this repository due to size constraints but can be reproduced by running the data acquisition pipeline.

## Citation

If you use this code or findings in your work, please cite:

> Farquhar H. NLP Analysis of Australian Health Practitioner Disciplinary Tribunal Decisions, 1999--2025. 2025. Available at: https://github.com/hayden-farquhar/Health-Practitioner-Misconduct-NLP

## Licence

This project is licensed under the MIT Licence. See [LICENCE](LICENCE) for details.

## Acknowledgements

- [AustLII](https://www.austlii.edu.au/) for maintaining open access to Australian legal materials
- The [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) for curated NSW tribunal decisions
- The [Internet Archive Wayback Machine](https://web.archive.org/) for enabling recovery of legacy tribunal decisions
