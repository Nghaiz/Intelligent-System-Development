# Intelligent System Development

Coursework for **Intelligent System Development** at the Posts and Telecommunications Institute of Technology (PTIT), Hanoi.

**Student:** Nguyen Duy Nghia · B23DCCN600 · Class D23CTPM01
**Instructor:** Assoc. Prof. Dr. Tran Dinh Que
**Term:** Semester 1, academic year 2026–2027

---

## Assignments

| # | Topic | Highlights | Source | Report |
|---|---|---|---|---|
| **01** | From Data Representation to a First Intelligent System | Two systems (diabetes diagnosis, Vietnamese real-estate pricing), five ML models each, four controlled experiments, a Neo4j knowledge graph, and a Streamlit web app | [`src/Assignment 01`](src/Assignment%2001) | [PDF](Report/Assignment%2001/A01_CT_nghiand.600.pdf) |
| **02** | Deployable Intelligent Systems | Three applications (diabetes, house prices, e-commerce customer behavior), each with a REST API, a web UI and a mobile UI, running side by side on three ports | [`src/Assignment 02`](src/Assignment%2002) | [PDF](Report/Assignment%2002/A02_CT_nghiand.600.pdf) |
| **03** | Neural Networks and Representation Learning | A deep neural network written entirely in NumPy (no TensorFlow or PyTorch), scaled up to three large systems with web and mobile front ends | [`src/Assignment 03`](src/Assignment%2003) | [PDF](Report/Assignment%2003/A03_CT_nghiand.600.pdf) |
| **04** | Convolutional Neural Networks | 1D and 2D convolution implemented three times (pure NumPy, PyTorch, TensorFlow) across five data domains (text, tabular, grayscale and color images), plus an MLP-vs-CNN comparison and PCA analysis of the latent space | [`src/Assignment 04`](src/Assignment%2004) | [PDF](Report/Assignment%2004/A04_CT_nghiand.600.pdf) |
| **05** | CNNs as Compositions of Functions | CNN concepts written as NumPy functions, nine architectures from LeNet to ViT, a baseline CNN and three improved models (VGG, ResNet, SE-ResNet) trained on CIFAR-10, CIFAR-100 and Diabetes; LaTeX report with every figure drawn in TikZ | [`src/Assignment 05`](src/Assignment%2005) | [PDF](Report/Assignment%2005/A05_CT_nghiand.600.pdf) |
| **06** | Recurrent Neural Networks on Sequence Data | Simple RNN, LSTM, GRU and BiLSTM in PyTorch and Keras (16 models) on AMZN stock prices 1997–2024 and listening logs of 100,000 KKBox users; theory implemented in NumPy, cross-framework benchmarking, Flask web app | [`src/Assignment 06`](src/Assignment%2006) | [PDF](Report/Assignment%2006/A06_CT_nghiand.600.pdf) |
| **Agent 01** | Discovering AI Agents + Q-Learning | A 12×12 maze environment formulated as an MDP, a tabular Q-Learning agent (epsilon-greedy, Bellman updates) verified against BFS, reward curves and Q-table visualization in a generated Jupyter notebook | [`src/Agent 01`](src/Agent%2001) | [PDF](Report/Agent%2001/agent01_CT_07_nghiand.pdf) |

## Essays

| # | Topic | Highlights | Source | Report |
|---|---|---|---|---|
| **TL01** | Classical ML, CNNs and RNNs in Intelligent Systems | History of AI; Linear/Logistic Regression, SVM, Random Forest and MLP (NumPy, Keras, PyTorch) on diabetes, Vietnamese house prices and customer reviews; Custom CNN, VGG-11, MobileNetV2 and ResNet-18 on SVHN and EuroSAT; RNN/LSTM/GRU/BiLSTM on VN30 stocks and KKBox; six-module Flask web app | [`src/TieuLuan 01`](src/TieuLuan%2001) | [PDF](Report/TieuLuan%2001/tieuluan01_CT_07_nghiand.600.pdf) |
| **TL02** | Knowledge Graphs, RAG and AI Agents | Ontologies, RDFS inference, RDF export and SPARQL; 5 chunking strategies × 5 retrievers, HyDE, Multi-Query, reranking and GraphRAG; a type 2 diabetes KG with an MLP risk model (100,000 records); a signed causal finance KG with BiLSTM forecasting of VN30 and SJC gold; a ReAct agent with 7 tools, citation and number checking, and a Flask web app, all on a local Qwen2.5-1.5B | [`src/TieuLuan 02`](src/TieuLuan%2002) | [PDF](Report/TieuLuan%2002/tieuluan02_CT_07_nghiand.pdf) |
| **TL03** | Analysis of Intelligent Systems | History of AI; ML from scratch in NumPy plus K-means and PCA; CNNs on MNIST, Fashion-MNIST, Rock/Paper/Scissors and CIFAR-10; RNNs on e-commerce revenue, AAPL and GLD; KG and RAG across three domains with six retrievers and a routing agent; GCN, GraphSAGE and GAT on Cora/CiteSeer; Direct, CoT, PAL and multi-agent reasoning on GSM8K | [`src/TieuLuan 03`](src/TieuLuan%2003) | [PDF](Report/TieuLuan%2003/tieuluan03_CT_07_nghiand.pdf) |

Each project has its own README in its source folder with instructions for rebuilding it from scratch.

---

## Repository layout

```
Intelligent-System-Development/
├── src/                        Source code, one folder per project
│   ├── Assignment 01/            src/ app/ data/ models/ notebooks/ outputs/ figures/ tests/
│   ├── Assignment 02/            diabetes/ house_price/ customer_behavior/
│   ├── Assignment 03/            diabetes_baseline/ diabetes_large/ house_price_large/
│   │                             customer_comments/ knowledge_graph/
│   ├── Assignment 04/            customer_comments/ diabetes/ house_price/
│   │                             mnist/ cifar10/ mlp_vs_cnn/ analysis/
│   ├── Assignment 05/            a05/ (shared package) notebooks/ models/ outputs/
│   ├── Assignment 06/            a06/ (shared package) app/ (web app) notebooks/ models/ outputs/
│   ├── Agent 01/                 ag1/ (shared package) notebooks/ scripts/ outputs/
│   ├── TieuLuan 01/              tl/ (shared package) scripts/ app/ data/ models/ outputs/
│   ├── TieuLuan 02/              tl2/ (shared package) scripts/ app/ data/ models/ outputs/
│   └── TieuLuan 03/              tl3/ (shared package) scripts/ app/ data/ models/ outputs/
│
├── Report/                     Reports (PDF + the sources that build them)
│   ├── Assignment 01/            LaTeX: main.tex, chapters/, refs.bib, figures/
│   ├── Assignment 02–04/         Python: build_report.py, chapters_*.py, figures/
│   ├── Assignment 05–06/         LaTeX + TikZ: build.py, main.tex, chapters/, tikz/
│   ├── Agent 01/                 LaTeX + TikZ: build.py, main.tex, chapters/, tikz/
│   └── TieuLuan 01–03/           LaTeX + TikZ: build.py, main.tex, chapters/, tikz/
│
└── TAILIEU/                    Course materials
    ├── 1. BASIC MACHINE LEARNING ... .pdf
    ├── 2. Deep learning with python ... .pdf
    └── slide-assign_01 … 04/     Lecture slides and weekly assignment briefs
```

Every project uses the same folder name under `src/` and `Report/`, so moving between the code and its report only means switching the root folder.

### Adding a new project

A new project needs three folders with the same name and no changes anywhere else:

```
src/Assignment 07/            source code
Report/Assignment 07/         report
TAILIEU/slide-assign_07/      slides and assignment brief
```

---

## Conventions

**Subfolder names** inside each system under `src/` are consistent across projects:

| Folder | Contents |
|---|---|
| `data/` | Raw data and external validation data |
| `model/` or `models/` | Trained models (`.joblib`, `.npz`, `.pt`) with `metadata.json` |
| `notebook/` or `notebooks/` | Analysis and training notebooks |
| `reports/figures/` | Charts generated by the notebooks |
| `api/` | REST services |
| `web/`, `mobile/` | User interfaces |

**Report file names:** `A<number>_CT_nghiand.600.pdf` for assignments, `tieuluan<number>_CT_07_nghiand*.pdf` for essays.

**Environment:** each project has its own `requirements.txt`. Create a virtual environment inside that project's folder:

```bash
cd "src/Assignment 03"
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Environment variables:** projects that use Neo4j read their connection settings from `.env`. Copy `.env.example` to `.env` and fill in your own values. `.env` is listed in `.gitignore` and is never committed.

**Large files:** some datasets and the HuggingFace models used by TL02 and TL03 (about 8 GB) are not in git. Each project's README lists the download scripts that fetch them.

---

## Notes

Everything runs on a local machine; there is no public deployment. Rebuild instructions are in each project's README.
