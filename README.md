# 🔍 ResearchRover

**ResearchRover** is a full-stack, AI-powered **Research Paper Recommendation System** that combines **Graph Neural Networks (GNN)** and **Semantic Embeddings (SBERT)** to recommend relevant academic papers. It includes an interactive **Streamlit UI** for real-time use.

---

## 📌 Features

- 🧠 **GNN-Based Recommendations** using paper similarity graph
- 🗣️ **SBERT Embeddings** for deep semantic understanding of abstracts
- 🧪 **Hybrid Recommender**: combines structural (GNN) + semantic (SBERT)
- 🌐 **Streamlit Web UI** to explore recommendations
- 🔗 **Link Prediction** using GCN + MLP to suggest hidden connections

---

## 📁 Project Structure

ResearchRover/
│
├── streamlit_app.py # Streamlit UI frontend
├── sample_1000_arxiv.csv # Cleaned and preprocessed paper data (1000 samples)
├── gnn_embeddings.pt # Learned GNN embeddings for each paper
├── sbert_embeddings.pt # Precomputed SBERT embeddings
├── requirements.txt # Required Python packages
└── README.md # This file

---

## ⚙️ How It Works

### 🧾 Step 1: Data Preparation

- Load and clean two datasets: `arxiv_data_210930-054931.csv` and `arvix_data.csv`
- Standardize column names (`terms`, `title`, `abstract`)
- Combine datasets and remove duplicates/nulls
- Sample 1000 papers for demo scale

### 🔁 Step 2: Graph Construction

- Compute **TF-IDF embeddings** of abstracts
- Calculate **cosine similarity** between papers
- Construct an undirected **similarity graph (NetworkX)**:
  - Nodes: papers
  - Edges: if similarity > 0.3

### 🧠 Step 3: Graph Neural Network (GNN)

- Train a 2-layer **GCN (Graph Convolutional Network)** using `torch_geometric`
- Node features: TF-IDF vectors
- Output: 64-dim GNN embeddings per paper

```python
class GCNRecommender(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        ...
🔗 Step 4: Link Prediction
Positive edges = actual graph edges

Negative edges = random non-edges

Train an MLP (LinkPredictor) to classify whether an edge should exist

python
Copy
Edit
class LinkPredictor(nn.Module):
    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        ...
💬 Step 5: SBERT Semantic Embeddings
Use sentence-transformers to encode abstract text

all-MiniLM-L6-v2 model outputs 384-dim embedding per abstract

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
sbert_embeddings = sbert_model.encode(abstracts)
🔀 Step 6: Hybrid Recommender
Combine GNN + SBERT embeddings:

hybrid_embeddings = [GNN | SBERT]  --> shape: [1000, 448]
Use cosine similarity over hybrid vector to recommend papers

🌐 Streamlit UI
Features:
Select paper from dropdown or paste a new abstract

View top-5 hybrid recommended papers

Supports cold-start (no graph) queries

Usage:

streamlit run streamlit_app.py
