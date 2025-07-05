import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("sample_1000_arxiv.csv")  # save sample_df beforehand
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    gnn_emb = torch.load("gnn_embeddings.pt")  # save paper_embeddings earlier
    sbert_emb = torch.load("sbert_embeddings.pt")
    hybrid_emb = torch.cat([gnn_emb, sbert_emb], dim=1)
    return df, sbert_model, hybrid_emb

df, sbert_model, hybrid_embeddings = load_data()

# UI
st.title("📚 Research Paper Recommender")
st.subheader("Using GNN + SBERT Hybrid Embeddings")

option = st.selectbox("Choose Input Method", ["Select from Dataset", "Paste Your Abstract"])

if option == "Select from Dataset":
    paper_titles = df['title'].tolist()
    selected_title = st.selectbox("Choose a Paper", paper_titles)
    idx = df[df['title'] == selected_title].index[0]
    input_emb = hybrid_embeddings[idx].unsqueeze(0)
else:
    input_abstract = st.text_area("Paste Abstract Below")
    if st.button("Generate Recommendations"):
        input_sbert = sbert_model.encode([input_abstract], convert_to_tensor=True)
        input_sbert = torch.tensor(input_sbert, dtype=torch.float)
        # fake GNN as zeros for cold start
        fake_gnn = torch.zeros((1, hybrid_embeddings.shape[1] - input_sbert.shape[1]))
        input_emb = torch.cat([fake_gnn, input_sbert], dim=1)

        sims = cosine_similarity(input_emb, hybrid_embeddings)[0]
        top_indices = sims.argsort()[-6:][::-1][1:]
        st.subheader("📑 Recommended Papers:")
        for i in top_indices:
            st.markdown(f"- **{df.loc[i, 'title']}**")
