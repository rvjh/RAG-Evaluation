import streamlit as st
from core.rag import RAGPipeline
from evaluation.evaluation import evaluate_full

# ---------------------------
# App setup
# ---------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📚 RAG Chatbot + Evaluation Dashboard")

# ---------------------------
# Initialize RAG (cached)
# ---------------------------
@st.cache_resource
def load_rag():
    rag = RAGPipeline()
    rag.initialize()
    return rag

rag = load_rag()

# ---------------------------
# Sidebar - Controls
# ---------------------------
st.sidebar.header("⚙️ Controls")

run_eval = st.sidebar.button("🚀 Run Full Evaluation")

show_context = st.sidebar.toggle("Show Retrieved Context", value=True)

# ---------------------------
# Chat Section
# ---------------------------
st.subheader("💬 Ask the RAG System")

query = st.text_input("Enter your question")

if query:

    answer, docs = rag.run(query)

    st.markdown("### 🧠 Answer")
    st.success(answer)

    if show_context:
        st.markdown("### 📄 Retrieved Context")

        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.info(d.page_content)

# ---------------------------
# Evaluation Section
# ---------------------------
st.divider()
st.subheader("📊 Evaluation Dashboard")

if run_eval:

    with st.spinner("Running evaluation..."):

        results = evaluate_full(rag)

    # ---------------------------
    # Retrieval Metrics
    # ---------------------------
    st.markdown("### 🔎 Retrieval Metrics")

    retrieval = results["retrieval"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Precision@K", f"{retrieval['Precision@K']:.3f}")
    col2.metric("Recall@K", f"{retrieval['Recall@K']:.3f}")
    col3.metric("MRR", f"{retrieval['MRR']:.3f}")

    # ---------------------------
    # Generation Metrics (RAGAS)
    # ---------------------------
    st.markdown("### 🧠 Generation Metrics (RAGAS)")

    gen = results["generation"]

    try:
        st.write(gen)
    except Exception:
        st.warning("RAGAS output format not displayable directly.")

else:
    st.info("Click **Run Full Evaluation** to compute metrics.")