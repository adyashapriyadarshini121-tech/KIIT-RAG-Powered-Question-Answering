
import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="KIIT RAG-Powered Question Answering",
    page_icon="🎓"
)

st.title("🎓 KIIT RAG-Powered Question Answering")
st.write(
    "Ask questions about KIIT academic regulations, student guidelines, "
    "academic calendar, and syllabus."
)

# -----------------------------
# Groq API
# -----------------------------
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not available.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# -----------------------------
# Connect to ChromaDB
# -----------------------------
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="kiit_knowledge_base"
)

# -----------------------------
# RAG function
# -----------------------------
def ask_rag(question, top_k=5):

    query_embedding = embedding_model.encode([question])[0]

    search_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]

    context = "\n\n".join(documents)

    prompt = f"""
You are a helpful KIIT academic assistant.

Answer the question using ONLY the information provided
in the context below.

If the answer is not available in the context, say:

"I could not find this information in the provided KIIT documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    return answer, metadatas


# -----------------------------
# User interface
# -----------------------------
question = st.text_input(
    "Enter your question:"
)

if st.button("Ask Question"):

    if question.strip():

        with st.spinner("Searching KIIT documents..."):

            answer, sources = ask_rag(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        unique_sources = []

        for metadata in sources:
            source = metadata.get("source", "Unknown source")

            if source not in unique_sources:
                unique_sources.append(source)

        for source in unique_sources:
            st.write("📄", source)

    else:
        st.warning("Please enter a question.")
