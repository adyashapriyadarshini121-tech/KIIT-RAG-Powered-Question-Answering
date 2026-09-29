import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="KIIT RAG-Powered Question Answering",
    page_icon="🎓"
)


# -----------------------------
# Title
# -----------------------------
st.title("🎓 KIIT RAG-Powered Question Answering")

st.write(
    "Ask questions about KIIT academic regulations, student guidelines, "
    "academic calendar, and syllabus."
)


# -----------------------------
# Groq API
# -----------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not available.")
    st.stop()

client = Groq(api_key=groq_api_key)


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Connect to ChromaDB
# -----------------------------
# The chroma_db folder must be present
# inside the GitHub project repository.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

if not os.path.exists(CHROMA_PATH):
    st.error(
        "ChromaDB database not found. "
        "Please make sure the 'chroma_db' folder is uploaded to the GitHub repository."
    )
    st.stop()


chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = chroma_client.get_or_create_collection(
    name="kiit_knowledge_base"
)


# -----------------------------
# Check Knowledge Base
# -----------------------------
document_count = collection.count()

if document_count == 0:
    st.error(
        "The KIIT knowledge base is empty. "
        "Please upload the correct ChromaDB database."
    )
    st.stop()


# -----------------------------
# RAG Function
# -----------------------------
def ask_rag(question, top_k=5):

    # Create embedding for user question
    query_embedding = embedding_model.encode([question])[0]

    # Search ChromaDB
    search_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    documents = search_results["documents"][0]
    metadatas = search_results["metadatas"][0]

    # Combine retrieved documents
    context = "\n\n".join(documents)

    # Prompt for Groq
    prompt = f"""
You are a helpful KIIT academic assistant.

Answer the question using ONLY the information provided
in the context below.

Do not make up information.

If the answer is not available in the context, say:

"I could not find this information in the provided KIIT documents."

Context:
{context}

Question:
{question}

Answer:
"""

    # Generate answer
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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
# User Interface
# -----------------------------
question = st.text_input(
    "Enter your question:"
)


# -----------------------------
# Ask Question Button
# -----------------------------
if st.button("Ask Question"):

    if question.strip():

        with st.spinner("Searching KIIT documents..."):

            try:
                answer, sources = ask_rag(question)

                # -----------------------------
                # Display Answer
                # -----------------------------
                st.subheader("Answer")
                st.write(answer)


                # -----------------------------
                # Display Sources
                # -----------------------------
                st.subheader("Sources")

                unique_sources = []

                for metadata in sources:

                    source = metadata.get(
                        "source",
                        "Unknown source"
                    )

                    if source not in unique_sources:
                        unique_sources.append(source)


                if unique_sources:

                    for source in unique_sources:
                        st.write("📄", source)

                else:
                    st.write("No source information available.")

            except Exception as e:

                st.error(
                    "An error occurred while processing your question."
                )

                st.exception(e)

    else:

        st.warning("Please enter a question.")
