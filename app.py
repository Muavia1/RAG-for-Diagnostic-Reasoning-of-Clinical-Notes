import streamlit as st
import os
import json
from pathlib import Path
import glob
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer  # New embedding model
from langchain.embeddings import HuggingFaceEmbeddings  # Wrapper for SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
from evaluation import evaluate_rag_response  
import time

load_dotenv()

# Initialize components
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# Using clinical-BERT for embeddings (no API needed)
embeddings = HuggingFaceEmbeddings(
    model_name="emilyalsentzer/Bio_ClinicalBERT",
    model_kwargs={"device": "cpu"}  # Use "cuda" if you have GPU
)

st.title("Clinical Diagnosis RAG Chat")

# ====== DATA LOADING ======
def load_clinical_data():
    """Load both flowcharts and patient cases"""
    docs = []
    
    # 1. Load diagnosis flowcharts
    for fpath in glob.glob("./Diagnosis_flowchart/*.json"):
        with open(fpath) as f:
            data = json.load(f)
            content = f"""
            DIAGNOSTIC FLOWCHART: {Path(fpath).stem}
            Diagnostic Path: {data['diagnostic']}
            Key Criteria: {data['knowledge']}
            """
            docs.append(Document(
                page_content=content,
                metadata={"source": fpath, "type": "flowchart"}
            ))
    
    # 2. Load patient cases
    for category_dir in glob.glob("./Finished/*"):
        if os.path.isdir(category_dir):
            for case_file in glob.glob(f"{category_dir}/*.json"):
                with open(case_file) as f:
                    case_data = json.load(f)
                    notes = "\n".join(
                        f"{k}: {v}" for k, v in case_data.items() 
                        if k.startswith("input")
                    )
                    docs.append(Document(
                        page_content=f"""
                        PATIENT CASE: {Path(case_file).stem}
                        Category: {Path(category_dir).name}
                        Notes: {notes}
                        """,
                        metadata={"source": case_file, "type": "patient_case"}
                    ))
    return docs

# ====== VECTOR STORE SETUP ======
def init_knowledge_base():
    if "vectors" not in st.session_state:
        with st.spinner("Building clinical knowledge base..."):
            # Load and split documents
            documents = load_clinical_data()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(documents)
            
            # Create vectorstore
            st.session_state.vectors = FAISS.from_documents(
                splits,
                embeddings  # Using local SentenceTransformer
            )
        st.success("Clinical RAG system ready!")

# ====== CHAT INTERFACE ======
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize button
if st.button("🛠️ Initialize System"):
    init_knowledge_base()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about clinical cases or guidelines"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if "vectors" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("Please initialize the system first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Consulting clinical knowledge..."):
                # RAG chain (identical to your original)
                prompt_template = ChatPromptTemplate.from_template("""
                Answer based on these clinical resources:
                <context>
                {context}
                </context>
                Question: {input}
                """)
                
                retriever = st.session_state.vectors.as_retriever()
                chain = create_retrieval_chain(
                    retriever,
                    create_stuff_documents_chain(llm, prompt_template)
                )
                
                response = chain.invoke({"input": prompt})
                st.markdown(response['answer'])
                
                #  # Evaluate the response (modify ground_truth if available)
                # eval_scores = evaluate_rag_response(response)
                # # Inside your chat assistant block (after response = chain.invoke(...))
                # if "evaluation" not in st.session_state:
                #     st.session_state.evaluation = []

               
                # st.session_state.evaluation.append(eval_scores)

               
                # Show sources
                with st.expander("Reference Materials"):
                    for i, doc in enumerate(response["context"]):
                        st.write(f"📄 {doc.metadata['source']}")
                        st.text(doc.page_content[:300] + "...")

                # --- Evaluation with Error Handling ---
                try:
                    eval_scores = evaluate_rag_response(response, embeddings)  # Pass embeddings explicitly
                    if "evaluation" not in st.session_state:
                        st.session_state.evaluation = []
                    st.session_state.evaluation.append(eval_scores)
                    
                    with st.expander("🧪 Evaluation Metrics"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Hit Rate (Top-3)", f"{eval_scores['hit_rate']:.2f}",
                                    help="% correct docs in top 3")
                        with col2:
                            st.metric("Faithfulness", f"{eval_scores['faithfulness']:.2f}",
                                    help="Answer-context alignment (0-1)")
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    if "vectors" not in st.session_state:  # Help users debug
                        st.warning("Did you initialize the system?")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['answer']
            })