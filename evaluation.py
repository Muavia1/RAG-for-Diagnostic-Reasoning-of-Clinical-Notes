from sentence_transformers import util
import numpy as np
import streamlit as st

def calculate_hit_rate(retriever, query, expected_docs, k=3):
    """
    Custom hit rate calculation for top-k retrieved documents.
    Args:
        retriever: FAISS retriever (st.session_state.vectors.as_retriever())
        query: User's input query
        expected_docs: List of expected document contents
        k: Top-k documents to consider
    Returns:
        hit_rate: Percentage of expected docs found in top-k results
    """
    retrieved_docs = retriever.get_relevant_documents(query, k=k)
    retrieved_contents = [doc.page_content for doc in retrieved_docs]
    
    hits = 0
    for expected in expected_docs:
        if any(expected in retrieved for retrieved in retrieved_contents):
            hits += 1
    
    return hits / len(expected_docs) if expected_docs else 0.0

def evaluate_rag_response(response, embeddings):
    scores = {}
    
    # 1. Faithfulness: Answer-Context Similarity
    answer_embed = embeddings.embed_query(response["answer"])
    context_embeds = [embeddings.embed_query(doc.page_content) for doc in response["context"]]
    similarities = [util.cos_sim(answer_embed, ctx_embed).item() for ctx_embed in context_embeds]
    scores["faithfulness"] = float(np.mean(similarities)) if similarities else 0.0

    # 2. Custom Hit Rate Calculation
    retriever = st.session_state.vectors.as_retriever()
    scores["hit_rate"] = calculate_hit_rate(
        retriever,
        query=response["input"],
        expected_docs=[doc.page_content for doc in response["context"]],
        k=3
    )
    
    return scores