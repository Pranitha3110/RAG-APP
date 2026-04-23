import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A Bot")
st.title("DMV Assistant")

@st.cache_resource
def load_vectorstore():
    # Use the same model used in process_docs.py [cite: 188]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# 1. REQUIRED SAFETY PROMPT [cite: 192-194]
template = """You are a helpful assistant. Answer the user's question using ONLY the context provided below. 
If the answer is not contained in the context, respond with: 
"I'm sorry, I am only authorized to talk about the provided document." 
Do not use outside knowledge.

Context: {context}
Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4", temperature=0) # Deterministic output [cite: 107, 118]

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # 2. RETRIEVE TOP 4 CHUNKS [cite: 195]
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": prompt_template}
)

question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Searching..."):
        response = qa_chain.invoke(question)
        st.success("Answer:")
        st.write(response["result"])

        # 3. DISPLAY 3 SOURCES [cite: 196]
        with st.expander("View source passages"):
            docs = vectorstore.similarity_search(question, k=3) 
            for i, doc in enumerate(docs):
                st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                st.text(doc.page_content)
                st.divider()
