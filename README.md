# 🚗 DMV Assistant (RAG Application)

This is a **Retrieval-Augmented Generation (RAG)** application built with Python and Streamlit. It allows users to ask specific questions about the California Driver's Handbook and receive accurate answers based directly on the official document.

## 🌟 Key Features
* **Document Intelligence:** Uses LangChain to process and "read" the DMV PDF.
* **Vector Search:** Powered by FAISS to find the most relevant sections of the handbook instantly.
* **AI Responses:** Integrated with OpenAI's GPT-4o for natural, conversational answers.
* **Source Transparency:** The app displays the exact page numbers and text snippets used to generate the answer.
* **Safety Guardrails:** Custom prompts ensure the assistant stays on-topic and politely declines non-DMV questions.

## 🛠️ Built With
* **Streamlit:** For the web interface.
* **LangChain:** For the RAG logic and chain management.
* **OpenAI:** For the Large Language Model and Embeddings.
* **FAISS:** For efficient local vector storage.
* **PyPDF:** For document parsing.

## 🚀 Live Demo
You can try out the live app here: https://rag-app-f6iznzwro6zyjqr7scm36w.streamlit.app/
