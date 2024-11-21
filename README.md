# RAG with LLMs

## Project Overview

The "RAG with LLMs" project involves the use of retrieval-augmented generation (RAG) techniques in conjunction with large language models (LLMs) using LangChain and OpenAI's API. The project focuses on processing and retrieving information from PDF documents, and enabling conversational AI capabilities.

### Notebooks Included:

- **llm_rag.ipynb**: This notebook demonstrates the integration of LangChain libraries to load and process PDF documents, split text, store vectors, and interact with a language model for information retrieval and conversation.
- **chatbot_rag.ipynb**: This notebook further explores the chatbot functionalities, using LangChain for embeddings, text splitting, and vector storage to enhance conversational AI interactions.

### Key Features:

- **PDF Processing**: Load and process PDF documents using `PyPDFLoader` for document ingestion.
- **Text Splitting**: Use recursive character text splitting for efficient text chunking.
- **Vector Storage and Retrieval**: Store processed document vectors using `Chroma` and retrieve them for conversational AI.
- **Conversational AI**: Build a conversational retrieval chain using OpenAI's language models to answer user queries based on document context.
- **Interactive UI**: Utilize the `panel` library to create an interactive interface for file input and chatbot management.

## Key Technologies and Libraries

- **OpenAI API**: For leveraging large language models to process and generate text.
- **LangChain**: A library that provides tools for working with language models, including document loading, text splitting, embeddings, and vector storage.
- **Chroma**: A vector storage solution used to store and retrieve document vectors.
- **Panel**: A library for creating interactive web applications in Python.

## Usage Examples

### Setting Up the Environment

Ensure you have the necessary environment variables set before starting:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "<your_api_key_here>"
```

### Running the `llm_rag` Notebook

1. **Clean Existing Data**:  
   To start with a fresh environment, remove old database files:
   ```python
   !rm -rf ./docs/chroma
   ```

2. **Load PDF and Process Text**:  
   Load your PDF documents and split the text into manageable chunks.

### Running the `chatbot_rag` Notebook

1. **Load Libraries**:  
   Import necessary libraries for embeddings, text splitting, and vector storage:
   ```python
   from langchain.embeddings.openai import OpenAIEmbeddings
   from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
   from langchain.vectorstores import DocArrayInMemorySearch
   ```

2. **Initialize Panel for Interactive UI**:  
   Set up the `panel` extension to manage the UI components:
   ```python
   import panel as pn
   import param

   pn.extension()
   ```

3. **File Input Widget**:  
   Create a file input widget for uploading PDF documents:
   ```python
   file_input = pn.widgets.FileInput(accept='.pdf')
   ```
