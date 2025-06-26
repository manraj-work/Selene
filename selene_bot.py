import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Set up Azure OpenAI credentials
azure_endpoint = os.getenv("AZURE_API_BASE")
api_key = os.getenv("AZURE_API_KEY")
api_version = os.getenv("AZURE_API_VERSION")

# File paths
PDF_PATH = "data/Sexual_Offences_Act_2003.pdf"
CHROMA_DB_PATH = "chroma_db"  # Directory to save ChromaDB
COLLECTION_NAME = "sexual_offences_act"

def setup_embeddings():
    """Initialize embeddings object"""
    return AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment="text-embedding-ada-002"
    )

def create_vector_store():
    """Create new vector store from PDF"""
    print("Creating new vector store from PDF...")
    
    # Load and process PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings
    embeddings = setup_embeddings()
    
    # Create ChromaDB vector store
    print("Creating embeddings and storing in ChromaDB... (This costs money)")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print(f"Vector store created and saved to {CHROMA_DB_PATH}")
    return vector_store

def load_existing_vector_store():
    """Load existing ChromaDB vector store"""
    print("Loading existing ChromaDB vector store...")
    
    embeddings = setup_embeddings()
    
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    print("Existing vector store loaded successfully!")
    return vector_store

def setup_vector_store():
    """Setup vector store - load existing or create new"""
    
    # Check if ChromaDB already exists
    if os.path.exists(CHROMA_DB_PATH):
        try:
            return load_existing_vector_store()
        except Exception as e:
            print(f"Error loading existing vector store: {e}")
            print("Creating new vector store...")
            return create_vector_store()
    else:
        return create_vector_store()

def setup_rag_chain(vector_store):
    """Setup the RAG chain"""
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment="gpt-35-turbo",
        temperature=0.1
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
You are Selene, an empathetic and helpful support assistant specialising in the Sexual Offences Act 2003 designed to aid those who have been victims of sexual offences.
Answer the question based on the following context from the Act:

{context}

Question: {input}

Provide a clear and accurate answer based on the legal text provided. Be compassionate and supportive in your response.
""")

    # Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return rag_chain

def reset_vector_store():
    """Delete existing vector store"""
    import shutil
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print("Existing vector store deleted. Next run will create a new one.")
    else:
        print("No existing vector store found.")

def chat():
    """Main chat interface"""
    print("Selene RAG Chatbot with ChromaDB initializing...")
    print("Type 'exit' to end, 'reset' to recreate vector store")
    
    # Setup vector store
    vector_store = setup_vector_store()
    
    # Setup RAG chain
    rag_chain = setup_rag_chain(vector_store)
    
    print("Ready! Ask questions about the Sexual Offences Act 2003.")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        print(f"Question: {user_input}")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "reset":
            reset_vector_store()
            print("Please restart the program to recreate the vector store.")
            continue
        
        try:
            response = rag_chain.invoke({"input": user_input})
            print(f"Selene: {response['answer']}")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    chat()