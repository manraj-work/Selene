import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Your existing configuration
azure_endpoint = os.getenv("AZURE_API_BASE")
api_key = os.getenv("AZURE_API_KEY")
api_version = os.getenv("AZURE_API_VERSION")

PDF_PATH = "data/Sexual_Offences_Act_2003.pdf"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "sexual_offences_act"

# Global variable for the RAG chain
rag_chain = None

# Your existing functions (copied from your original code)
def setup_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment="text-embedding-ada-002"
    )

def create_vector_store():
    print("Creating new vector store from PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = setup_embeddings()
    print("Creating embeddings and storing in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print(f"Vector store created and saved to {CHROMA_DB_PATH}")
    return vector_store

def load_existing_vector_store():
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment="gpt-35-turbo",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template("""
You are Selene, an empathetic and helpful support assistant specialising in the Sexual Offences Act 2003 designed to aid those who have been victims of sexual offences.
Answer the question based on the following context from the Act:

{context}

Question: {input}

Provide a clear and accurate answer based on the legal text provided. Be compassionate and supportive in your response.
""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain

def initialize_rag():
    global rag_chain
    if rag_chain is None:
        print("Initializing RAG system...")
        vector_store = setup_vector_store()
        rag_chain = setup_rag_chain(vector_store)
        print("RAG system ready!")

# Web routes
@app.route('/')
def home():
    # This will serve our HTML page
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        initialize_rag()
        
        data = request.get_json()
        user_message = data['message']
        
        response = rag_chain.invoke({"input": user_message})
        
        return jsonify({
            "response": response['answer'],
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "error": "Sorry, I encountered an error. Please try again.",
            "status": "error"
        }), 500

# Simple HTML template (embedded in the Python file)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Selene - Support Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: #667eea;
            color: white;
        }
        
        .message.selene .message-content {
            background: white;
            border: 1px solid #e0e0e0;
            color: #333;
        }
        
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
        }
        
        .input-area input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #ddd;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
        }
        
        .input-area input:focus {
            border-color: #667eea;
        }
        
        .send-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .send-btn:hover {
            background: #5a6fd8;
        }
        
        .send-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .loading {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .loading-dot {
            width: 8px;
            height: 8px;
            background: #667eea;
            border-radius: 50%;
            animation: loading 1.4s infinite ease-in-out;
        }
        
        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes loading {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <h1>💜 Selene</h1>
            <p>Sexual Offences Act 2003 Support Assistant</p>
        </div>
        
        <div class="messages" id="messages">
            <div class="message selene">
                <div class="message-content">
                    Hello, I'm Selene. I'm here to help you with questions about the Sexual Offences Act 2003. Please feel free to ask me anything, and I'll do my best to provide you with clear, supportive guidance.
                </div>
            </div>
        </div>
        
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="Ask me about the Sexual Offences Act 2003..." />
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'selene'}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addLoadingMessage() {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message selene';
            messageDiv.id = 'loading-message';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = '<div class="loading"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div></div>';
            
            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function removeLoadingMessage() {
            const loadingMsg = document.getElementById('loading-message');
            if (loadingMsg) {
                loadingMsg.remove();
            }
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            
            // Disable input
            sendBtn.disabled = true;
            messageInput.disabled = true;
            
            // Add loading message
            addLoadingMessage();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                removeLoadingMessage();
                
                if (data.status === 'success') {
                    addMessage(data.response, false);
                } else {
                    addMessage('Sorry, I encountered an error. Please try again.', false);
                }
                
            } catch (error) {
                removeLoadingMessage();
                addMessage('Sorry, I could not connect to the server. Please try again.', false);
            }
            
            // Re-enable input
            sendBtn.disabled = false;
            messageInput.disabled = false;
            messageInput.focus();
        }
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Focus on input when page loads
        messageInput.focus();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting Selene Web App...")
    print("Once running, open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)