import streamlit as st
import os
import pickle
import json
import requests
import zipfile
import tempfile
import shutil
from typing import List, Dict
from collections import OrderedDict
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Page configuration
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .config-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .source-link {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .answer-box {
        background: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration constants
GOOGLE_DRIVE_FOLDER_ID = "1EJ3q3gltaPW0RgC2kmQ32w52BBCd3-hr"
LOCAL_FAISS_PATH = "./faiss_index"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Source URL mapping
SOURCE_URL_MAP = {
    "genai-platform.txt": "https://huyenchip.com/2024/07/25/genai-platform.html",
    "hallucination.txt": "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "quora_engineering.txt": "https://quoraengineering.quora.com/Building-Embedding-Search-at-Quora"
}

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'judge_llm' not in st.session_state:
    st.session_state.judge_llm = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'faiss_downloaded' not in st.session_state:
    st.session_state.faiss_downloaded = False

# Example ground truth dataset
GOLD_DATA = [
    {
        "question": "What is ColBERT and why late interaction matters?",
        "ground_truth": "ColBERT is a retrieval model developed at Stanford that uses BERT embeddings with a late interaction mechanism. Late interaction improves efficiency by separating query and document processing until the final scoring step, balancing accuracy with scalability.",
        "expected_citations": [
            "https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/"
        ]
    },
    {
        "question": "How do you handle hallucinations in LLMs?",
        "ground_truth": "Hallucinations in LLMs can be handled through techniques like retrieval-augmented generation (RAG), fact-checking mechanisms, confidence scoring, and using external knowledge bases to ground responses in factual information.",
        "expected_citations": [
            "https://lilianweng.github.io/posts/2024-07-07-hallucination/"
        ]
    },
    {
        "question": "What are the benefits of using embeddings in search?",
        "ground_truth": "Embeddings enable semantic search by capturing meaning beyond keyword matching, allowing for better retrieval of relevant documents, handling synonyms and context, and supporting multi-modal search capabilities.",
        "expected_citations": [
            "https://quoraengineering.quora.com/Building-Embedding-Search-at-Quora"
        ]
    }
]

# Header
st.markdown('<h1 class="main-header">🔍 Hybrid RAG System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BM25 + FAISS Vector Search with Cross-Encoder Reranking & LLM Judge</p>', unsafe_allow_html=True)

# API Key Management
def get_api_key():
    """Get API key from secrets or environment"""
    try:
        # Try Streamlit secrets first
        return st.secrets["GOOGLE_API_KEY"]
    except:
        try:
            # Fall back to environment variable
            return os.environ.get("GOOGLE_API_KEY")
        except:
            return None

# Set API key
api_key = get_api_key()
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
else:
    st.error("🔑 Google API key not found in secrets. Please configure GOOGLE_API_KEY in Streamlit secrets.")
    st.info("💡 Add your API key in `.streamlit/secrets.toml` or Streamlit Cloud settings")
    st.stop()

# Google Drive file download functions
def get_drive_file_list(folder_id):
    """Get list of files in Google Drive folder"""
    try:
        # Use Google Drive API to list files (public folder)
        api_url = f"https://www.googleapis.com/drive/v3/files"
        params = {
            'q': f"'{folder_id}' in parents",
            'key': api_key,  # You might need a separate Drive API key
            'fields': 'files(id,name,mimeType)'
        }
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            return response.json().get('files', [])
        return []
    except:
        return []

def download_drive_file(file_id, filename, local_path):
    """Download file from Google Drive"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except:
        return False

@st.cache_data
def setup_faiss_index():
    """Download and setup FAISS index from Google Drive"""
    if os.path.exists(LOCAL_FAISS_PATH) and len(os.listdir(LOCAL_FAISS_PATH)) > 0:
        return True
    
    try:
        os.makedirs(LOCAL_FAISS_PATH, exist_ok=True)
        
        # Known file IDs from your drive folder structure
        # You'll need to replace these with actual file IDs from your shared folder
        drive_files = {
            "index.faiss": "your_index_faiss_file_id",
            "index.pkl": "your_index_pkl_file_id", 
            # Add other necessary files
        }
        
        # Alternative: Try direct download if files are publicly accessible
        base_url = f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"
        
        # For now, we'll create a placeholder message
        st.warning("⚠️ FAISS index download from Google Drive needs to be configured with proper file IDs")
        
        # Create dummy files for demonstration
        dummy_files = ['index.faiss', 'index.pkl']
        for file in dummy_files:
            Path(LOCAL_FAISS_PATH) / file
            
        return False  # Return False until proper implementation
        
    except Exception as e:
        st.error(f"Error setting up FAISS index: {str(e)}")
        return False

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # System Status
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("🔧 System Status")
        
        # API Key Status
        if api_key:
            st.markdown('<p class="status-success">✅ API Key: Configured</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ API Key: Missing</p>', unsafe_allow_html=True)
        
        # FAISS Index Status
        if os.path.exists(LOCAL_FAISS_PATH) and len(os.listdir(LOCAL_FAISS_PATH)) > 0:
            st.markdown('<p class="status-success">✅ FAISS Index: Ready</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ FAISS Index: Not Found</p>', unsafe_allow_html=True)
            if st.button("📥 Download FAISS Index"):
                with st.spinner("Downloading FAISS index..."):
                    success = setup_faiss_index()
                    if success:
                        st.success("✅ FAISS index downloaded!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to download FAISS index")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System Configuration
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("🛠️ System Settings")
        
        use_reranker = st.toggle(
            "Enable Cross-Encoder Reranking", 
            value=True,
            help="Use cross-encoder for result reranking"
        )
        
        if use_reranker:
            reranker_model = st.selectbox(
                "Reranker Model",
                ["cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 "cross-encoder/ms-marco-TinyBERT-L-2-v2"],
                help="Choose cross-encoder model for reranking"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Search Parameters
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("🎯 Search Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            bm25_top_k = st.slider("BM25 Top K", 5, 20, 10)
            faiss_top_k = st.slider("FAISS Top K", 5, 20, 10)
        with col2:
            final_top_k = st.slider("Final Top K", 1, 10, 3)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # LLM Judge Configuration
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.subheader("⚖️ LLM Judge")
        
        enable_judge = st.toggle(
            "Enable LLM Judge Evaluation", 
            value=False,
            help="Use LLM to evaluate response quality"
        )
        
        if enable_judge:
            judge_model = st.selectbox(
                "Judge Model",
                ["gemini-1.5-flash", "gemini-1.5-pro"],
                help="Choose the LLM model for judging responses"
            )
        st.markdown('</div>', unsafe_allow_html=True)

# System initialization functions
@st.cache_resource
def initialize_system(_use_reranker, _reranker_model):
    """Initialize the RAG system components"""
    try:
        if not os.path.exists(LOCAL_FAISS_PATH):
            st.error("FAISS index not found. Please download it first.")
            return None, None, None, None, None, None
            
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load FAISS index
        vectorstore = FAISS.load_local(LOCAL_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        # Build BM25 index
        docs = []
        sources = []
        for doc_id, doc in vectorstore.docstore._dict.items():
            text = getattr(doc, "page_content", None) or doc.page_content
            docs.append(text)
            # Clean up source paths
            raw_source = doc.metadata.get("source", "Unknown")
            clean_source = os.path.basename(raw_source) if raw_source != "Unknown" else "Unknown"
            sources.append(clean_source)
        
        tokenized = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        
        # Load reranker if enabled
        reranker = None
        if _use_reranker:
            reranker = CrossEncoder(_reranker_model)
        
        # Build text to docs mapping
        text_to_docs = {}
        for doc in vectorstore.docstore._dict.values():
            text = doc.page_content
            text_to_docs.setdefault(text, []).append(doc)
        
        return vectorstore, bm25, docs, sources, reranker, text_to_docs
    
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, None, None, None, None

@st.cache_resource
def initialize_judge_llm(_judge_model):
    """Initialize the LLM judge"""
    return ChatGoogleGenerativeAI(
        model=_judge_model,
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

def format_sources(sources):
    """Format source paths to URLs where possible"""
    formatted = []
    for source in sources:
        clean_source = os.path.basename(source) if source else "Unknown"
        formatted_source = SOURCE_URL_MAP.get(clean_source, clean_source)
        formatted.append(formatted_source)
    return formatted

def hybrid_search(query: str, bm25, bm25_docs, bm25_sources, vectorstore, 
                 text_to_docs, reranker=None, bm_k=10, faiss_k=10, top_k=3):
    """Perform hybrid search combining BM25 and FAISS"""
    # BM25 candidates
    tokenized_q = query.split()
    bm25_scores = bm25.get_scores(tokenized_q)
    bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm_k]
    bm25_candidates = []
    for idx in bm25_indices:
        txt = bm25_docs[idx]
        src = bm25_sources[idx]
        bm25_candidates.append((txt, src, bm25_scores[idx]))
    
    # FAISS candidates
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": faiss_k})
    faiss_docs = faiss_retriever.get_relevant_documents(query)
    faiss_candidates = []
    for d in faiss_docs:
        clean_source = os.path.basename(d.metadata.get("source", "Unknown"))
        faiss_candidates.append((d.page_content, clean_source, None))
    
    # Merge candidates
    merged = OrderedDict()
    for txt, src, score in bm25_candidates + faiss_candidates:
        key = txt.strip()
        if key not in merged:
            merged[key] = {"text": txt, "source": src, "bm25_score": score}
    
    candidates = list(merged.values())
    
    # Rerank if enabled
    if reranker and candidates:
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    else:
        candidates = sorted(candidates, key=lambda x: (0 if x["bm25_score"] is not None else 1, -(x["bm25_score"] or 0)))
    
    # Return top_k Document objects
    final_docs = []
    for c in candidates[:top_k]:
        text = c["text"]
        docs_list = text_to_docs.get(text, [])
        if docs_list:
            final_docs.append(docs_list[0])
        else:
            final_docs.append(Document(page_content=text, metadata={"source": c.get("source","Unknown")}))
    
    return final_docs, candidates[:top_k]

def query_rag_system(question: str, vectorstore, bm25, bm25_docs, bm25_sources, 
                    text_to_docs, reranker, bm_k, faiss_k, top_k):
    """Query the RAG system and return answer with sources"""
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )
    
    # Create prompt template
    prompt_template = """
    Use the provided context to answer the question.
    If the answer is not in the context, say: I couldn't find anything in my knowledge base about that topic. I can only answer questions related to AI, RAG, and the documents you provided.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    # Get hybrid search results
    docs, candidates = hybrid_search(
        question, bm25, bm25_docs, bm25_sources, vectorstore, 
        text_to_docs, reranker, bm_k, faiss_k, top_k
    )
    
    # Create context and query LLM
    context = "\n\n".join([doc.page_content for doc in docs])
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    answer_text = response.content.strip()
    
    # Check for fallback response - be more specific about the fallback message
    fallback_phrases = [
        "i couldn't find anything in my knowledge base about that topic",
        "i can only answer questions related to ai, rag",
        "the answer is not in the context"
    ]
    
    is_fallback = any(phrase.lower() in answer_text.lower() for phrase in fallback_phrases)
    
    if is_fallback:
        sources = []
    else:
        # Get unique sources from retrieved documents
        raw_sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            # Clean up source path to get just the filename
            clean_source = os.path.basename(source) if source != "Unknown" else "Unknown"
            raw_sources.append(clean_source)
        
        # Remove duplicates while preserving order
        unique_sources = list(OrderedDict.fromkeys(raw_sources))
        
        # Format sources - convert filenames to URLs where possible
        sources = []
        for source in unique_sources:
            # Map known filenames to their URLs
            if source in SOURCE_URL_MAP:
                sources.append(SOURCE_URL_MAP[source])
            elif source != "Unknown":
                sources.append(source)
        
        # Remove any empty sources
        sources = [s for s in sources if s and s.strip()]
    
    return {
        "answer": answer_text, 
        "sources": sources,
        "retrieved_docs": docs,
        "candidates": candidates
    }
def llm_judge(question: str, system_answer: str, citations: List[str], 
             ground_truth: str, expected_citations: List[str], judge_llm) -> Dict:
    """Evaluate system answer using LLM judge"""
    judge_prompt = f"""You are an expert evaluator for RAG systems. Please evaluate the system's answer against the ground truth.

Question: {question}

System Answer: {system_answer}

Ground Truth Answer: {ground_truth}

System Citations: {citations}

Expected Citations: {expected_citations}

Please evaluate on three dimensions (scale 0.0 to 1.0):

1. ACCURACY: How factually correct is the system answer compared to ground truth?
2. COVERAGE: How complete is the system answer in covering the key points?  
3. CITATION_MATCH: How relevant are the provided citations to the expected ones?

You MUST respond with ONLY a valid JSON object in this exact format:
{{"accuracy": 0.85, "coverage": 0.92, "citation_match": 0.67}}

Do not include any other text, explanations, or formatting. Only the JSON object."""

    try:
        response = judge_llm.invoke(judge_prompt)
        response_text = response.content.strip()
        
        # Clean up common formatting issues
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Remove any leading/trailing whitespace and newlines
        response_text = response_text.strip()
        
        # Try to find JSON object in the response
        import re
        json_match = re.search(r'\{[^}]*\}', response_text)
        if json_match:
            response_text = json_match.group()
        
        # Parse JSON
        scores = json.loads(response_text)
        
        # Validate structure and ranges
        required_keys = ["accuracy", "coverage", "citation_match"]
        if not all(key in scores for key in required_keys):
            raise ValueError("Missing required keys in response")
        
        # Ensure values are floats between 0 and 1
        for key in required_keys:
            scores[key] = max(0.0, min(1.0, float(scores[key])))
        
        return scores
        
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        # If JSON parsing fails, try to extract numbers from the response
        try:
            import re
            response_text = response.content.strip()
            
            # Look for patterns like "accuracy": 0.85, "coverage": 0.92, etc.
            accuracy_match = re.search(r'"?accuracy"?\s*:?\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            coverage_match = re.search(r'"?coverage"?\s*:?\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            citation_match = re.search(r'"?citation[_\s]?match"?\s*:?\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)
            
            scores = {
                "accuracy": max(0.0, min(1.0, float(accuracy_match.group(1)) if accuracy_match else 0.5)),
                "coverage": max(0.0, min(1.0, float(coverage_match.group(1)) if coverage_match else 0.5)),
                "citation_match": max(0.0, min(1.0, float(citation_match.group(1)) if citation_match else 0.5))
            }
            
            return scores
            
        except:
            # Final fallback - return neutral scores
            st.warning(f"⚠️ Judge response parsing failed. Using neutral scores. Raw response: {response.content[:200]}...")
            return {"accuracy": 0.5, "coverage": 0.5, "citation_match": 0.5}

def run_evaluation_suite(vectorstore, bm25, bm25_docs, bm25_sources, 
                        text_to_docs, reranker, judge_llm, gold_data,
                        bm_k, faiss_k, top_k):
    """Run evaluation on the entire gold dataset"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(gold_data):
        status_text.text(f"Evaluating question {i+1}/{len(gold_data)}")
        progress_bar.progress((i + 1) / len(gold_data))
        
        question = item["question"]
        ground_truth = item["ground_truth"]
        expected_citations = item["expected_citations"]
        
        # Get system response
        result = query_rag_system(
            question, vectorstore, bm25, bm25_docs, bm25_sources,
            text_to_docs, reranker, bm_k, faiss_k, top_k
        )
        
        # Judge evaluation
        scores = llm_judge(
            question, result["answer"], result["sources"], 
            ground_truth, expected_citations, judge_llm
        )
        
        eval_result = {
            "question": question,
            "system_answer": result["answer"],
            "ground_truth": ground_truth,
            "citations": result["sources"],
            "expected_citations": expected_citations,
            "scores": scores
        }
        
        results.append(eval_result)
    
    status_text.text("Evaluation complete!")
    progress_bar.progress(1.0)
    
    return results

# Main application
if api_key:
    # Check if FAISS index exists, if not show download option
    if not os.path.exists(LOCAL_FAISS_PATH) or len(os.listdir(LOCAL_FAISS_PATH)) == 0:
        st.warning("⚠️ FAISS index not found. Please download it from the sidebar configuration.")
        
        # Manual file upload option as fallback
        st.info("💡 **Alternative**: You can manually upload your FAISS index files")
        uploaded_files = st.file_uploader(
            "Upload FAISS index files (index.faiss, index.pkl)", 
            type=['faiss', 'pkl'],
            accept_multiple_files=True,
            help="Upload the FAISS index files from your local machine"
        )
        
        if uploaded_files:
            os.makedirs(LOCAL_FAISS_PATH, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(LOCAL_FAISS_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
            st.success("✅ Files uploaded successfully!")
            st.rerun()
    
    elif not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing RAG system... This may take a moment."):
            components = initialize_system(use_reranker, reranker_model if use_reranker else None)
            if all(comp is not None for comp in components):
                st.session_state.vectorstore, st.session_state.bm25, st.session_state.bm25_docs, \
                st.session_state.bm25_sources, st.session_state.reranker, st.session_state.text_to_docs = components
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system. Please check your FAISS index.")
    
    # Initialize judge LLM if enabled
    if enable_judge and st.session_state.judge_llm is None:
        with st.spinner("⚖️ Initializing LLM Judge..."):
            st.session_state.judge_llm = initialize_judge_llm(judge_model)
            st.success("✅ LLM Judge initialized!")

if st.session_state.system_initialized:
    # Create tabs for different functionality
    tab1, tab2, tab3 = st.tabs(["💬 Query System", "⚖️ LLM Judge", "📊 Evaluation Results"])
    
    with tab1:
        # Main query interface
        st.header("💬 Ask Your Question")
        
        # Quick examples
        st.markdown("**💡 Try these sample queries:**")
        sample_cols = st.columns(3)
        sample_queries = [
            "What is ColBERT and why does late interaction matter?",
            "How do you handle hallucinations in LLMs?", 
            "What are the benefits of using embeddings in search?"
        ]
        
        for i, (col, sample) in enumerate(zip(sample_cols, sample_queries)):
            with col:
                if st.button(f"📝 Example {i+1}", key=f"example_{i}", help=sample):
                    st.session_state.current_query = sample
        
        # Query input
        query = st.text_input(
            "Enter your question:", 
            value=st.session_state.get('current_query', ''),
            placeholder="Ask questions about AI, RAG, or content in your knowledge base",
            help="Ask questions about AI, RAG, or content in your knowledge base"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
        
        if clear_button:
            st.session_state.query_history = []
            st.session_state.current_query = ""
            st.rerun()
        
        # Process query
        if search_button and query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = query_rag_system(
                        query, 
                        st.session_state.vectorstore,
                        st.session_state.bm25,
                        st.session_state.bm25_docs,
                        st.session_state.bm25_sources,
                        st.session_state.text_to_docs,
                        st.session_state.reranker,
                        bm25_top_k,
                        faiss_top_k,
                        final_top_k
                    )
                    
                    # LLM Judge evaluation if enabled
                    judge_scores = None
                    if enable_judge and st.session_state.judge_llm:
                        # Check if query matches any gold data for evaluation
                        matching_gold = next((item for item in GOLD_DATA if item["question"].lower() == query.lower()), None)
                        if matching_gold:
                            with st.spinner("⚖️ Evaluating response..."):
                                judge_scores = llm_judge(
                                    query, result["answer"], result["sources"],
                                    matching_gold["ground_truth"], matching_gold["expected_citations"],
                                    st.session_state.judge_llm
                                )
                    
                    # Add to history
                    st.session_state.query_history.insert(0, {
                        "query": query,
                        "result": result,
                        "judge_scores": judge_scores
                    })
                    
                    # Display result
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                    
                    # Display LLM Judge scores if available
                    if judge_scores:
                        st.markdown("### ⚖️ LLM Judge Evaluation")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            accuracy = judge_scores.get("accuracy", 0)
                            st.metric("🎯 Accuracy", f"{accuracy:.2f}", help="Factual correctness vs ground truth")
                        with col2:
                            coverage = judge_scores.get("coverage", 0)
                            st.metric("📋 Coverage", f"{coverage:.2f}", help="Completeness of the answer")
                        with col3:
                            citation_match = judge_scores.get("citation_match", 0)
                            st.metric("📚 Citation Match", f"{citation_match:.2f}", help="Relevance of citations")
                    
                    # Display sources
                    if result["sources"]:
                        st.markdown("### 📚 Sources")
                        for i, source in enumerate(result["sources"], 1):
                            if source.startswith("http"):
                                st.markdown(f'<div class="source-link">📎 <a href="{source}" target="_blank">Source {i}</a></div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="source-link">📎 Source {i}: {source}</div>', unsafe_allow_html=True)
                    
                    # Search metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown('<div class="metric-card"><h3>📊 Retrieved</h3><h2>{}</h2><p>Documents</p></div>'.format(len(result["retrieved_docs"])), unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card"><h3>🎯 Sources</h3><h2>{}</h2><p>Found</p></div>'.format(len(result["sources"])), unsafe_allow_html=True)
                    with col3:
                        rerank_status = "✅ Enabled" if use_reranker else "❌ Disabled"
                        st.markdown('<div class="metric-card"><h3>🔄 Reranking</h3><h2>{}</h2></div>'.format(rerank_status), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        
        # Query history
        if st.session_state.query_history:
            st.header("📜 Query History")
            for i, entry in enumerate(st.session_state.query_history[:5]):  # Show last 5 queries
                with st.expander(f"Q: {entry['query'][:100]}..."):
                    st.markdown("**Answer:**")
                    st.write(entry['result']['answer'])
                    
                    if entry.get('judge_scores'):
                        st.markdown("**Judge Scores:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{entry['judge_scores'].get('accuracy', 0):.2f}")
                        with col2:
                            st.metric("Coverage", f"{entry['judge_scores'].get('coverage', 0):.2f}")
                        with col3:
                            st.metric("Citation Match", f"{entry['judge_scores'].get('citation_match', 0):.2f}")
                    
                    if entry['result']['sources']:
                        st.markdown("**Sources:**")
                        for source in entry['result']['sources']:
                            st.write(f"• {source}")
    
    with tab2:
        st.header("⚖️ LLM Judge Evaluation")
        
        if not enable_judge:
            st.info("🔧 Enable LLM Judge in the sidebar configuration to use this feature.")
        else:
            st.markdown("### 📋 Gold Dataset")
            st.write(f"Current dataset contains **{len(GOLD_DATA)}** evaluation questions.")
            
            # Display gold data
            for i, item in enumerate(GOLD_DATA, 1):
                with st.expander(f"Question {i}: {item['question'][:80]}..."):
                    st.markdown("**Question:**")
                    st.write(item["question"])
                    st.markdown("**Ground Truth:**")
                    st.write(item["ground_truth"])
                    st.markdown("**Expected Citations:**")
                    for citation in item["expected_citations"]:
                        st.write(f"• {citation}")
            
            # Run evaluation suite
            if st.button("🚀 Run Full Evaluation Suite", type="primary"):
                if st.session_state.judge_llm:
                    with st.spinner("🔄 Running evaluation suite..."):
                        results = run_evaluation_suite(
                            st.session_state.vectorstore,
                            st.session_state.bm25,
                            st.session_state.bm25_docs,
                            st.session_state.bm25_sources,
                            st.session_state.text_to_docs,
                            st.session_state.reranker,
                            st.session_state.judge_llm,
                            GOLD_DATA,
                            bm25_top_k,
                            faiss_top_k,
                            final_top_k
                        )
                        st.session_state.evaluation_results = results
                        st.success("✅ Evaluation complete! Check the 'Evaluation Results' tab.")
                else:
                    st.error("❌ Please initialize the judge LLM first.")
    
    with tab3:
        st.header("📊 Evaluation Results")
        
        if not st.session_state.evaluation_results:
            st.info("🔍 Run the evaluation suite to see results here.")
        else:
            results = st.session_state.evaluation_results
            
            # Overall metrics
            st.markdown("### 📈 Overall Performance")
            
            avg_accuracy = sum(r["scores"].get("accuracy", 0) for r in results) / len(results)
            avg_coverage = sum(r["scores"].get("coverage", 0) for r in results) / len(results)
            avg_citation = sum(r["scores"].get("citation_match", 0) for r in results) / len(results)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Questions", len(results))
            with col2:
                st.metric("🎯 Avg Accuracy", f"{avg_accuracy:.3f}")
            with col3:
                st.metric("📋 Avg Coverage", f"{avg_coverage:.3f}")
            with col4:
                st.metric("📚 Avg Citation Match", f"{avg_citation:.3f}")
            
            # Performance visualization
            st.markdown("### 📊 Performance Breakdown")
            
            # Create performance chart data
            chart_data = []
            for i, result in enumerate(results, 1):
                scores = result["scores"]
                chart_data.append({
                    "Question": f"Q{i}",
                    "Accuracy": scores.get("accuracy", 0),
                    "Coverage": scores.get("coverage", 0), 
                    "Citation Match": scores.get("citation_match", 0)
                })
            
            if chart_data:
                import pandas as pd
                df = pd.DataFrame(chart_data)
                st.bar_chart(df.set_index("Question"))
            
            # Detailed results
            st.markdown("### 📝 Detailed Results")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}: {result['question'][:80]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**System Answer:**")
                        st.write(result["system_answer"])
                        st.markdown("**System Citations:**")
                        for citation in result["citations"]:
                            st.write(f"• {citation}")
                    
                    with col2:
                        st.markdown("**Ground Truth:**")
                        st.write(result["ground_truth"])
                        st.markdown("**Expected Citations:**")
                        for citation in result["expected_citations"]:
                            st.write(f"• {citation}")
                    
                    # Scores
                    st.markdown("**Evaluation Scores:**")
                    score_col1, score_col2, score_col3 = st.columns(3)
                    scores = result["scores"]
                    with score_col1:
                        st.metric("Accuracy", f"{scores.get('accuracy', 0):.3f}")
                    with score_col2:
                        st.metric("Coverage", f"{scores.get('coverage', 0):.3f}")
                    with score_col3:
                        st.metric("Citation Match", f"{scores.get('citation_match', 0):.3f}")
            
            # Export results option
            if st.button("💾 Export Results as JSON"):
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="📥 Download Results",
                    data=results_json,
                    file_name=f"rag_evaluation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

else:
    st.info("🚀 Please ensure your Google Drive FAISS index is accessible and API key is configured!")
    
    # System requirements info
    st.header("🔧 Setup Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔑 API Key Setup")
        st.code("""
# Create .streamlit/secrets.toml file:
GOOGLE_API_KEY = "your-google-api-key-here"

# Or set environment variable:
export GOOGLE_API_KEY="your-google-api-key-here"
        """)
        
    with col2:
        st.markdown("### 📁 FAISS Index Setup")
        st.markdown(f"""
        **Google Drive Folder**: [Your FAISS Index]({f"https://drive.google.com/drive/folders/{GOOGLE_DRIVE_FOLDER_ID}"})
        
        **Expected Files**:
        - `index.faiss`
        - `index.pkl`
        - Other vectorstore files
        """)
    
    # Sample queries for inspiration
    st.header("💡 Sample Queries to Try")
    sample_queries = [
        "What is ColBERT and why does late interaction matter?",
        "How do you handle hallucinations in LLMs?", 
        "What are the benefits of using embeddings in search?",
        "Explain the architecture of a RAG system",
        "What are the challenges in building GenAI platforms?"
    ]
    
    for query in sample_queries:
        st.markdown(f"• {query}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚀 <strong>Hybrid RAG System</strong> - Built with Streamlit</p>
        <p>💡 Powered by BM25 + FAISS + Cross-Encoder + LLM Judge</p>
        <p>🔗 <a href="https://github.com/Balaji-itz-me/RAG-hybrid-retrieval-evaluator" target="_blank">GitHub Repository</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Debug info (only show in development)
if st.sidebar.toggle("🔍 Debug Info", value=False):
    st.sidebar.markdown("### 🔍 Debug Information")
    st.sidebar.json({
        "API Key Status": "✅ Set" if api_key else "❌ Missing",
        "FAISS Path": LOCAL_FAISS_PATH,
        "FAISS Exists": os.path.exists(LOCAL_FAISS_PATH),
        "System Initialized": st.session_state.system_initialized,
        "Query History Count": len(st.session_state.query_history),
        "Evaluation Results": len(st.session_state.evaluation_results)
    })



