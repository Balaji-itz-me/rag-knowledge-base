# Setup imports and basic configuration
import os
import json
import time
import shutil
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
from collections import OrderedDict
import hashlib
import re
import uuid
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from threading import Thread

# Import the RAG components
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✅ All imports successful!")

# ========================================
# STEP 2: CONFIGURATION AND API KEYS
# ========================================

# Configuration for AWS deployment
class Config:
    # Base path for AWS EC2
    BASE_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo"
    
    # Static index (existing)
    STATIC_FAISS_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/faiss_index"

    # Dynamic index (new)
    DYNAMIC_BASE_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index"
    DYNAMIC_FAISS_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index/faiss_index"
    METADATA_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index/metadata"
    BACKUP_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index/backups"

    # Conversation settings
    CONVERSATIONS_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/conversations"
    MAX_CONVERSATION_LENGTH = 10  # Maximum number of message pairs
    CONVERSATION_TIMEOUT = 1800   # 30 minutes in seconds
    MAX_CONTEXT_LENGTH = 3        # Last N message pairs to include in context

    # Indexing settings
    USE_RERANKER = True
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    BM25_TOP_K = 10
    FAISS_TOP_K = 10
    FINAL_TOP_K = 3

    # Web scraping settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RE_INDEX_DAYS = 7
    MAX_VERSIONS = 3

    # Evaluation settings
    EVALUATION_DATASET_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/evaluation"

    # Authentication - YOU CAN CHANGE THESE KEYS
    VALID_API_KEYS = {
        "demo-api-key-123": {"user": "demo_user", "permissions": ["read", "query", "index", "chat"]},
        "eval-key-456": {"user": "evaluator", "permissions": ["read", "query", "chat", "eval"]},
    }

config = Config()

# Get API keys from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set!")
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
print("✅ Configuration loaded!")

# ========================================
# STEP 3: PYDANTIC MODELS (Same as before)
# ========================================

# Chat-specific models
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_items=1, max_items=20)
    session_id: Optional[str] = Field(default=None)
    use_dynamic_index: Optional[bool] = Field(default=True)
    use_reranker: Optional[bool] = Field(default=True)
    top_k: Optional[int] = Field(default=3, ge=1, le=5)

class ChatResponse(BaseModel):
    session_id: str
    response: Dict[str, Any]
    conversation_length: int
    timestamp: datetime = Field(default_factory=datetime.now)

# Original models (updated)
class IndexRequest(BaseModel):
    url: List[str] = Field(..., min_items=1, max_items=5, description="URLs to index")

    @validator('url')
    def validate_urls(cls, v):
        for url in v:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")
        return v

class IndexResponse(BaseModel):
    status: str
    indexed_url: List[str]
    failed_url: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    components: dict
    conversations_active: int

# Sources models
class SourceInfo(BaseModel):
    source_url: str
    title: Optional[str] = None
    indexed_at: Optional[datetime] = None
    document_count: int = 0
    source_type: str = Field(..., description="static or dynamic")
    last_updated: Optional[datetime] = None

class SourcesResponse(BaseModel):
    total_sources: int
    static_sources: List[SourceInfo]
    dynamic_sources: List[SourceInfo]
    timestamp: datetime = Field(default_factory=datetime.now)

# Evaluation models
class EvaluationRequest(BaseModel):
    test_cases: List[Dict[str, Any]]
    session_id: Optional[str] = None

class EvaluationResponse(BaseModel):
    overall_score: float
    detailed_results: List[Dict[str, Any]]
    metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)

print("✅ Pydantic models defined!")

# ========================================
# STEP 4: GLOBAL VARIABLES AND UTILITIES
# ========================================

# Global variables
embedding_model = None
static_vectorstore = None
dynamic_vectorstore = None
llm = None
bm25_static = None
bm25_dynamic = None
bm25_docs_static = None
bm25_docs_dynamic = None
bm25_sources_static = None
bm25_sources_dynamic = None
reranker = None
text_to_docs_static = None
text_to_docs_dynamic = None
chat_prompt = None
text_splitter = None

# In-memory conversation storage (will use file-based backup)
active_conversations = {}

# Updated source URL mapping for static content (AWS paths)
source_url_map = {
    "/content/drive/MyDrive/RAG_demo/data/genai-platform.txt": "https://huyenchip.com/2024/07/25/genai-platform.html",
    "/content/drive/MyDrive/RAG_demo/data/hallucination.txt": "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "/content/drive/MyDrive/RAG_demo/data/quora_engineering.txt": "https://quoraengineering.quora.com/Building-Embedding-Search-at-Quora"
}

# Utility functions
def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        config.BASE_PATH,
        config.DYNAMIC_BASE_PATH,
        config.DYNAMIC_FAISS_PATH,
        config.METADATA_PATH,
        config.BACKUP_PATH,
        config.CONVERSATIONS_PATH,
        config.EVALUATION_DATASET_PATH
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    logger.info("✅ Directory structure created")

def get_url_hash(url: str) -> str:
    """Generate a hash for URL to use as unique identifier"""
    return hashlib.md5(url.encode()).hexdigest()

def load_metadata(file_path: str) -> Dict:
    """Load metadata from JSON file"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {file_path}: {str(e)}")
    return {}

def save_metadata(data: Dict, file_path: str):
    """Save metadata to JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving metadata to {file_path}: {str(e)}")
        raise

def format_sources(sources):
    """Convert file paths to URLs"""
    formatted = []
    for source in sources:
        if source.startswith('http'):
            formatted.append(source)
        else:
            formatted.append(source_url_map.get(source, source))
    return formatted

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def clean_old_conversations():
    """Remove old conversations from memory and save important ones"""
    current_time = datetime.now()
    expired_sessions = []

    for session_id, conv_data in active_conversations.items():
        last_active = conv_data.get('last_active', current_time)
        if isinstance(last_active, str):
            last_active = datetime.fromisoformat(last_active)

        if (current_time - last_active).seconds > config.CONVERSATION_TIMEOUT:
            # Save conversation before deletion
            try:
                conv_file = os.path.join(config.CONVERSATIONS_PATH, f"{session_id}.json")
                save_metadata(conv_data, conv_file)
            except Exception as e:
                logger.error(f"Error saving conversation {session_id}: {e}")
            
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        del active_conversations[session_id]
        logger.info(f"Cleaned expired conversation: {session_id}")

print("✅ Utility functions defined!")

# ========================================
# REMAINING CODE STAYS THE SAME...
# (ConversationManager, WebScraper, search functions, etc.)
# ========================================

# [Include all your existing classes and functions here - they don't need changes]
# Just copy-paste from: ConversationManager through RAGEvaluator

# ... [ALL YOUR EXISTING CODE] ...

# ========================================
# STEP 5: CONVERSATION MANAGEMENT
# ========================================

class ConversationManager:
    def __init__(self):
        self.conversations = active_conversations

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id and session_id in self.conversations:
            # Update last active time
            self.conversations[session_id]['last_active'] = datetime.now()
            return session_id

        # Create new session
        new_session_id = generate_session_id()
        self.conversations[new_session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'metadata': {}
        }
        logger.info(f"Created new conversation session: {new_session_id}")
        return new_session_id

    def add_message(self, session_id: str, message: ChatMessage):
        """Add message to conversation"""
        if session_id not in self.conversations:
            raise ValueError(f"Session {session_id} not found")

        self.conversations[session_id]['messages'].append({
            'role': message.role,
            'content': message.content,
            'timestamp': message.timestamp.isoformat() if message.timestamp else datetime.now().isoformat()
        })
        self.conversations[session_id]['last_active'] = datetime.now()

        # Trim conversation if too long
        messages = self.conversations[session_id]['messages']
        if len(messages) > config.MAX_CONVERSATION_LENGTH * 2:  # *2 for user+assistant pairs
            self.conversations[session_id]['messages'] = messages[-config.MAX_CONVERSATION_LENGTH * 2:]

    def get_conversation_context(self, session_id: str, max_pairs: int = None) -> str:
        """Get conversation context for LLM"""
        if session_id not in self.conversations:
            return ""

        messages = self.conversations[session_id]['messages']
        if not messages:
            return ""

        # Get last N pairs (user + assistant messages)
        max_pairs = max_pairs or config.MAX_CONTEXT_LENGTH
        recent_messages = messages[-(max_pairs * 2):]

        context_parts = []
        for msg in recent_messages:
            role = "Human" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")

        return "\n".join(context_parts)

    def get_conversation_length(self, session_id: str) -> int:
        """Get number of message exchanges"""
        if session_id not in self.conversations:
            return 0
        return len(self.conversations[session_id]['messages'])

    def save_conversation(self, session_id: str):
        """Save conversation to disk"""
        if session_id not in self.conversations:
            return

        conv_file = os.path.join(config.CONVERSATIONS_PATH, f"{session_id}.json")
        save_metadata(self.conversations[session_id], conv_file)

conversation_manager = ConversationManager()
print("✅ Conversation manager initialized!")

# ========================================
# STEP 6: WEB SCRAPING AND INDEXING
# ========================================

class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content from URL with retry logic"""
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                    element.decompose()

                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else url

                # Extract main content
                content_selectors = [
                    'article', 'main', '[role="main"]',
                    '.content', '.post', '.article',
                    'div.container', 'div.wrapper'
                ]

                content_text = ""
                for selector in content_selectors:
                    content = soup.select_one(selector)
                    if content:
                        content_text = content.get_text()
                        break

                if not content_text:
                    # Fallback to body
                    body = soup.find('body')
                    content_text = body.get_text() if body else ""

                # Clean text
                content_text = re.sub(r'\s+', ' ', content_text).strip()

                if not content_text:
                    raise ValueError("No content extracted")

                return {
                    'title': title_text,
                    'content': content_text,
                    'url': url,
                    'success': True,
                    'error': None
                }

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY * (2 ** attempt))
                else:
                    return {
                        'title': None,
                        'content': None,
                        'url': url,
                        'success': False,
                        'error': str(e)
                    }

def build_bm25_from_vectorstore(vectorstore):
    """Build BM25 index from FAISS vectorstore"""
    try:
        docs = []
        sources = []
        for doc_id, doc in vectorstore.docstore._dict.items():
            text = getattr(doc, "page_content", None) or doc.page_content
            docs.append(text)
            sources.append(doc.metadata.get("source", "Unknown"))

        tokenized = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized)

        # Build text to docs mapping
        text_to_docs = {}
        for doc in vectorstore.docstore._dict.values():
            text = doc.page_content
            text_to_docs.setdefault(text, []).append(doc)

        logger.info(f"BM25 built over {len(docs)} chunks")
        return bm25, docs, sources, text_to_docs
    except Exception as e:
        logger.error(f"Error building BM25: {str(e)}")
        raise

print("✅ Web scraping and indexing functions defined!")

# ========================================
# STEP 7: HYBRID SEARCH SYSTEM
# ========================================

def hybrid_search(query: str, use_dynamic: bool = True, bm_k: int = 10, faiss_k: int = 10, top_k: int = 3, use_reranker: bool = True, conversation_context: str = ""):
    """Perform hybrid search on static and/or dynamic indices with conversation context"""
    try:
        # Enhanced query with context
        enhanced_query = query
        if conversation_context:
            # Add context to help with pronouns and references
            enhanced_query = f"Context: {conversation_context}\n\nCurrent question: {query}"

        all_candidates = []

        # Search static index
        if static_vectorstore:
            candidates_static = _search_single_index(
                enhanced_query, static_vectorstore, bm25_static, bm25_docs_static,
                bm25_sources_static, text_to_docs_static, bm_k, faiss_k
            )
            all_candidates.extend(candidates_static)

        # Search dynamic index
        if use_dynamic and dynamic_vectorstore:
            candidates_dynamic = _search_single_index(
                enhanced_query, dynamic_vectorstore, bm25_dynamic, bm25_docs_dynamic,
                bm25_sources_dynamic, text_to_docs_dynamic, bm_k, faiss_k
            )
            all_candidates.extend(candidates_dynamic)

        if not all_candidates:
            return []

        # Merge and deduplicate candidates
        merged = OrderedDict()
        for candidate in all_candidates:
            key = candidate["text"].strip()[:100]  # Use first 100 chars as key
            if key not in merged or (candidate.get("bm25_score") or 0) > (merged[key].get("bm25_score") or 0):
                merged[key] = candidate

        candidates = list(merged.values())

        # Optional reranking - use original query for reranking
        if use_reranker and reranker and candidates:
            pairs = [(query, c["text"]) for c in candidates]  # Use original query for reranking
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
            # Try to find in both mappings
            docs_list = text_to_docs_static.get(text, []) if text_to_docs_static else []
            if not docs_list and text_to_docs_dynamic:
                docs_list = text_to_docs_dynamic.get(text, [])

            if docs_list:
                final_docs.append(docs_list[0])
            else:
                final_docs.append(Document(page_content=text, metadata={"source": c.get("source", "Unknown")}))

        return final_docs

    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise

def _search_single_index(query: str, vectorstore, bm25, bm25_docs, bm25_sources, text_to_docs, bm_k: int, faiss_k: int):
    """Search a single index (static or dynamic)"""
    candidates = []

    # BM25 candidates
    if bm25 and bm25_docs:
        tokenized_q = query.split()
        bm25_scores = bm25.get_scores(tokenized_q)
        bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm_k]
        for idx in bm25_indices:
            txt = bm25_docs[idx]
            src = bm25_sources[idx]
            candidates.append({"text": txt, "source": src, "bm25_score": bm25_scores[idx]})

    # FAISS candidates
    if vectorstore:
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": faiss_k})
        faiss_docs = faiss_retriever.get_relevant_documents(query)
        for d in faiss_docs:
            candidates.append({"text": d.page_content, "source": d.metadata.get("source", "Unknown"), "bm25_score": None})

    return candidates

print("✅ Hybrid search system defined!")

# ========================================
# STEP 8: EVALUATION SYSTEM
# ========================================

class RAGEvaluator:
    def __init__(self):
        self.test_cases = []
        self.results = []

    def create_test_dataset(self):
        """Create evaluation test cases"""
        test_cases = [
            # Context retention tests
            {
                "conversation": [
                    {"role": "user", "content": "What is attention mechanism in transformers?"},
                    {"role": "user", "content": "How does it help with long sequences?"}
                ],
                "expected_context": "attention mechanism",
                "test_type": "context_retention"
            },

            # Citation accuracy tests
            {
                "conversation": [
                    {"role": "user", "content": "Tell me about RAG systems"}
                ],
                "expected_citations": True,
                "test_type": "citation_accuracy"
            },

            # Topic switching tests
            {
                "conversation": [
                    {"role": "user", "content": "Explain neural networks"},
                    {"role": "user", "content": "Now tell me about cooking recipes"}
                ],
                "expected_behavior": "topic_switch",
                "test_type": "topic_switching"
            },

            # Multi-source tests
            {
                "conversation": [
                    {"role": "user", "content": "What are the latest trends in AI?"}
                ],
                "expected_sources": "mixed",
                "test_type": "multi_source"
            }
        ]

        # Save test cases
        test_file = os.path.join(config.EVALUATION_DATASET_PATH, "test_cases.json")
        save_metadata(test_cases, test_file)
        return test_cases

    def evaluate_response_relevance(self, question: str, answer: str, context_docs: List[Document]) -> float:
        """Evaluate how relevant the answer is to the question"""
        try:
            # Simple keyword overlap score
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())

            overlap = len(question_words.intersection(answer_words))
            relevance_score = overlap / len(question_words) if question_words else 0.0

            # Bonus for using context
            context_text = " ".join([doc.page_content for doc in context_docs])
            context_words = set(context_text.lower().split())
            context_usage = len(answer_words.intersection(context_words)) / len(context_words) if context_words else 0.0

            final_score = min(1.0, (relevance_score + context_usage) / 2)
            return final_score

        except Exception as e:
            logger.error(f"Error evaluating relevance: {str(e)}")
            return 0.0

    def evaluate_citation_accuracy(self, answer: str, sources: List[str]) -> float:
        """Evaluate citation accuracy"""
        if not sources:
            return 0.0

        # Check if answer contains factual claims (simple heuristic)
        factual_indicators = ["according to", "research shows", "studies indicate", "data reveals", "analysis found"]
        has_factual_claims = any(indicator in answer.lower() for indicator in factual_indicators)

        if has_factual_claims and sources:
            return 1.0
        elif not has_factual_claims:
            return 0.8  # No factual claims, so no citations needed
        else:
            return 0.0  # Factual claims but no citations

    def evaluate_context_retention(self, conversation_history: List[Dict], current_answer: str) -> float:
        """Evaluate how well context from previous messages is retained"""
        if len(conversation_history) <= 1:
            return 1.0  # No previous context to retain

        # Look for references to previous topics
        previous_content = " ".join([msg["content"] for msg in conversation_history[:-1]])
        previous_words = set(previous_content.lower().split())
        answer_words = set(current_answer.lower().split())

        # Check for pronouns and references
        references = ["this", "that", "it", "they", "these", "those"]
        has_references = any(ref in current_answer.lower() for ref in references)

        # Calculate context retention score
        word_overlap = len(previous_words.intersection(answer_words)) / len(previous_words) if previous_words else 0.0
        reference_bonus = 0.3 if has_references else 0.0

        context_score = min(1.0, word_overlap + reference_bonus)
        return context_score

    async def run_evaluation(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        results = []
        total_scores = {"relevance": [], "citation": [], "context": []}

        for i, test_case in enumerate(test_cases):
            try:
                logger.info(f"Running test case {i+1}/{len(test_cases)}")

                # Simulate conversation
                session_id = generate_session_id()
                conversation_history = []

                for message in test_case["conversation"]:
                    # Add user message
                    user_msg = ChatMessage(role="user", content=message["content"])
                    conversation_manager.add_message(session_id, user_msg)
                    conversation_history.append({"role": "user", "content": message["content"]})

                    # Get bot response
                    context = conversation_manager.get_conversation_context(session_id)
                    docs = hybrid_search(
                        message["content"],
                        use_dynamic=True,
                        conversation_context=context
                    )

                    if docs:
                        doc_context = "\n\n".join([doc.page_content for doc in docs])
                        formatted_prompt = chat_prompt.format(
                            conversation_context=context,
                            context=doc_context,
                            question=message["content"]
                        )
                        response = llm.invoke(formatted_prompt)
                        answer = response.content.strip()
                        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
                        sources = format_sources(sources)
                    else:
                        answer = "I couldn't find specific information about that in my knowledge base."
                        sources = []

                    # Add assistant message
                    assistant_message = ChatMessage(role="assistant", content=answer)
                    conversation_manager.add_message(session_id, assistant_message)
                    conversation_history.append({"role": "assistant", "content": answer})

                # Evaluate the last response
                last_question = test_case["conversation"][-1]["content"]
                last_answer = conversation_history[-1]["content"]

                # Calculate scores
                relevance_score = self.evaluate_response_relevance(last_question, last_answer, docs if docs else [])
                citation_score = self.evaluate_citation_accuracy(last_answer, sources if sources else [])
                context_score = self.evaluate_context_retention(conversation_history[:-1], last_answer)

                result = {
                    "test_case_id": i,
                    "test_type": test_case.get("test_type", "general"),
                    "question": last_question,
                    "answer": last_answer,
                    "sources": sources if sources else [],
                    "scores": {
                        "relevance": relevance_score,
                        "citation": citation_score,
                        "context_retention": context_score
                    },
                    "overall_score": (relevance_score + citation_score + context_score) / 3
                }

                results.append(result)
                total_scores["relevance"].append(relevance_score)
                total_scores["citation"].append(citation_score)
                total_scores["context"].append(context_score)

            except Exception as e:
                logger.error(f"Error in test case {i}: {str(e)}")
                continue

        # Calculate overall metrics
        metrics = {
            "average_relevance": np.mean(total_scores["relevance"]) if total_scores["relevance"] else 0.0,
            "average_citation": np.mean(total_scores["citation"]) if total_scores["citation"] else 0.0,
            "average_context": np.mean(total_scores["context"]) if total_scores["context"] else 0.0,
            "overall_average": np.mean([np.mean(scores) for scores in total_scores.values()]) if any(total_scores.values()) else 0.0,
            "test_cases_completed": len(results),
            "test_cases_failed": len(test_cases) - len(results)
        }

        return {
            "overall_score": metrics["overall_average"],
            "detailed_results": results,
            "metrics": metrics
        }

evaluator = RAGEvaluator()
print("✅ Evaluation system initialized!")

# ========================================
# MODIFIED: MODEL INITIALIZATION
# ========================================

def initialize_models():
    """Initialize all models and components for AWS deployment"""
    global embedding_model, static_vectorstore, dynamic_vectorstore, llm
    global bm25_static, bm25_dynamic, bm25_docs_static, bm25_docs_dynamic
    global bm25_sources_static, bm25_sources_dynamic, reranker
    global text_to_docs_static, text_to_docs_dynamic, chat_prompt, text_splitter

    try:
        print("🔄 Loading models for AWS deployment...")
        ensure_directories()
        clean_old_conversations()

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        print("✅ Text splitter initialized")

        # Try to load static FAISS vectorstore (may not exist initially)
        try:
            if os.path.exists(config.STATIC_FAISS_PATH):
                static_vectorstore = FAISS.load_local(
                    config.STATIC_FAISS_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                bm25_static, bm25_docs_static, bm25_sources_static, text_to_docs_static = build_bm25_from_vectorstore(static_vectorstore)
                print("✅ Static FAISS vectorstore loaded")
            else:
                print("ℹ️ No static FAISS index found - will work with dynamic index only")
        except Exception as e:
            print(f"⚠️ Could not load static FAISS: {str(e)}")

        # Try to load dynamic FAISS vectorstore (may not exist initially)
        try:
            if os.path.exists(config.DYNAMIC_FAISS_PATH):
                dynamic_vectorstore = FAISS.load_local(
                    config.DYNAMIC_FAISS_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                bm25_dynamic, bm25_docs_dynamic, bm25_sources_dynamic, text_to_docs_dynamic = build_bm25_from_vectorstore(dynamic_vectorstore)
                print("✅ Dynamic FAISS vectorstore loaded")
            else:
                print("ℹ️ No dynamic FAISS index found - will create on first index")
        except Exception as e:
            print(f"ℹ️ Dynamic FAISS not found (will create on first index): {str(e)}")

        # Initialize LLM with environment variable
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        print("✅ LLM initialized")

        # Create chat prompt template
        chat_prompt_template = """
You are a helpful AI assistant with access to multiple knowledge sources. You can maintain context across conversations and provide accurate citations.

Previous conversation context:
{conversation_context}

Current context from knowledge base:
{context}

Current question:
{question}

Instructions:
1. Use the conversation context to understand references like "it", "this", "that", etc.
2. Provide accurate answers based on the knowledge base context
3. If you reference specific information, it should be from the provided context
4. If the answer is not in the knowledge base, say: "I couldn't find specific information about that in my knowledge base."
5. Be conversational and natural in your responses
6. Handle follow-up questions by connecting them to previous context when appropriate

Answer:
        """
        chat_prompt = PromptTemplate(
            input_variables=["conversation_context", "context", "question"],
            template=chat_prompt_template
        )
        print("✅ Chat prompt template created")

        # Load reranker
        if config.USE_RERANKER:
            reranker = CrossEncoder(config.RERANKER_MODEL)
            print("✅ Reranker loaded")

        print("🎉 All models initialized successfully!")

    except Exception as e:
        print(f"❌ Error initializing models: {str(e)}")
        raise

# ========================================
# MODIFIED: SERVER STARTUP FOR AWS
# ========================================

def start_aws_server():
    """Start the FastAPI server for AWS deployment"""
    print("=" * 60)
    print("🚀 CONVERSATIONAL RAG SYSTEM - AWS DEPLOYMENT")
    print("=" * 60)
    print("🔑 API Keys:")
    print("   • demo-api-key-123 (full access)")
    print("   • eval-key-456 (evaluation access)")
    print("")
    print("🎯 ENDPOINTS will be available at:")
    print("  • POST /api/v1/chat - Chat with the system")
    print("  • POST /api/v1/index - Index new URLs")
    print("  • GET  /api/v1/sources - List all sources")
    print("  • GET  /api/v1/sources/{hash} - Get source details")
    print("  • POST /api/v1/evaluate - Run automated evaluation")
    print("  • GET  /api/v1/conversations - List conversations")
    print("  • GET  /health - Health check")
    print("")
    print("🌐 Server starting on 0.0.0.0:8000...")
    print("📚 API docs will be at: http://your-ec2-ip:8000/docs")
    print("=" * 60)

    # Start the server without ngrok
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ========================================
# FASTAPI APPLICATION (Same as before)
# ========================================

app = FastAPI(
    title="Conversational RAG API with Dynamic Indexing and Sources",
    description="Hybrid Retrieval-Augmented Generation API with Conversation Support, BM25 + FAISS, Dynamic Web Indexing, and Source Management",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Authentication (same as before)
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header"""
    api_key = credentials.credentials
    if api_key not in config.VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return config.VALID_API_KEYS[api_key]

# ========================================
# ALL YOUR EXISTING ENDPOINTS (Copy exactly as they are)
# ========================================

# [Include all your existing endpoints here - no changes needed]
# @app.get("/health", response_model=HealthResponse)
# @app.post("/api/v1/chat", response_model=ChatResponse)
# @app.get("/api/v1/sources", response_model=SourcesResponse)
# etc.
# ========================================
# STEP 11: API ENDPOINTS
# ========================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    clean_old_conversations()  # Cleanup old conversations on health check

    components = {
        "static_vectorstore": static_vectorstore is not None,
        "dynamic_vectorstore": dynamic_vectorstore is not None,
        "llm": llm is not None,
        "bm25_static": bm25_static is not None,
        "bm25_dynamic": bm25_dynamic is not None,
        "reranker": reranker is not None if config.USE_RERANKER else "disabled",
        "conversation_manager": True
    }
    status = "healthy" if llm is not None else "unhealthy"
    return HealthResponse(
        status=status,
        components=components,
        conversations_active=len(active_conversations)
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    user_info: dict = Depends(verify_api_key)
):
    """Main conversational RAG endpoint"""
    try:
        # Check permissions
        if "chat" not in user_info.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for chat"
            )

        logger.info(f"Chat request from user {user_info['user']}")

        # Get or create session
        session_id = conversation_manager.get_or_create_session(request.session_id)

        # Add conversation history to session (except the last message which is the current question)
        for message in request.messages[:-1]:
            conversation_manager.add_message(session_id, message)

        # Get current question
        current_message = request.messages[-1]
        if current_message.role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user"
            )

        # Add current question to conversation
        conversation_manager.add_message(session_id, current_message)

        # Get conversation context for retrieval and generation
        conversation_context = conversation_manager.get_conversation_context(session_id)

        # Perform hybrid search with conversation context
        docs = hybrid_search(
            current_message.content,
            use_dynamic=request.use_dynamic_index,
            bm_k=config.BM25_TOP_K,
            faiss_k=config.FAISS_TOP_K,
            top_k=request.top_k,
            use_reranker=request.use_reranker and config.USE_RERANKER,
            conversation_context=conversation_context
        )

        if not docs:
            answer = "I couldn't find specific information about that in my knowledge base."
            sources = []
        else:
            # Create context and query LLM
            doc_context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = chat_prompt.format(
                conversation_context=conversation_context,
                context=doc_context,
                question=current_message.content
            )

            response = llm.invoke(formatted_prompt)
            answer = response.content.strip()

            # Get unique sources
            sources = list(OrderedDict.fromkeys(doc.metadata.get("source", "Unknown") for doc in docs))
            sources = format_sources(sources)

        # Add assistant response to conversation
        assistant_message = ChatMessage(role="assistant", content=answer)
        conversation_manager.add_message(session_id, assistant_message)

        # Save conversation periodically
        if conversation_manager.get_conversation_length(session_id) % 4 == 0:  # Every 4 messages
            conversation_manager.save_conversation(session_id)

        # UPDATED RESPONSE STRUCTURE WITH ENHANCED SOURCES
        response_data = {
            "answer": {
                "content": answer,
                "role": "assistant"
            },
            "sources": sources,  # Make sources more prominent
            "citations": sources,  # Keep backward compatibility
            "retrieved_documents": [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "title": doc.metadata.get("title", ""),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                } for doc in docs
            ] if docs else [],
            "metadata": {
                "num_docs_retrieved": len(docs),
                "num_sources": len(sources),
                "reranker_used": request.use_reranker and config.USE_RERANKER,
                "dynamic_index_used": request.use_dynamic_index,
                "conversation_length": conversation_manager.get_conversation_length(session_id),
                "user": user_info["user"]
            }
        }

        return ChatResponse(
            session_id=session_id,
            response=response_data,
            conversation_length=conversation_manager.get_conversation_length(session_id)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# NEW: Sources management endpoints
@app.get("/api/v1/sources", response_model=SourcesResponse)
async def get_sources(user_info: dict = Depends(verify_api_key)):
    """Get all available sources in the system"""
    try:
        # Check permissions
        if "read" not in user_info.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to view sources"
            )

        static_sources = []
        dynamic_sources = []

        # Get static sources
        if static_vectorstore:
            static_source_map = {}
            for doc_id, doc in static_vectorstore.docstore._dict.items():
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "")

                if source not in static_source_map:
                    static_source_map[source] = {
                        "count": 0,
                        "title": title
                    }
                static_source_map[source]["count"] += 1

            for source, info in static_source_map.items():
                # Convert file paths to URLs using source_url_map
                display_source = source_url_map.get(source, source)
                static_sources.append(SourceInfo(
                    source_url=display_source,
                    title=info["title"],
                    document_count=info["count"],
                    source_type="static",
                    indexed_at=None,  # Static sources don't have index timestamp
                    last_updated=None
                ))

        # Get dynamic sources
        if dynamic_vectorstore:
            dynamic_source_map = {}
            for doc_id, doc in dynamic_vectorstore.docstore._dict.items():
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "")
                indexed_at = doc.metadata.get("indexed_at")

                if source not in dynamic_source_map:
                    dynamic_source_map[source] = {
                        "count": 0,
                        "title": title,
                        "indexed_at": indexed_at
                    }
                dynamic_source_map[source]["count"] += 1

                # Keep the most recent indexed_at
                if indexed_at and (not dynamic_source_map[source]["indexed_at"] or indexed_at > dynamic_source_map[source]["indexed_at"]):
                    dynamic_source_map[source]["indexed_at"] = indexed_at

            for source, info in dynamic_source_map.items():
                indexed_datetime = None
                if info["indexed_at"]:
                    try:
                        indexed_datetime = datetime.fromisoformat(info["indexed_at"]) if isinstance(info["indexed_at"], str) else info["indexed_at"]
                    except:
                        indexed_datetime = None

                dynamic_sources.append(SourceInfo(
                    source_url=source,
                    title=info["title"],
                    document_count=info["count"],
                    source_type="dynamic",
                    indexed_at=indexed_datetime,
                    last_updated=indexed_datetime
                ))

        # Sort sources by document count (descending)
        static_sources.sort(key=lambda x: x.document_count, reverse=True)
        dynamic_sources.sort(key=lambda x: x.document_count, reverse=True)

        return SourcesResponse(
            total_sources=len(static_sources) + len(dynamic_sources),
            static_sources=static_sources,
            dynamic_sources=dynamic_sources
        )

    except Exception as e:
        logger.error(f"Error retrieving sources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sources: {str(e)}"
        )

@app.get("/api/v1/sources/{source_hash}")
async def get_source_details(
    source_hash: str,
    user_info: dict = Depends(verify_api_key)
):
    """Get detailed information about a specific source"""
    # Check permissions
    if "read" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view source details"
        )

    # Find source by hash or URL
    source_found = False
    source_details = {
        "source_url": None,
        "title": None,
        "chunks": [],
        "source_type": None,
        "indexed_at": None
    }

    # Search in both vectorstores
    for vectorstore, source_type in [(static_vectorstore, "static"), (dynamic_vectorstore, "dynamic")]:
        if vectorstore:
            for doc_id, doc in vectorstore.docstore._dict.items():
                source = doc.metadata.get("source", "")
                url_hash = get_url_hash(source)

                if url_hash == source_hash or source.endswith(source_hash):
                    source_found = True
                    source_details.update({
                        "source_url": source_url_map.get(source, source),
                        "title": doc.metadata.get("title", ""),
                        "source_type": source_type,
                        "indexed_at": doc.metadata.get("indexed_at")
                    })

                    source_details["chunks"].append({
                        "chunk_id": doc.metadata.get("chunk_id", 0),
                        "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "content_length": len(doc.page_content)
                    })

    if not source_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found"
        )

    return source_details

@app.post("/api/v1/index", response_model=IndexResponse)
async def index_urls(
    request: IndexRequest,
    user_info: dict = Depends(verify_api_key)
):
    """Index URLs into the dynamic vector database"""
    global dynamic_vectorstore, bm25_dynamic, bm25_docs_dynamic, bm25_sources_dynamic, text_to_docs_dynamic

    # Check permissions
    if "index" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for indexing"
        )

    logger.info(f"Indexing request from {user_info['user']}: {len(request.url)} URLs")

    scraper = WebScraper()
    indexed_urls = []
    failed_urls = []

    try:
        new_documents = []

        for url in request.url:
            try:
                logger.info(f"Processing URL: {url}")

                # Extract content
                result = scraper.extract_content(url)

                if not result['success']:
                    failed_urls.append({
                        "url": url,
                        "error": result['error'],
                        "error_type": "EXTRACTION_FAILED"
                    })
                    continue

                # Split content into chunks
                chunks = text_splitter.split_text(result['content'])

                # Create documents
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "title": result['title'],
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "indexed_at": datetime.now().isoformat(),
                            "url_hash": get_url_hash(url)
                        }
                    )
                    new_documents.append(doc)

                indexed_urls.append(url)
                logger.info(f"✅ Successfully processed {url} - {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                failed_urls.append({
                    "url": url,
                    "error": str(e),
                    "error_type": "PROCESSING_ERROR"
                })

        # Update vector database if we have new documents
        if new_documents:
            try:
                if dynamic_vectorstore is None:
                    # Create new FAISS index
                    dynamic_vectorstore = FAISS.from_documents(new_documents, embedding_model)
                    logger.info("✅ Created new dynamic FAISS index")
                else:
                    # Add to existing index
                    dynamic_vectorstore.add_documents(new_documents)
                    logger.info(f"✅ Added {len(new_documents)} documents to existing index")

                # Save updated index
                dynamic_vectorstore.save_local(config.DYNAMIC_FAISS_PATH)

                # Rebuild BM25 and mappings
                bm25_dynamic, bm25_docs_dynamic, bm25_sources_dynamic, text_to_docs_dynamic = build_bm25_from_vectorstore(dynamic_vectorstore)

                logger.info(f"✅ Dynamic index updated")

            except Exception as e:
                logger.error(f"Error updating vector database: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update vector database: {str(e)}"
                )

        # Prepare response
        response_status = "success" if indexed_urls else "failed"
        if indexed_urls and failed_urls:
            response_status = "partial_success"

        metadata = {
            "total_requested": len(request.url),
            "successfully_indexed": len(indexed_urls),
            "failed": len(failed_urls),
            "new_documents_added": len(new_documents),
            "user": user_info["user"]
        }

        return IndexResponse(
            status=response_status,
            indexed_url=indexed_urls,
            failed_url=failed_urls if failed_urls else None,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Critical error in indexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
async def evaluate_system(
    request: EvaluationRequest,
    user_info: dict = Depends(verify_api_key)
):
    """Run automated evaluation of the RAG system"""
    # Check permissions
    if "eval" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for evaluation"
        )

    logger.info(f"Evaluation request from user {user_info['user']}")

    try:
        # Use provided test cases or create default ones
        test_cases = request.test_cases if request.test_cases else evaluator.create_test_dataset()

        # Run evaluation
        evaluation_results = await evaluator.run_evaluation(test_cases)

        return EvaluationResponse(
            overall_score=evaluation_results["overall_score"],
            detailed_results=evaluation_results["detailed_results"],
            metrics=evaluation_results["metrics"]
        )

    except Exception as e:
        logger.error(f"Error running evaluation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )

@app.get("/api/v1/conversations/{session_id}")
async def get_conversation(
    session_id: str,
    user_info: dict = Depends(verify_api_key)
):
    """Get conversation history"""
    if session_id not in active_conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    conversation_data = active_conversations[session_id]
    return {
        "session_id": session_id,
        "messages": conversation_data["messages"],
        "created_at": conversation_data["created_at"],
        "last_active": conversation_data["last_active"],
        "message_count": len(conversation_data["messages"])
    }

@app.delete("/api/v1/conversations/{session_id}")
async def delete_conversation(
    session_id: str,
    user_info: dict = Depends(verify_api_key)
):
    """Delete a conversation"""
    if session_id not in active_conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    del active_conversations[session_id]
    return {"status": "deleted", "session_id": session_id}

@app.get("/api/v1/conversations")
async def list_conversations(user_info: dict = Depends(verify_api_key)):
    """List active conversations"""
    clean_old_conversations()

    conversations_summary = []
    for session_id, conv_data in active_conversations.items():
        conversations_summary.append({
            "session_id": session_id,
            "message_count": len(conv_data["messages"]),
            "created_at": conv_data["created_at"],
            "last_active": conv_data["last_active"],
            "last_message_preview": conv_data["messages"][-1]["content"][:100] if conv_data["messages"] else ""
        })

    return {
        "active_conversations": len(conversations_summary),
        "conversations": conversations_summary
    }

print("✅ API endpoints defined!")
# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("🔄 Initializing conversational RAG system for AWS...")
    
    # Initialize models first
    initialize_models()
    
    # Create demo test cases
    print("\n📝 Creating demo test cases...")
    # create_demo_test_cases()  # Include this function from your original code
    
    # Print startup info
    print("\n🎓 SYSTEM READY FOR DEPLOYMENT!")
    print("✅ All models loaded")
    print("✅ API endpoints configured") 
    print("✅ Ready for recruiter demos")
    
    # Start the AWS server
    start_aws_server()
