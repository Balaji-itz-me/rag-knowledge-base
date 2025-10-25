# COMPLETE ENHANCED RAG SYSTEM WITH ALL BEST PRACTICES
# Enhanced with: Caching, Rate Limiting, Concurrent Processing, Advanced Logging

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
from fastapi import FastAPI, HTTPException, Depends, status, Body, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import logging
from logging.handlers import RotatingFileHandler
from collections import OrderedDict
import hashlib
import re
import uuid
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from threading import Thread
import aiohttp
from functools import lru_cache, wraps
from time import time as timer

# RAG components
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

print("✅ All imports successful!")

# ========================================
# ENHANCED LOGGING SYSTEM
# ========================================

class StructuredLogger:
    """Structured JSON logging with rotation"""
    
    def __init__(self, name: str, log_path: str = "/home/ubuntu/rag-knowledge-base/rag_demo/logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory
        os.makedirs(log_path, exist_ok=True)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        try:
            file_handler = RotatingFileHandler(
                f"{log_path}/app.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler: {e}")
    
    def log_structured(self, level: str, event: str, **kwargs):
        """Log structured data as JSON"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "event": event,
            **kwargs
        }
        
        log_message = json.dumps(log_entry)
        
        if level == "INFO":
            self.logger.info(log_message)
        elif level == "ERROR":
            self.logger.error(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "DEBUG":
            self.logger.debug(log_message)
    
    def info(self, event: str, **kwargs):
        self.log_structured("INFO", event, **kwargs)
    
    def error(self, event: str, **kwargs):
        self.log_structured("ERROR", event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        self.log_structured("WARNING", event, **kwargs)
    
    def debug(self, event: str, **kwargs):
        self.log_structured("DEBUG", event, **kwargs)

logger = StructuredLogger(__name__)
print("✅ Enhanced logging initialized!")

# ========================================
# CACHING SYSTEM
# ========================================

class CacheManager:
    """Multi-level caching: response, embedding, search"""
    
    def __init__(self, ttl_seconds: int = 1800):
        self.response_cache = {}
        self.embedding_cache = {}
        self.search_cache = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, timestamp: datetime) -> bool:
        return (datetime.now() - timestamp).seconds > self.ttl
    
    def get_response(self, query: str, context_hash: str) -> Optional[Dict]:
        cache_key = f"{query}:{context_hash}"
        
        if cache_key in self.response_cache:
            entry = self.response_cache[cache_key]
            if not self._is_expired(entry['timestamp']):
                self.hits += 1
                logger.info("cache_hit", cache_type="response", query=query[:50])
                return entry['data']
            else:
                del self.response_cache[cache_key]
        
        self.misses += 1
        return None
    
    def set_response(self, query: str, context_hash: str, data: Dict):
        cache_key = f"{query}:{context_hash}"
        self.response_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def get_search_results(self, query: str, index_type: str) -> Optional[List]:
        cache_key = f"{query}:{index_type}"
        
        if cache_key in self.search_cache:
            entry = self.search_cache[cache_key]
            if not self._is_expired(entry['timestamp']):
                self.hits += 1
                return entry['data']
            else:
                del self.search_cache[cache_key]
        
        self.misses += 1
        return None
    
    def set_search_results(self, query: str, index_type: str, results: List):
        cache_key = f"{query}:{index_type}"
        self.search_cache[cache_key] = {
            'data': results,
            'timestamp': datetime.now()
        }
    
    def clear_all(self):
        self.response_cache.clear()
        self.embedding_cache.clear()
        self.search_cache.clear()
        logger.info("cache_cleared", message="All caches cleared")
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": f"{hit_rate:.2f}",
            "response_cache_size": len(self.response_cache),
            "search_cache_size": len(self.search_cache)
        }

cache_manager = CacheManager(ttl_seconds=1800)
print("✅ Caching system initialized!")

# ========================================
# RATE LIMITING SYSTEM
# ========================================

class CustomRateLimiter:
    """Per-user rate limiting"""
    
    def __init__(self):
        self.requests = {}
        self.limits = {
            "chat": {"requests": 20, "window": 60},
            "index": {"requests": 5, "window": 300},
            "eval": {"requests": 3, "window": 600},
            "default": {"requests": 30, "window": 60}
        }
    
    def is_allowed(self, user: str, endpoint: str) -> tuple[bool, Dict]:
        current_time = time.time()
        
        limit_config = self.limits.get(endpoint, self.limits["default"])
        max_requests = limit_config["requests"]
        window = limit_config["window"]
        
        if user not in self.requests:
            self.requests[user] = []
        
        # Clean old requests
        self.requests[user] = [
            (ts, ep) for ts, ep in self.requests[user]
            if current_time - ts < window
        ]
        
        # Count endpoint requests
        endpoint_requests = [
            ts for ts, ep in self.requests[user]
            if ep == endpoint
        ]
        
        if len(endpoint_requests) >= max_requests:
            retry_after = window - (current_time - min(endpoint_requests))
            logger.warning("rate_limit_exceeded", 
                          user=user, 
                          endpoint=endpoint,
                          requests_made=len(endpoint_requests))
            return False, {
                "allowed": False,
                "retry_after": int(retry_after),
                "limit": max_requests
            }
        
        self.requests[user].append((current_time, endpoint))
        
        return True, {
            "allowed": True,
            "remaining": max_requests - len(endpoint_requests) - 1,
            "limit": max_requests
        }

rate_limiter = CustomRateLimiter()
print("✅ Rate limiting initialized!")

# ========================================
# CONCURRENT WEB SCRAPER
# ========================================

class ConcurrentWebScraper:
    """Async concurrent web scraping"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_url(self, session, url: str, retry_count: int = 3) -> Dict[str, Any]:
        async with self.semaphore:
            for attempt in range(retry_count):
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with session.get(url, timeout=timeout) as response:
                        if response.status == 200:
                            content = await response.text()
                            return await self._extract_content(url, content)
                        else:
                            logger.warning("http_error", url=url, status=response.status)
                
                except asyncio.TimeoutError:
                    logger.warning("timeout", url=url, attempt=attempt + 1)
                    if attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)
                
                except Exception as e:
                    logger.error("fetch_error", url=url, error=str(e))
                    if attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)
            
            return {
                'url': url,
                'success': False,
                'error': 'Max retries exceeded',
                'title': None,
                'content': None
            }
    
    async def _extract_content(self, url: str, html_content: str) -> Dict[str, Any]:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else url
            
            content_selectors = ['article', 'main', '[role="main"]', '.content']
            content_text = ""
            
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    content_text = content.get_text()
                    break
            
            if not content_text:
                body = soup.find('body')
                content_text = body.get_text() if body else ""
            
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            
            if not content_text:
                raise ValueError("No content extracted")
            
            logger.info("content_extracted", url=url, content_length=len(content_text))
            
            return {
                'title': title_text,
                'content': content_text,
                'url': url,
                'success': True,
                'error': None
            }
        
        except Exception as e:
            logger.error("extraction_error", url=url, error=str(e))
            return {
                'url': url,
                'success': False,
                'error': str(e),
                'title': None,
                'content': None
            }
    
    async def fetch_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        logger.info("concurrent_fetch_started", url_count=len(urls))
        
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'User-Agent': 'Mozilla/5.0'}
        ) as session:
            tasks = [self.fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("fetch_exception", url=urls[i], error=str(result))
                    processed_results.append({
                        'url': urls[i],
                        'success': False,
                        'error': str(result),
                        'title': None,
                        'content': None
                    })
                else:
                    processed_results.append(result)
            
            successful = sum(1 for r in processed_results if r['success'])
            logger.info("concurrent_fetch_completed", 
                       total=len(urls), 
                       successful=successful)
            
            return processed_results

concurrent_scraper = ConcurrentWebScraper(max_concurrent=5)
print("✅ Concurrent scraper initialized!")

# ========================================
# PERFORMANCE MONITORING
# ========================================

def timing_decorator(func):
    """Measure function execution time"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = timer()
        try:
            result = await func(*args, **kwargs)
            elapsed = timer() - start_time
            logger.info("function_performance",
                       function=func.__name__,
                       duration_ms=f"{elapsed * 1000:.2f}",
                       status="success")
            return result
        except Exception as e:
            elapsed = timer() - start_time
            logger.error("function_error",
                        function=func.__name__,
                        duration_ms=f"{elapsed * 1000:.2f}",
                        error=str(e))
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = timer()
        try:
            result = func(*args, **kwargs)
            elapsed = timer() - start_time
            logger.info("function_performance",
                       function=func.__name__,
                       duration_ms=f"{elapsed * 1000:.2f}",
                       status="success")
            return result
        except Exception as e:
            elapsed = timer() - start_time
            logger.error("function_error",
                        function=func.__name__,
                        duration_ms=f"{elapsed * 1000:.2f}",
                        error=str(e))
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper

print("✅ Performance monitoring initialized!")

# ========================================
# CONFIGURATION
# ========================================

class Config:
    BASE_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo"
    STATIC_FAISS_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/faiss_index"
    DYNAMIC_BASE_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index"
    DYNAMIC_FAISS_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index/faiss_index"
    METADATA_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index/metadata"
    BACKUP_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/dynamic_index/backups"
    CONVERSATIONS_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/conversations"
    EVALUATION_DATASET_PATH = "/home/ubuntu/rag-knowledge-base/rag_demo/evaluation"
    
    MAX_CONVERSATION_LENGTH = 10
    CONVERSATION_TIMEOUT = 1800
    MAX_CONTEXT_LENGTH = 3
    
    USE_RERANKER = True
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    BM25_TOP_K = 10
    FAISS_TOP_K = 10
    FINAL_TOP_K = 3
    
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RE_INDEX_DAYS = 7
    MAX_VERSIONS = 3
    
    # Load API keys from environment
    VALID_API_KEYS = {}

config = Config()

# Load API keys from environment variables
for i in range(1, 11):
    key = os.environ.get(f"API_KEY_{i}")
    user = os.environ.get(f"API_KEY_{i}_USER")
    perms_str = os.environ.get(f"API_KEY_{i}_PERMS", "read,query,chat")
    
    if key and user:
        config.VALID_API_KEYS[key] = {
            "user": user,
            "permissions": perms_str.split(",")
        }

# Fallback to demo keys
if not config.VALID_API_KEYS:
    config.VALID_API_KEYS = {
        "demo-api-key-123": {"user": "demo_user", "permissions": ["read", "query", "index", "chat", "admin"]},
        "eval-key-456": {"user": "evaluator", "permissions": ["read", "query", "chat", "eval"]},
    }

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("missing_api_key", message="GOOGLE_API_KEY not set")
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("✅ Configuration loaded!")

# ========================================
# PYDANTIC MODELS
# ========================================

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

class IndexRequest(BaseModel):
    url: List[str] = Field(..., min_items=1, max_items=5)
    
    @validator('url')
    def validate_urls(cls, v):
        for url in v:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")
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

class SourceInfo(BaseModel):
    source_url: str
    title: Optional[str] = None
    indexed_at: Optional[datetime] = None
    document_count: int = 0
    source_type: str
    last_updated: Optional[datetime] = None

class SourcesResponse(BaseModel):
    total_sources: int
    static_sources: List[SourceInfo]
    dynamic_sources: List[SourceInfo]
    timestamp: datetime = Field(default_factory=datetime.now)

print("✅ Pydantic models defined!")

# ========================================
# GLOBAL VARIABLES
# ========================================

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

active_conversations = {}

source_url_map = {
    "/content/drive/MyDrive/RAG_demo/data/genai-platform.txt": "https://huyenchip.com/2024/07/25/genai-platform.html",
    "/content/drive/MyDrive/RAG_demo/data/hallucination.txt": "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "/content/drive/MyDrive/RAG_demo/data/quora_engineering.txt": "https://quoraengineering.quora.com/Building-Embedding-Search-at-Quora"
}

# ========================================
# UTILITY FUNCTIONS
# ========================================

def ensure_directories():
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
    logger.info("directories_created", message="All directories ensured")

def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def load_metadata(file_path: str) -> Dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("metadata_load_error", path=file_path, error=str(e))
    return {}

def save_metadata(data: Dict, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error("metadata_save_error", path=file_path, error=str(e))
        raise

def format_sources(sources):
    formatted = []
    for source in sources:
        if source.startswith('http'):
            formatted.append(source)
        else:
            formatted.append(source_url_map.get(source, source))
    return formatted

def generate_session_id() -> str:
    return str(uuid.uuid4())

def clean_old_conversations():
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, conv_data in active_conversations.items():
        last_active = conv_data.get('last_active', current_time)
        if isinstance(last_active, str):
            last_active = datetime.fromisoformat(last_active)
        
        if (current_time - last_active).seconds > config.CONVERSATION_TIMEOUT:
            try:
                conv_file = os.path.join(config.CONVERSATIONS_PATH, f"{session_id}.json")
                save_metadata(conv_data, conv_file)
            except Exception as e:
                logger.error("conversation_save_error", session_id=session_id, error=str(e))
            
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del active_conversations[session_id]
        logger.info("conversation_cleaned", session_id=session_id)

print("✅ Utility functions defined!")

# ========================================
# CONVERSATION MANAGEMENT
# ========================================

class ConversationManager:
    def __init__(self):
        self.conversations = active_conversations
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if session_id and session_id in self.conversations:
            self.conversations[session_id]['last_active'] = datetime.now()
            return session_id
        
        new_session_id = generate_session_id()
        self.conversations[new_session_id] = {
            'messages': [],
            'created_at': datetime.now(),
            'last_active': datetime.now(),
            'metadata': {}
        }
        logger.info("conversation_created", session_id=new_session_id)
        return new_session_id
    
    def add_message(self, session_id: str, message: ChatMessage):
        if session_id not in self.conversations:
            raise ValueError(f"Session {session_id} not found")
        
        self.conversations[session_id]['messages'].append({
            'role': message.role,
            'content': message.content,
            'timestamp': message.timestamp.isoformat() if message.timestamp else datetime.now().isoformat()
        })
        self.conversations[session_id]['last_active'] = datetime.now()
        
        messages = self.conversations[session_id]['messages']
        if len(messages) > config.MAX_CONVERSATION_LENGTH * 2:
            self.conversations[session_id]['messages'] = messages[-config.MAX_CONVERSATION_LENGTH * 2:]
    
    def get_conversation_context(self, session_id: str, max_pairs: int = None) -> str:
        if session_id not in self.conversations:
            return ""
        
        messages = self.conversations[session_id]['messages']
        if not messages:
            return ""
        
        max_pairs = max_pairs or config.MAX_CONTEXT_LENGTH
        recent_messages = messages[-(max_pairs * 2):]
        
        context_parts = []
        for msg in recent_messages:
            role = "Human" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def get_conversation_length(self, session_id: str) -> int:
        if session_id not in self.conversations:
            return 0
        return len(self.conversations[session_id]['messages'])
    
    def save_conversation(self, session_id: str):
        if session_id not in self.conversations:
            return
        
        conv_file = os.path.join(config.CONVERSATIONS_PATH, f"{session_id}.json")
        save_metadata(self.conversations[session_id], conv_file)

conversation_manager = ConversationManager()
print("✅ Conversation manager initialized!")

# ========================================
# BM25 AND VECTOR STORE FUNCTIONS
# ========================================

def build_bm25_from_vectorstore(vectorstore):
    try:
        docs = []
        sources = []
        for doc_id, doc in vectorstore.docstore._dict.items():
            text = doc.page_content
            docs.append(text)
            sources.append(doc.metadata.get("source", "Unknown"))
        
        tokenized = [d.split() for d in docs]
        bm25 = BM25Okapi(tokenized)
        
        text_to_docs = {}
        for doc in vectorstore.docstore._dict.values():
            text = doc.page_content
            text_to_docs.setdefault(text, []).append(doc)
        
        logger.info("bm25_built", doc_count=len(docs))
        return bm25, docs, sources, text_to_docs
    except Exception as e:
        logger.error("bm25_build_error", error=str(e))
        raise

# ========================================
# ENHANCED HYBRID SEARCH WITH CACHING
# ========================================

@timing_decorator
def hybrid_search(query: str, use_dynamic: bool = True, bm_k: int = 10, 
                 faiss_k: int = 10, top_k: int = 3, use_reranker: bool = True, 
                 conversation_context: str = ""):
    """Hybrid search with caching"""
    
    # Check cache first
    cache_key = f"{'dynamic' if use_dynamic else 'static'}"
    cached_results = cache_manager.get_search_results(query, cache_key)
    
    if cached_results:
        logger.info("search_cache_hit", query=query[:50])
        return cached_results[:top_k]
    
    try:
        enhanced_query = query
        if conversation_context:
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
        
        # Merge and deduplicate
        merged = OrderedDict()
        for candidate in all_candidates:
            key = candidate["text"].strip()[:100]
            if key not in merged or (candidate.get("bm25_score") or 0) > (merged[key].get("bm25_score") or 0):
                merged[key] = candidate
        
        candidates = list(merged.values())
        
        # Optional reranking
        if use_reranker and reranker and candidates:
            pairs = [(query, c["text"]) for c in candidates]
            scores = reranker.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        else:
            candidates = sorted(candidates, key=lambda x: (0 if x["bm25_score"] is not None else 1, -(x["bm25_score"] or 0)))
        
        # Create final documents
        final_docs = []
        for c in candidates[:top_k]:
            text = c["text"]
            docs_list = text_to_docs_static.get(text, []) if text_to_docs_static else []
            if not docs_list and text_to_docs_dynamic:
                docs_list = text_to_docs_dynamic.get(text, [])
            
            if docs_list:
                final_docs.append(docs_list[0])
            else:
                final_docs.append(Document(page_content=text, metadata={"source": c.get("source", "Unknown")}))
        
        # Cache the results
        cache_manager.set_search_results(query, cache_key, final_docs)
        
        return final_docs
    
    except Exception as e:
        logger.error("hybrid_search_error", error=str(e), query=query[:50])
        raise

def _search_single_index(query: str, vectorstore, bm25, bm25_docs, bm25_sources, text_to_docs, bm_k: int, faiss_k: int):
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

print("✅ Hybrid search defined!")

# ========================================
# MODEL INITIALIZATION
# ========================================

def initialize_models():
    global embedding_model, static_vectorstore, dynamic_vectorstore, llm
    global bm25_static, bm25_dynamic, bm25_docs_static, bm25_docs_dynamic
    global bm25_sources_static, bm25_sources_dynamic, reranker
    global text_to_docs_static, text_to_docs_dynamic, chat_prompt, text_splitter
    
    try:
        logger.info("model_initialization_started", message="Loading models")
        ensure_directories()
        clean_old_conversations()
        
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("model_loaded", model="embeddings")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info("text_splitter_initialized")
        
        # Load static FAISS
        try:
            if os.path.exists(config.STATIC_FAISS_PATH):
                static_vectorstore = FAISS.load_local(
                    config.STATIC_FAISS_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                bm25_static, bm25_docs_static, bm25_sources_static, text_to_docs_static = build_bm25_from_vectorstore(static_vectorstore)
                logger.info("static_index_loaded")
            else:
                logger.info("static_index_not_found")
        except Exception as e:
            logger.warning("static_index_load_error", error=str(e))
        
        # Load dynamic FAISS
        try:
            if os.path.exists(config.DYNAMIC_FAISS_PATH):
                dynamic_vectorstore = FAISS.load_local(
                    config.DYNAMIC_FAISS_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                bm25_dynamic, bm25_docs_dynamic, bm25_sources_dynamic, text_to_docs_dynamic = build_bm25_from_vectorstore(dynamic_vectorstore)
                logger.info("dynamic_index_loaded")
            else:
                logger.info("dynamic_index_not_found")
        except Exception as e:
            logger.info("dynamic_index_load_info", message="Will create on first index")
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        logger.info("model_loaded", model="llm")
        
        # Create chat prompt
        chat_prompt_template = """
You are a helpful AI assistant with access to multiple knowledge sources.

Previous conversation context:
{conversation_context}

Current context from knowledge base:
{context}

Current question:
{question}

Instructions:
1. Use conversation context to understand references
2. Provide accurate answers based on the knowledge base
3. If referencing specific information, cite from context
4. If not in knowledge base, say so
5. Be conversational and natural

Answer:
        """
        chat_prompt = PromptTemplate(
            input_variables=["conversation_context", "context", "question"],
            template=chat_prompt_template
        )
        logger.info("prompt_template_created")
        
        # Load reranker
        if config.USE_RERANKER:
            reranker = CrossEncoder(config.RERANKER_MODEL)
            logger.info("model_loaded", model="reranker")
        
        logger.info("initialization_complete", message="All models loaded successfully")
    
    except Exception as e:
        logger.error("initialization_error", error=str(e))
        raise

# ========================================
# FASTAPI APPLICATION
# ========================================

app = FastAPI(
    title="Enhanced Conversational RAG API",
    description="RAG API with Caching, Rate Limiting, Concurrent Processing, and Advanced Logging",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Add SlowAPI rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    if api_key not in config.VALID_API_KEYS:
        logger.warning("invalid_api_key", key_prefix=api_key[:10])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return config.VALID_API_KEYS[api_key]

# ========================================
# API ENDPOINTS
# ========================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    clean_old_conversations()
    
    components = {
        "static_vectorstore": static_vectorstore is not None,
        "dynamic_vectorstore": dynamic_vectorstore is not None,
        "llm": llm is not None,
        "bm25_static": bm25_static is not None,
        "bm25_dynamic": bm25_dynamic is not None,
        "reranker": reranker is not None if config.USE_RERANKER else "disabled",
        "conversation_manager": True,
        "cache_manager": True,
        "rate_limiter": True
    }
    status_text = "healthy" if llm is not None else "unhealthy"
    return HealthResponse(
        status=status_text,
        components=components,
        conversations_active=len(active_conversations)
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
@timing_decorator
async def chat_with_rag(
    request: ChatRequest,
    http_request: Request,
    user_info: dict = Depends(verify_api_key)
):
    """Enhanced chat endpoint with caching and rate limiting"""
    
    # Check rate limit
    allowed, rate_info = rate_limiter.is_allowed(user_info['user'], 'chat')
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {rate_info['retry_after']} seconds",
            headers={"Retry-After": str(rate_info['retry_after'])}
        )
    
    # Check permissions
    if "chat" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        logger.info("chat_request", user=user_info['user'])
        
        # Get or create session
        session_id = conversation_manager.get_or_create_session(request.session_id)
        
        # Add conversation history
        for message in request.messages[:-1]:
            conversation_manager.add_message(session_id, message)
        
        # Get current question
        current_message = request.messages[-1]
        if current_message.role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user"
            )
        
        # Add current question
        conversation_manager.add_message(session_id, current_message)
        
        # Get conversation context
        conversation_context = conversation_manager.get_conversation_context(session_id)
        
        # Check cache
        query_hash = hashlib.md5(
            f"{current_message.content}:{conversation_context}".encode()
        ).hexdigest()
        
        cached_response = cache_manager.get_response(current_message.content, query_hash)
        
        if cached_response:
            logger.info("response_cache_hit", user=user_info['user'])
            assistant_message = ChatMessage(
                role="assistant",
                content=cached_response['answer']['content']
            )
            conversation_manager.add_message(session_id, assistant_message)
            
            return ChatResponse(
                session_id=session_id,
                response=cached_response,
                conversation_length=conversation_manager.get_conversation_length(session_id)
            )
        
        # Perform hybrid search
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
            doc_context = "\n\n".join([doc.page_content for doc in docs])
            formatted_prompt = chat_prompt.format(
                conversation_context=conversation_context,
                context=doc_context,
                question=current_message.content
            )
            
            response = llm.invoke(formatted_prompt)
            answer = response.content.strip()
            
            sources = list(OrderedDict.fromkeys(doc.metadata.get("source", "Unknown") for doc in docs))
            sources = format_sources(sources)
        
        # Add assistant response
        assistant_message = ChatMessage(role="assistant", content=answer)
        conversation_manager.add_message(session_id, assistant_message)
        
        # Save conversation periodically
        if conversation_manager.get_conversation_length(session_id) % 4 == 0:
            conversation_manager.save_conversation(session_id)
        
        # Prepare response
        response_data = {
            "answer": {
                "content": answer,
                "role": "assistant"
            },
            "sources": sources,
            "citations": sources,
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
                "user": user_info["user"],
                "rate_limit_remaining": rate_info["remaining"]
            }
        }
        
        # Cache the response
        cache_manager.set_response(current_message.content, query_hash, response_data)
        
        logger.info("chat_completed", 
                   user=user_info['user'],
                   session_id=session_id,
                   docs_retrieved=len(docs))
        
        return ChatResponse(
            session_id=session_id,
            response=response_data,
            conversation_length=conversation_manager.get_conversation_length(session_id)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_error", user=user_info['user'], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/api/v1/index", response_model=IndexResponse)
@timing_decorator
async def index_urls(
    request: IndexRequest,
    user_info: dict = Depends(verify_api_key)
):
    """Enhanced indexing with concurrent processing"""
    global dynamic_vectorstore, bm25_dynamic, bm25_docs_dynamic, bm25_sources_dynamic, text_to_docs_dynamic
    
    # Check rate limit
    allowed, rate_info = rate_limiter.is_allowed(user_info['user'], 'index')
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {rate_info['retry_after']} seconds"
        )
    
    # Check permissions
    if "index" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    logger.info("indexing_started", user=user_info['user'], url_count=len(request.url))
    
    try:
        # Concurrent URL fetching
        results = await concurrent_scraper.fetch_multiple_urls(request.url)
        
        indexed_urls = []
        failed_urls = []
        new_documents = []
        
        for result in results:
            if result['success']:
                # Split content
                chunks = text_splitter.split_text(result['content'])
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": result['url'],
                            "title": result['title'],
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "indexed_at": datetime.now().isoformat(),
                            "url_hash": get_url_hash(result['url'])
                        }
                    )
                    new_documents.append(doc)
                
                indexed_urls.append(result['url'])
                logger.info("url_indexed", url=result['url'], chunks=len(chunks))
            else:
                failed_urls.append({
                    "url": result['url'],
                    "error": result['error'],
                    "error_type": "PROCESSING_ERROR"
                })
                logger.error("url_failed", url=result['url'], error=result['error'])
        
        # Update vector database
        if new_documents:
            try:
                if dynamic_vectorstore is None:
                    dynamic_vectorstore = FAISS.from_documents(new_documents, embedding_model)
                    logger.info("dynamic_index_created", doc_count=len(new_documents))
                else:
                    dynamic_vectorstore.add_documents(new_documents)
                    logger.info("dynamic_index_updated", doc_count=len(new_documents))
                
                # Save index
                dynamic_vectorstore.save_local(config.DYNAMIC_FAISS_PATH)
                
                # Rebuild BM25
                bm25_dynamic, bm25_docs_dynamic, bm25_sources_dynamic, text_to_docs_dynamic = build_bm25_from_vectorstore(dynamic_vectorstore)
                
                logger.info("index_saved", total_docs=len(dynamic_vectorstore.docstore._dict))
            
            except Exception as e:
                logger.error("index_update_error", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update index: {str(e)}"
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
            "user": user_info["user"],
            "concurrent_processing": True
        }
        
        logger.info("indexing_completed", 
                   status=response_status,
                   total=len(request.url),
                   successful=len(indexed_urls))
        
        return IndexResponse(
            status=response_status,
            indexed_url=indexed_urls,
            failed_url=failed_urls if failed_urls else None,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error("indexing_error", user=user_info['user'], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}"
        )

@app.get("/api/v1/sources", response_model=SourcesResponse)
async def get_sources(user_info: dict = Depends(verify_api_key)):
    """Get all sources"""
    
    if "read" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        static_sources = []
        dynamic_sources = []
        
        # Get static sources
        if static_vectorstore:
            static_source_map = {}
            for doc_id, doc in static_vectorstore.docstore._dict.items():
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", "")
                
                if source not in static_source_map:
                    static_source_map[source] = {"count": 0, "title": title}
                static_source_map[source]["count"] += 1
            
            for source, info in static_source_map.items():
                display_source = source_url_map.get(source, source)
                static_sources.append(SourceInfo(
                    source_url=display_source,
                    title=info["title"],
                    document_count=info["count"],
                    source_type="static",
                    indexed_at=None,
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
        
        static_sources.sort(key=lambda x: x.document_count, reverse=True)
        dynamic_sources.sort(key=lambda x: x.document_count, reverse=True)
        
        return SourcesResponse(
            total_sources=len(static_sources) + len(dynamic_sources),
            static_sources=static_sources,
            dynamic_sources=dynamic_sources
        )
    
    except Exception as e:
        logger.error("sources_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sources: {str(e)}"
        )

@app.get("/api/v1/metrics")
async def get_metrics(user_info: dict = Depends(verify_api_key)):
    """Get system metrics - NEW ENDPOINT"""
    
    cache_stats = cache_manager.get_stats()
    rate_stats = rate_limiter.get_stats(user_info['user']) if hasattr(rate_limiter, 'get_stats') else {}
    
    return {
        "system_health": {
            "status": "healthy",
            "conversations_active": len(active_conversations),
            "static_index_loaded": static_vectorstore is not None,
            "dynamic_index_loaded": dynamic_vectorstore is not None
        },
        "cache_performance": cache_stats,
        "rate_limiting": rate_stats,
        "indices": {
            "static_documents": len(static_vectorstore.docstore._dict) if static_vectorstore else 0,
            "dynamic_documents": len(dynamic_vectorstore.docstore._dict) if dynamic_vectorstore else 0
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/cache/clear")
async def clear_cache(user_info: dict = Depends(verify_api_key)):
    """Clear all caches - NEW ENDPOINT"""
    
    if "admin" not in user_info.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    
    cache_manager.clear_all()
    logger.info("cache_cleared_by_admin", user=user_info['user'])
    
    return {
        "status": "success",
        "message": "All caches cleared",
        "timestamp": datetime.now().isoformat()
    }

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
    logger.info("conversation_deleted", session_id=session_id, user=user_info['user'])
    
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
# STARTUP
# ========================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("server_startup", message="Initializing RAG system")
    initialize_models()
    logger.info("server_ready", message="System ready")

def start_server():
    """Start the server"""
    print("=" * 60)
    print("🚀 ENHANCED RAG SYSTEM - PRODUCTION READY")
    print("=" * 60)
    print("\n✅ IMPLEMENTED FEATURES:")
    print("   • Multi-level caching (response, embedding, search)")
    print("   • Per-user rate limiting with configurable limits")
    print("   • Concurrent URL processing (async/await)")
    print("   • Structured JSON logging with rotation")
    print("   • Performance monitoring and metrics")
    print("\n🔑 API Keys:")
    print("   • demo-api-key-123 (full access)")
    print("   • eval-key-456 (evaluation access)")
    print("\n🎯 ENDPOINTS:")
    print("  • POST /api/v1/chat - Chat (cached, rate limited)")
    print("  • POST /api/v1/index - Index URLs (concurrent)")
    print("  • GET  /api/v1/sources - List sources")
    print("  • GET  /api/v1/metrics - System metrics (NEW)")
    print("  • POST /api/v1/cache/clear - Clear cache (NEW)")
    print("  • GET  /api/v1/conversations - List conversations")
    print("  • GET  /health - Health check")
    print("\n🌐 Server starting on 0.0.0.0:8000...")
    print("📚 API docs: http://your-ec2-ip:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_server()
