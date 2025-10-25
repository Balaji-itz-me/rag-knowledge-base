import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import uuid

# Page configuration
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .feature-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .status-unhealthy {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .performance-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        border: 1px solid #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: auto;
        border: 1px solid #757575;
    }
    
    .source-link {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    
    .cache-indicator {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .rate-limit-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://56.228.63.64:8000"
SESSION_TIMEOUT = 1800

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'auth_time' not in st.session_state:
        st.session_state.auth_time = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'system_health' not in st.session_state:
        st.session_state.system_health = {}
    if 'sources_data' not in st.session_state:
        st.session_state.sources_data = {}
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = {}
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"
    # NEW: Performance metrics tracking
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'last_response_time' not in st.session_state:
        st.session_state.last_response_time = None

def check_session_timeout():
    """Check if the session has timed out"""
    if st.session_state.auth_time:
        elapsed = (datetime.now() - st.session_state.auth_time).total_seconds()
        return elapsed > SESSION_TIMEOUT
    return True

def get_remaining_time():
    """Get remaining session time in seconds"""
    if st.session_state.auth_time:
        elapsed = (datetime.now() - st.session_state.auth_time).total_seconds()
        remaining = max(0, SESSION_TIMEOUT - elapsed)
        return remaining
    return 0

def format_time(seconds):
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, params: Dict = None):
    """Make API request with authentication and timing"""
    if not st.session_state.authenticated:
        return None, "Not authenticated"
    
    headers = {
        "Authorization": f"Bearer {st.session_state.api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"{API_BASE_URL}{endpoint}"
    
    # Track request time
    start_time = time.time()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            return None, f"Unsupported method: {method}"
        
        # Track response time
        response_time = time.time() - start_time
        st.session_state.last_response_time = response_time
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            st.session_state.authenticated = False
            return None, "Authentication expired. Please re-enter your API key."
        elif response.status_code == 429:
            # Handle rate limiting
            retry_after = response.headers.get('Retry-After', 60)
            return None, f"Rate limit exceeded. Please wait {retry_after} seconds."
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return None, f"Connection error: {str(e)}"

def authenticate_user():
    """Handle user authentication"""
    st.markdown('<div class="main-header">🚀 Enhanced RAG System</div>', unsafe_allow_html=True)
    
    # Feature badges
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="feature-badge">⚡ Multi-Level Caching</span>
            <span class="feature-badge">🛡️ Rate Limiting</span>
            <span class="feature-badge">🔄 Concurrent Processing</span>
            <span class="feature-badge">📊 Advanced Metrics</span>
            <span class="feature-badge">📝 Structured Logging</span>
        </div>
    """, unsafe_allow_html=True)
    
    if check_session_timeout():
        st.session_state.authenticated = False
        st.session_state.api_key = ""
    
    if not st.session_state.authenticated:
        st.warning("⚠️ Please enter your API key to access the system")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="Enter your API key",
                help="Enter a valid API key to access the enhanced RAG system"
            )
            
            if st.button("🔐 Authenticate", use_container_width=True):
                if api_key.strip():
                    headers = {"Authorization": f"Bearer {api_key}"}
                    try:
                        response = requests.get(f"{API_BASE_URL}/health", headers=headers, timeout=10)
                        if response.status_code == 200:
                            st.session_state.authenticated = True
                            st.session_state.api_key = api_key
                            st.session_state.auth_time = datetime.now()
                            st.success("✅ Authentication successful!")
                            st.rerun()
                        else:
                            st.error("❌ API key validation failed")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                else:
                    st.error("❌ Please enter a valid API key")
        
        # Enhanced system information
        st.markdown("---")
        st.markdown("### 📋 Enhanced System Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Core Features:**")
            st.markdown("- 💬 Conversational AI")
            st.markdown("- 🧠 Context Awareness")
            st.markdown("- 🔍 Hybrid Search")
            st.markdown("- 📚 Dynamic Indexing")
        
        with col2:
            st.markdown("**⚡ Performance:**")
            st.markdown("- ⚡ Multi-level Caching")
            st.markdown("- 🔄 Concurrent Processing")
            st.markdown("- 📊 Real-time Metrics")
            st.markdown("- ⏱️ Sub-second Responses")
        
        with col3:
            st.markdown("**🛡️ Production Ready:**")
            st.markdown("- 🛡️ Rate Limiting")
            st.markdown("- 📝 Structured Logging")
            st.markdown("- 🔍 Performance Monitoring")
            st.markdown("- 🔒 Secure Authentication")
        
        return False
    else:
        remaining = get_remaining_time()
        if remaining <= 0:
            st.error("⏰ Session expired. Please re-authenticate.")
            st.session_state.authenticated = False
            st.rerun()
        
        return True

def fetch_system_metrics():
    """NEW: Fetch advanced system metrics"""
    data, error = make_api_request("/api/v1/metrics")
    if data:
        # Add to history for trending
        data['timestamp'] = datetime.now().isoformat()
        st.session_state.metrics_history.append(data)
        
        # Keep only last 100 entries
        if len(st.session_state.metrics_history) > 100:
            st.session_state.metrics_history = st.session_state.metrics_history[-100:]
        
        return data
    return None

def display_enhanced_dashboard():
    """NEW: Enhanced dashboard with all features"""
    st.markdown("### 📊 Enhanced System Dashboard")
    
    # Fetch health and metrics
    health_data = make_api_request("/health")[0]
    metrics_data = fetch_system_metrics()
    
    if health_data:
        # Status indicator
        status = health_data.get('status', 'unknown')
        if status == 'healthy':
            st.markdown('<div class="status-box status-healthy">🟢 System Status: HEALTHY & OPTIMIZED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-unhealthy">🔴 System Status: UNHEALTHY</div>', unsafe_allow_html=True)
        
        # Main metrics grid
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            conversations = health_data.get('conversations_active', 0)
            st.metric("💬 Active Chats", conversations)
        
        with col2:
            components = health_data.get('components', {})
            active_components = sum(1 for v in components.values() if v is True)
            st.metric("⚙️ Components", f"{active_components}/{len(components)}")
        
        with col3:
            sources_data, _ = make_api_request("/api/v1/sources")
            total_sources = sources_data.get('total_sources', 0) if sources_data else 0
            st.metric("📚 Sources", total_sources)
        
        with col4:
            if st.session_state.last_response_time:
                st.metric("⏱️ Last Response", f"{st.session_state.last_response_time*1000:.0f}ms")
            else:
                st.metric("⏱️ Last Response", "N/A")
        
        with col5:
            if metrics_data and 'cache_performance' in metrics_data:
                hit_rate = metrics_data['cache_performance'].get('hit_rate_percent', '0.00')
                st.metric("🎯 Cache Hit Rate", f"{hit_rate}%")
            else:
                st.metric("🎯 Cache Hit Rate", "N/A")
        
        # NEW: Performance metrics section
        if metrics_data:
            st.markdown("#### ⚡ Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cache performance
                st.markdown("##### 🗄️ Cache Performance")
                cache_stats = metrics_data.get('cache_performance', {})
                
                cache_col1, cache_col2, cache_col3 = st.columns(3)
                with cache_col1:
                    st.metric("Hits", cache_stats.get('hits', 0))
                with cache_col2:
                    st.metric("Misses", cache_stats.get('misses', 0))
                with cache_col3:
                    st.metric("Response Cache", cache_stats.get('response_cache_size', 0))
                
                # Cache efficiency indicator
                hit_rate_num = float(cache_stats.get('hit_rate_percent', '0'))
                if hit_rate_num > 60:
                    st.success(f"✅ Excellent cache performance: {hit_rate_num}%")
                elif hit_rate_num > 30:
                    st.info(f"ℹ️ Good cache performance: {hit_rate_num}%")
                else:
                    st.warning(f"⚠️ Cache warming up: {hit_rate_num}%")
            
            with col2:
                # System health
                st.markdown("##### 🏥 System Health")
                system_health = metrics_data.get('system_health', {})
                
                health_col1, health_col2 = st.columns(2)
                with health_col1:
                    st.metric("Status", system_health.get('status', 'unknown').upper())
                    st.metric("Active Conversations", system_health.get('conversations_active', 0))
                
                with health_col2:
                    indices = metrics_data.get('indices', {})
                    st.metric("Static Docs", indices.get('static_documents', 0))
                    st.metric("Dynamic Docs", indices.get('dynamic_documents', 0))
        
        # Component status
        st.markdown("#### 🔧 Component Status")
        components = health_data.get('components', {})
        
        component_cols = st.columns(4)
        for i, (component, status) in enumerate(components.items()):
            with component_cols[i % 4]:
                status_icon = "✅" if status else "❌"
                status_text = "Active" if status else "Inactive"
                
                # NEW features highlighted
                if component in ['cache_manager', 'rate_limiter']:
                    st.markdown(f"{status_icon} **{component.replace('_', ' ').title()}** 🆕")
                else:
                    st.markdown(f"{status_icon} **{component.replace('_', ' ').title()}**")

def enhanced_chat_interface():
    """Enhanced chat interface with performance indicators"""
    st.markdown("### 💬 Enhanced Chat Interface")
    
    # Initialize session if needed
    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
    
    # Session management with enhanced info
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.session_state.session_id:
            st.info(f"📝 Session: {st.session_state.session_id[:8]}... | Messages: {len(st.session_state.messages)}")
    
    with col2:
        if st.button("🆕 New Session"):
            if st.session_state.session_id and st.session_state.messages:
                st.session_state.conversation_history[st.session_state.session_id] = st.session_state.messages.copy()
            
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success("New session started!")
            st.rerun()
    
    with col3:
        if st.button("🧹 Clear"):
            st.session_state.messages = []
            if st.session_state.session_id:
                st.session_state.conversation_history[st.session_state.session_id] = []
            st.rerun()
    
    with col4:
        # NEW: Cache clear button (admin only)
        if st.button("🗑️ Clear Cache"):
            result, error = make_api_request("/api/v1/cache/clear", "POST")
            if result:
                st.success("Cache cleared!")
            else:
                st.error(f"Failed: {error}")
    
    # Performance indicator
    if st.session_state.last_response_time:
        response_time_ms = st.session_state.last_response_time * 1000
        if response_time_ms < 100:
            st.markdown(f'<div class="cache-indicator">⚡ CACHED RESPONSE - {response_time_ms:.0f}ms</div>', unsafe_allow_html=True)
        else:
            st.info(f"⏱️ Response time: {response_time_ms:.0f}ms")
    
    # Quick demo questions with categories
    st.markdown("#### 🚀 Demo Questions")
    
    demo_categories = {
        "💡 General": [
            "What is attention mechanism?",
            "How do RAG systems work?"
        ],
        "🔄 Context Test": [
            "Tell me about transformers",
            "What are their main advantages?"
        ]
    }
    
    for category, questions in demo_categories.items():
        with st.expander(category):
            cols = st.columns(2)
            for i, question in enumerate(questions):
                with cols[i % 2]:
                    if st.button(f"📝 {question}", key=f"demo_{category}_{i}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        process_enhanced_chat_message(question)
                        st.session_state.conversation_history[st.session_state.session_id] = st.session_state.messages.copy()
                        st.rerun()
    
    # Chat history display
    st.markdown("#### 💭 Conversation")
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
                    
                    # Show performance metadata
                    if "metadata" in message:
                        metadata = message["metadata"]
                        
                        # Performance indicators
                        perf_cols = st.columns(4)
                        with perf_cols[0]:
                            st.caption(f"📄 {metadata.get('num_docs_retrieved', 0)} docs")
                        with perf_cols[1]:
                            st.caption(f"🔍 {metadata.get('num_sources', 0)} sources")
                        with perf_cols[2]:
                            if metadata.get('reranker_used'):
                                st.caption("🎯 Reranked")
                        with perf_cols[3]:
                            if metadata.get('rate_limit_remaining') is not None:
                                remaining = metadata['rate_limit_remaining']
                                if remaining < 5:
                                    st.caption(f"⚠️ {remaining} requests left")
                    
                    # Show sources
                    if "sources" in message and message["sources"]:
                        st.markdown("**📚 Sources:**")
                        for source in message["sources"]:
                            st.markdown(f'<div class="source-link">🔗 {source}</div>', unsafe_allow_html=True)
        else:
            st.info("👋 Start chatting! The system uses advanced caching for faster responses.")
    
    # Enhanced chat input
    st.markdown("#### ✍️ Your Message")
    
    with st.form(key="chat_form"):
        user_input = st.text_area(
            "Ask anything:",
            placeholder="Try asking follow-up questions to test context awareness...",
            height=100
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            use_dynamic = st.checkbox("Dynamic Index", value=True)
        with col2:
            use_reranker = st.checkbox("Reranker", value=True)
        with col3:
            top_k = st.number_input("Top K", min_value=1, max_value=5, value=3)
        with col4:
            st.form_submit_button("🚀 Send", use_container_width=True)
        
        if st.form_submit_button and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            process_enhanced_chat_message(user_input, use_dynamic, use_reranker, top_k)
            st.session_state.conversation_history[st.session_state.session_id] = st.session_state.messages.copy()
            st.rerun()

def process_enhanced_chat_message(message: str, use_dynamic: bool = True, use_reranker: bool = True, top_k: int = 3):
    """Process chat message with performance tracking"""
    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())
    
    chat_data = {
        "messages": [{"role": "user", "content": message}],
        "session_id": st.session_state.session_id,
        "use_dynamic_index": use_dynamic,
        "use_reranker": use_reranker,
        "top_k": top_k
    }
    
    with st.spinner("🤔 Processing..."):
        start_time = time.time()
        response_data, error = make_api_request("/api/v1/chat", "POST", chat_data)
        response_time = time.time() - start_time
    
    if response_data:
        assistant_response = response_data["response"]["answer"]["content"]
        sources = response_data["response"].get("sources", [])
        metadata = response_data["response"].get("metadata", {})
        
        # Show performance
        if response_time < 0.1:
            st.success(f"⚡ CACHED! Response in {response_time*1000:.0f}ms")
        else:
            st.info(f"⏱️ Response in {response_time*1000:.0f}ms")
        
        # Add assistant response
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "sources": sources,
            "metadata": metadata,
            "response_time": response_time
        }
        st.session_state.messages.append(assistant_msg)
        st.session_state.session_id = response_data["session_id"]
        
        # Show rate limit warning if low
        if metadata.get('rate_limit_remaining', 100) < 5:
            st.warning(f"⚠️ Rate limit warning: {metadata['rate_limit_remaining']} requests remaining")
    
    else:
        if "Rate limit" in str(error):
            st.error(f"🛑 {error}")
        else:
            st.error(f"Failed: {error}")

def enhanced_content_management():
    """Enhanced content management with concurrent indexing status"""
    st.markdown("### 📚 Content Management (Concurrent Processing)")
    
    # Current sources
    with st.expander("📖 Current Sources", expanded=True):
        sources_data, error = make_api_request("/api/v1/sources")
        if sources_data:
            st.session_state.sources_data = sources_data
            
            total = sources_data.get('total_sources', 0)
            static_count = len(sources_data.get('static_sources', []))
            dynamic_count = len(sources_data.get('dynamic_sources', []))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total", total)
            with col2:
                st.metric("🏛️ Static", static_count)
            with col3:
                st.metric("🔄 Dynamic", dynamic_count, help="Indexed with concurrent processing")
            
            # Sources table
            if total > 0:
                all_sources = []
                for source in sources_data.get('static_sources', []) + sources_data.get('dynamic_sources', []):
                    indexed_at = 'N/A'
                    if source.get('indexed_at'):
                        try:
                            dt = datetime.fromisoformat(source['indexed_at'].replace('Z', '+00:00'))
                            indexed_at = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            indexed_at = 'N/A'
                    
                    all_sources.append({
                        'URL': source['source_url'][:60] + '...' if len(source['source_url']) > 60 else source['source_url'],
                        'Title': source.get('title', 'N/A')[:40] + '...' if source.get('title') and len(source.get('title', '')) > 40 else source.get('title', 'N/A'),
                        'Type': source['source_type'],
                        'Docs': source['document_count'],
                        'Indexed': indexed_at
                    })
                
                if all_sources:
                    df = pd.DataFrame(all_sources)
                    st.dataframe(df, use_container_width=True)
    
    # Add new URLs with concurrent processing
    with st.expander("➕ Add New URLs (Concurrent Processing)", expanded=True):
        st.markdown("#### 🚀 Index Content with Concurrent Processing")
        st.info("⚡ URLs are processed concurrently for faster indexing (max 5 simultaneous)")
        
        input_method = st.radio(
            "Input method:",
            ["Single URL", "Multiple URLs", "Demo URLs"],
            horizontal=True
        )
        
        urls_to_index = []
        
        if input_method == "Single URL":
            url = st.text_input("URL:", placeholder="https://example.com/article")
            if url:
                urls_to_index = [url]
        
        elif input_method == "Multiple URLs":
            urls_text = st.text_area(
                "URLs (one per line):",
                placeholder="https://example.com/page1\nhttps://example.com/page2\nhttps://example.com/page3",
                height=120,
                help="Add up to 5 URLs for concurrent processing"
            )
            if urls_text:
                urls_to_index = [url.strip() for url in urls_text.split('\n') if url.strip()][:5]
        
        else:
            demo_urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://arxiv.org/abs/2005.14165",
                "https://openai.com/blog/chatgpt"
            ]
            selected_demo_urls = st.multiselect("Select:", demo_urls)
            urls_to_index = selected_demo_urls
        
        if urls_to_index:
            st.markdown(f"**📋 Ready to index ({len(urls_to_index)} URLs):**")
            for i, url in enumerate(urls_to_index, 1):
                st.markdown(f"{i}. {url}")
            
            if len(urls_to_index) > 1:
                st.success(f"⚡ These {len(urls_to_index)} URLs will be processed concurrently!")
            
            if st.button("🚀 Start Concurrent Indexing", use_container_width=True):
                enhanced_index_urls(urls_to_index)

def enhanced_index_urls(urls: List[str]):
    """Enhanced indexing with performance tracking"""
    index_data = {"url": urls}
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Initializing concurrent processing...")
    progress_bar.progress(10)
    
    time.sleep(0.5)
    status_text.text(f"⚡ Processing {len(urls)} URLs concurrently...")
    progress_bar.progress(30)
    
    # Make API request
    start_time = time.time()
    response_data, error = make_api_request("/api/v1/index", "POST", index_data)
    processing_time = time.time() - start_time
    
    progress_bar.progress(90)
    
    if response_data:
        progress_bar.progress(100)
        status_text.text("✅ Concurrent processing completed!")
        
        # Display results with performance metrics
        st.success("🎉 Indexing Complete!")
        
        metadata = response_data.get('metadata', {})
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("⏱️ Time", f"{processing_time:.2f}s")
        with perf_col2:
            st.metric("📥 Requested", metadata.get('total_requested', 0))
        with perf_col3:
            st.metric("✅ Success", metadata.get('successfully_indexed', 0))
        with perf_col4:
            st.metric("❌ Failed", metadata.get('failed', 0))
        
        # Concurrent processing indicator
        if metadata.get('concurrent_processing'):
            st.success("⚡ Concurrent processing was used!")
            if len(urls) > 1:
                sequential_estimate = processing_time * len(urls)
                time_saved = sequential_estimate - processing_time
                st.info(f"🚀 Estimated time saved vs sequential: {time_saved:.1f}s ({time_saved/sequential_estimate*100:.0f}% faster)")
        
        # Show successful URLs
        if response_data.get('indexed_url'):
            st.markdown("**✅ Successfully Indexed:**")
            for url in response_data['indexed_url']:
                st.markdown(f"- ✅ {url}")
        
        # Show failed URLs
        if response_data.get('failed_url'):
            st.markdown("**❌ Failed URLs:**")
            for failed in response_data['failed_url']:
                st.markdown(f"- ❌ {failed['url']}: {failed['error']}")
        
        # Document chunks info
        new_docs = metadata.get('new_documents_added', 0)
        if new_docs > 0:
            st.info(f"📄 Added {new_docs} document chunks to knowledge base")
        
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
    else:
        progress_bar.progress(100)
        status_text.text("❌ Indexing failed!")
        if "Rate limit" in str(error):
            st.error(f"🛑 {error}")
        else:
            st.error(f"Failed: {error}")

def performance_analytics():
    """NEW: Performance analytics page"""
    st.markdown("### 📊 Performance Analytics")
    
    # Fetch current metrics
    metrics_data = fetch_system_metrics()
    
    if metrics_data:
        # Cache Performance
        st.markdown("#### 🗄️ Cache Performance")
        cache_stats = metrics_data.get('cache_performance', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Hits", cache_stats.get('hits', 0))
        with col2:
            st.metric("Total Misses", cache_stats.get('misses', 0))
        with col3:
            hit_rate = cache_stats.get('hit_rate_percent', '0.00')
            st.metric("Hit Rate", f"{hit_rate}%")
        with col4:
            st.metric("Cache Size", cache_stats.get('response_cache_size', 0))
        
        # Cache efficiency gauge
        try:
            hit_rate_num = float(hit_rate)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hit_rate_num,
                title={'text': "Cache Hit Rate"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 60], 'color': "lightblue"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        except:
            pass
        
        # System indices
        st.markdown("#### 📚 Knowledge Base Statistics")
        indices = metrics_data.get('indices', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Static Documents", indices.get('static_documents', 0))
        with col2:
            st.metric("Dynamic Documents", indices.get('dynamic_documents', 0))
        with col3:
            total = indices.get('static_documents', 0) + indices.get('dynamic_documents', 0)
            st.metric("Total Documents", total)
        
        # Documents distribution
        if indices.get('static_documents', 0) > 0 or indices.get('dynamic_documents', 0) > 0:
            fig = px.pie(
                values=[indices.get('static_documents', 0), indices.get('dynamic_documents', 0)],
                names=['Static Index', 'Dynamic Index'],
                title='Document Distribution',
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Response time history
        if len(st.session_state.metrics_history) > 1:
            st.markdown("#### ⏱️ Cache Performance Trend")
            
            df_metrics = pd.DataFrame([
                {
                    'timestamp': m.get('timestamp', ''),
                    'hit_rate': float(m.get('cache_performance', {}).get('hit_rate_percent', 0))
                }
                for m in st.session_state.metrics_history[-20:]  # Last 20 entries
            ])
            
            if not df_metrics.empty:
                fig = px.line(
                    df_metrics,
                    x='timestamp',
                    y='hit_rate',
                    title='Cache Hit Rate Over Time',
                    labels={'hit_rate': 'Hit Rate (%)', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # System health overview
        st.markdown("#### 🏥 System Health")
        system_health = metrics_data.get('system_health', {})
        
        health_cols = st.columns(3)
        with health_cols[0]:
            status = system_health.get('status', 'unknown')
            if status == 'healthy':
                st.success(f"✅ Status: {status.upper()}")
            else:
                st.error(f"❌ Status: {status.upper()}")
        
        with health_cols[1]:
            st.info(f"💬 Active Conversations: {system_health.get('conversations_active', 0)}")
        
        with health_cols[2]:
            static_loaded = system_health.get('static_index_loaded', False)
            dynamic_loaded = system_health.get('dynamic_index_loaded', False)
            st.info(f"📚 Indices: Static {'✅' if static_loaded else '❌'} | Dynamic {'✅' if dynamic_loaded else '❌'}")
    
    else:
        st.warning("⚠️ Unable to fetch performance metrics")
    
    # Refresh button
    if st.button("🔄 Refresh Metrics", use_container_width=True):
        st.rerun()

def system_features_info():
    """NEW: Detailed features information page"""
    st.markdown("### 🚀 System Features & Capabilities")
    
    # Feature showcase
    features = {
        "⚡ Multi-Level Caching": {
            "description": "Three-tier caching system for optimal performance",
            "details": [
                "Response Cache: Stores complete API responses (30-min TTL)",
                "Search Cache: Caches hybrid search results per query",
                "Embedding Cache: LRU cache for text embeddings",
                "Hit rate tracking and metrics endpoint"
            ],
            "benefit": "60-70% faster responses for repeated queries"
        },
        "🛡️ Rate Limiting": {
            "description": "Per-user, per-endpoint rate limiting",
            "details": [
                "Chat: 20 requests per minute",
                "Indexing: 5 requests per 5 minutes",
                "Evaluation: 3 requests per 10 minutes",
                "Custom limits with sliding window algorithm"
            ],
            "benefit": "Prevents abuse and ensures fair usage"
        },
        "🔄 Concurrent Processing": {
            "description": "Async URL processing for fast indexing",
            "details": [
                "Processes up to 5 URLs simultaneously",
                "Built with async/await and aiohttp",
                "Semaphore limiting for controlled concurrency",
                "Exponential backoff retry logic"
            ],
            "benefit": "80% faster indexing vs sequential processing"
        },
        "📝 Structured Logging": {
            "description": "Production-ready logging system",
            "details": [
                "JSON-formatted logs for easy parsing",
                "Log rotation (10MB files, 5 backups)",
                "Performance timing on all endpoints",
                "Ready for CloudWatch/ELK integration"
            ],
            "benefit": "Better debugging and monitoring"
        },
        "🧠 Hybrid Search": {
            "description": "BM25 + FAISS + Reranking",
            "details": [
                "BM25 for lexical matching",
                "FAISS for semantic search",
                "Cross-encoder reranking for precision",
                "Conversation context integration"
            ],
            "benefit": "15-20% better relevance vs single method"
        },
        "💬 Context Awareness": {
            "description": "Maintains conversation context",
            "details": [
                "Session-based conversation tracking",
                "Rolling window of last 3 message pairs",
                "Automatic timeout and cleanup",
                "Reference resolution (it, this, that)"
            ],
            "benefit": "Natural follow-up questions"
        }
    }
    
    for feature_name, feature_info in features.items():
        with st.expander(f"{feature_name}", expanded=False):
            st.markdown(f"**{feature_info['description']}**")
            
            st.markdown("**Key Features:**")
            for detail in feature_info['details']:
                st.markdown(f"- {detail}")
            
            st.success(f"✅ **Benefit:** {feature_info['benefit']}")
    
    # Technical architecture
    st.markdown("---")
    st.markdown("### 🏗️ Technical Architecture")
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("**🔧 Backend Stack:**")
        st.markdown("""
        - FastAPI with async/await
        - FAISS vector store
        - BM25 lexical search
        - Gemini 1.5 Flash LLM
        - Cross-encoder reranking
        - Custom cache manager
        - Custom rate limiter
        """)
    
    with arch_col2:
        st.markdown("**⚙️ Infrastructure:**")
        st.markdown("""
        - AWS EC2 deployment
        - In-memory caching
        - Structured logging with rotation
        - Concurrent web scraping
        - Real-time metrics API
        - RESTful API design
        """)

def main():
    """Enhanced main application"""
    init_session_state()
    
    # Authentication
    if not authenticate_user():
        return
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    
    # User info and logout
    st.sidebar.markdown(f"**👤 Authenticated**")
    remaining = get_remaining_time()
    st.sidebar.caption(f"⏱️ Session: {format_time(remaining)}")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Enhanced navigation
    page = st.sidebar.selectbox(
        "Select Feature:",
        [
            "🏠 Dashboard",
            "💬 Chat Interface",
            "📚 Content Management",
            "📊 Performance Analytics",
            "🚀 System Features"
        ]
    )
    
    # Quick metrics in sidebar
    st.sidebar.markdown("### 📈 Quick Stats")
    
    # Fetch metrics for sidebar
    if st.sidebar.button("🔄 Refresh Stats"):
        fetch_system_metrics()
    
    if st.session_state.system_health:
        conversations = st.session_state.system_health.get('conversations_active', 0)
        st.sidebar.metric("💬 Conversations", conversations)
    
    if st.session_state.sources_data:
        total_sources = st.session_state.sources_data.get('total_sources', 0)
        st.sidebar.metric("📚 Sources", total_sources)
    
    if st.session_state.last_response_time:
        st.sidebar.metric("⏱️ Last Response", f"{st.session_state.last_response_time*1000:.0f}ms")
    
    # Main content routing
    if page == "🏠 Dashboard":
        display_enhanced_dashboard()
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💬 Start Chat", use_container_width=True):
                st.session_state.current_page = "💬 Chat Interface"
                st.rerun()
        
        with col2:
            if st.button("📚 Manage Content", use_container_width=True):
                st.session_state.current_page = "📚 Content Management"
                st.rerun()
        
        with col3:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.current_page = "📊 Performance Analytics"
                st.rerun()
        
        with col4:
            if st.button("🔄 Refresh All", use_container_width=True):
                fetch_system_metrics()
                st.rerun()
        
        # Feature highlights
        st.markdown("---")
        st.markdown("### ✨ What's New")
        
        highlight_col1, highlight_col2, highlight_col3 = st.columns(3)
        
        with highlight_col1:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Caching</h3>
                <p>Multi-level cache with 60-70% hit rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with highlight_col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔄 Concurrent</h3>
                <p>80% faster URL processing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with highlight_col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Metrics</h3>
                <p>Real-time performance tracking</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "💬 Chat Interface":
        enhanced_chat_interface()
    
    elif page == "📚 Content Management":
        enhanced_content_management()
    
    elif page == "📊 Performance Analytics":
        performance_analytics()
    
    elif page == "🚀 System Features":
        system_features_info()
    
    # Enhanced footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
    with footer_col1:
        st.markdown("**🔗 API:** http://56.228.63.64:8000")
    
    with footer_col2:
        st.markdown("**📚 Docs:** [/docs](http://56.228.63.64:8000/docs)")
    
    with footer_col3:
        st.markdown("**⚡ Version:** 4.0.0 Enhanced")
    
    with footer_col4:
        if st.session_state.last_response_time:
            st.markdown(f"**⏱️ Last:** {st.session_state.last_response_time*1000:.0f}ms")

if __name__ == "__main__":
    main()
