import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd
import uuid

# CRITICAL: Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .chat-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .source-tag {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 5px;
        padding: 0.3rem 0.6rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://56.228.63.64:8000"
SESSION_TIMEOUT = 1800  # 30 minutes

# Initialize session state - ONLY ONCE
def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.auth_time = None
        st.session_state.chat_session_id = None
        st.session_state.chat_messages = []
        st.session_state.last_user_message = ""  # Track last message to prevent duplicates

# Call initialization
init_session_state()

def check_session_timeout():
    """Check if session expired"""
    if st.session_state.auth_time:
        elapsed = (datetime.now() - st.session_state.auth_time).total_seconds()
        return elapsed > SESSION_TIMEOUT
    return True

def make_api_request(endpoint: str, method: str = "GET", data: dict = None):
    """Make API request with proper error handling"""
    if not st.session_state.authenticated:
        return None, "Not authenticated"
    
    headers = {
        "Authorization": f"Bearer {st.session_state.api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        else:
            return None, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            st.session_state.authenticated = False
            return None, "Authentication expired"
        elif response.status_code == 429:
            return None, "Rate limit exceeded. Please wait."
        else:
            return None, f"Error {response.status_code}: {response.text[:100]}"
    
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to server. Please check if backend is running."
    except Exception as e:
        return None, f"Error: {str(e)[:100]}"

def authenticate_user():
    """Handle authentication - NO EXPOSED API KEY"""
    st.markdown('<div class="main-header">🚀 Enhanced RAG System</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                         font-size: 0.75rem; font-weight: bold; margin: 0.2rem;">⚡ Caching</span>
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                         font-size: 0.75rem; font-weight: bold; margin: 0.2rem;">🛡️ Rate Limiting</span>
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                         font-size: 0.75rem; font-weight: bold; margin: 0.2rem;">🔄 Concurrent</span>
            <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 0.3rem 0.8rem; border-radius: 15px; 
                         font-size: 0.75rem; font-weight: bold; margin: 0.2rem;">📊 Metrics</span>
        </div>
    """, unsafe_allow_html=True)
    
    if check_session_timeout():
        st.session_state.authenticated = False
        st.session_state.api_key = ""
    
    if not st.session_state.authenticated:
        st.warning("⚠️ Please enter your API key to access the system")
        
        # FIXED: No exposed API key - user must provide it
        with st.form("auth_form"):
            api_key = st.text_input(
                "API Key", 
                type="password", 
                placeholder="Enter your API key",
                help="Contact administrator for API key access"
            )
            submit = st.form_submit_button("🔐 Login", use_container_width=True)
            
            if submit and api_key.strip():
                with st.spinner("Authenticating..."):
                    headers = {"Authorization": f"Bearer {api_key}"}
                    try:
                        response = requests.get(f"{API_BASE_URL}/health", headers=headers, timeout=10)
                        if response.status_code == 200:
                            st.session_state.authenticated = True
                            st.session_state.api_key = api_key
                            st.session_state.auth_time = datetime.now()
                            st.success("✅ Authenticated!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Invalid API key")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
        
        # System info without API key
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✨ System Features:**")
            st.markdown("- 💬 Conversational AI")
            st.markdown("- 🧠 Context-aware responses")
            st.markdown("- 🔍 Hybrid search engine")
            st.markdown("- 📚 Dynamic indexing")
        
        with col2:
            st.markdown("**⚡ Performance:**")
            st.markdown("- Multi-level caching")
            st.markdown("- Rate limiting protection")
            st.markdown("- Concurrent processing")
            st.markdown("- Real-time metrics")
        
        return False
    
    return True

def show_dashboard():
    """Show system dashboard"""
    st.markdown("### 📊 System Dashboard")
    
    # Fetch health
    health_data, error = make_api_request("/health")
    
    if health_data:
        status = health_data.get('status', 'unknown')
        if status == 'healthy':
            st.markdown('<div class="status-healthy">🟢 System Healthy</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            conversations = health_data.get('conversations_active', 0)
            st.metric("💬 Conversations", conversations)
        
        with col2:
            components = health_data.get('components', {})
            active = sum(1 for v in components.values() if v)
            st.metric("⚙️ Components", f"{active}/{len(components)}")
        
        with col3:
            # Get sources
            sources_data, _ = make_api_request("/api/v1/sources")
            total = sources_data.get('total_sources', 0) if sources_data else 0
            st.metric("📚 Sources", total)
        
        with col4:
            # Get metrics
            metrics_data, _ = make_api_request("/api/v1/metrics")
            if metrics_data and 'cache_performance' in metrics_data:
                hit_rate = metrics_data['cache_performance'].get('hit_rate_percent', '0')
                st.metric("🎯 Cache Hit", f"{hit_rate}%")
            else:
                st.metric("🎯 Cache Hit", "N/A")
        
        # Quick info
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✨ Features:**")
            st.markdown("- ⚡ Multi-level caching")
            st.markdown("- 🛡️ Rate limiting")
            st.markdown("- 🔄 Concurrent indexing")
            st.markdown("- 📊 Real-time metrics")
        
        with col2:
            st.markdown("**🎯 Performance:**")
            st.markdown("- Cache hits: <20ms")
            st.markdown("- Cache misses: ~700ms")
            st.markdown("- 80% faster indexing")
            st.markdown("- 60-70% hit rate")
    
    else:
        st.error(f"Cannot fetch system health: {error}")

def show_chat():
    """Show chat interface - FIXED response display issue"""
    st.markdown("### 💬 Chat Interface")
    
    # Initialize chat session
    if not st.session_state.chat_session_id:
        st.session_state.chat_session_id = str(uuid.uuid4())
    
    # Session controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.info(f"📝 Session: {st.session_state.chat_session_id[:8]}... | Messages: {len(st.session_state.chat_messages)}")
    
    with col2:
        if st.button("🆕 New Session"):
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.session_state.chat_messages = []
            st.session_state.last_user_message = ""
            st.rerun()
    
    with col3:
        if st.button("🧹 Clear"):
            st.session_state.chat_messages = []
            st.session_state.last_user_message = ""
            st.rerun()
    
    # Demo questions
    st.markdown("#### 🚀 Quick Start")
    demo_cols = st.columns(2)
    
    demo_questions = [
        "What is attention mechanism?",
        "How do RAG systems work?",
        "Tell me about transformers",
        "What are the challenges in AI?"
    ]
    
    for i, question in enumerate(demo_questions):
        with demo_cols[i % 2]:
            # FIXED: Check if this message was already processed
            if st.button(f"💡 {question}", key=f"demo_q_{i}"):
                if st.session_state.last_user_message != question:
                    st.session_state.last_user_message = question
                    send_message_and_display(question)
    
    # Display chat history
    st.markdown("#### 💭 Conversation")
    
    if st.session_state.chat_messages:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 **You:** {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant">🤖 **Assistant:** {msg["content"]}</div>', unsafe_allow_html=True)
                
                # Show sources
                if msg.get("sources"):
                    st.markdown("**📚 Sources:**")
                    for source in msg["sources"]:
                        st.markdown(f'<span class="source-tag">🔗 {source[:50]}...</span>', unsafe_allow_html=True)
                
                # Show metadata
                if msg.get("metadata"):
                    meta = msg["metadata"]
                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.caption(f"📄 {meta.get('num_docs_retrieved', 0)} docs")
                    with meta_cols[1]:
                        st.caption(f"🔍 {meta.get('num_sources', 0)} sources")
                    with meta_cols[2]:
                        if meta.get('rate_limit_remaining') is not None:
                            remaining = meta['rate_limit_remaining']
                            if remaining < 5:
                                st.caption(f"⚠️ {remaining} left")
    else:
        st.info("👋 Start by asking a question or using quick start buttons above!")
    
    # Chat input - FIXED to display response immediately
    st.markdown("#### ✍️ Your Message")
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask anything:",
            placeholder="Type your question here...",
            height=100,
            key="user_input_field"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            use_dynamic = st.checkbox("Dynamic Index", value=True)
        with col2:
            use_reranker = st.checkbox("Reranker", value=True)
        
        # Submit button
        submitted = st.form_submit_button("🚀 Send", use_container_width=True)
        
        if submitted and user_input.strip():
            # FIXED: Check if this is a duplicate message
            if st.session_state.last_user_message != user_input:
                st.session_state.last_user_message = user_input
                send_message_and_display(user_input, use_dynamic, use_reranker)

def send_message_and_display(message: str, use_dynamic: bool = True, use_reranker: bool = True):
    """Send message and display response immediately - FIXED"""
    
    # Add user message to display
    st.session_state.chat_messages.append({
        "role": "user",
        "content": message
    })
    
    # Prepare request
    chat_data = {
        "messages": [{"role": "user", "content": message}],
        "session_id": st.session_state.chat_session_id,
        "use_dynamic_index": use_dynamic,
        "use_reranker": use_reranker,
        "top_k": 3
    }
    
    # Create placeholder for response
    response_placeholder = st.empty()
    
    # Show loading
    with response_placeholder:
        with st.spinner("🤔 Thinking..."):
            response_data, error = make_api_request("/api/v1/chat", "POST", chat_data)
    
    # Clear loading and show result
    response_placeholder.empty()
    
    if response_data:
        assistant_response = response_data["response"]["answer"]["content"]
        sources = response_data["response"].get("sources", [])
        metadata = response_data["response"].get("metadata", {})
        
        # Add assistant response to messages
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": assistant_response,
            "sources": sources,
            "metadata": metadata
        })
        
        # Update session ID
        st.session_state.chat_session_id = response_data.get("session_id", st.session_state.chat_session_id)
        
        # CRITICAL FIX: Force immediate rerun to display the message
        st.rerun()
    else:
        # Show error
        st.error(f"❌ Failed: {error}")
        # Remove the user message if failed
        if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":
            st.session_state.chat_messages.pop()
        st.rerun()

def show_content_management():
    """Show content management"""
    st.markdown("### 📚 Content Management")
    
    # Current sources
    with st.expander("📖 Current Sources", expanded=True):
        sources_data, error = make_api_request("/api/v1/sources")
        
        if sources_data:
            total = sources_data.get('total_sources', 0)
            static_count = len(sources_data.get('static_sources', []))
            dynamic_count = len(sources_data.get('dynamic_sources', []))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total", total)
            with col2:
                st.metric("🏛️ Static", static_count)
            with col3:
                st.metric("🔄 Dynamic", dynamic_count)
            
            # Show sources table
            if total > 0:
                all_sources = []
                for source in sources_data.get('static_sources', []) + sources_data.get('dynamic_sources', []):
                    all_sources.append({
                        'URL': source['source_url'][:60] + '...' if len(source['source_url']) > 60 else source['source_url'],
                        'Type': source['source_type'],
                        'Docs': source['document_count']
                    })
                
                if all_sources:
                    df = pd.DataFrame(all_sources)
                    st.dataframe(df, use_container_width=True)
        else:
            st.error(f"Cannot fetch sources: {error}")
    
    # Add new URLs
    with st.expander("➕ Add URLs (Concurrent)", expanded=False):
        st.info("⚡ URLs processed concurrently for faster indexing")
        
        with st.form("index_form"):
            urls_text = st.text_area(
                "URLs (one per line, max 5):",
                placeholder="https://example.com/page1\nhttps://example.com/page2",
                height=120
            )
            
            submit = st.form_submit_button("🚀 Start Indexing", use_container_width=True)
            
            if submit and urls_text.strip():
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()][:5]
                
                if urls:
                    st.markdown(f"**Indexing {len(urls)} URLs...**")
                    
                    index_data = {"url": urls}
                    
                    progress = st.progress(0)
                    progress.progress(30)
                    
                    start_time = time.time()
                    response_data, error = make_api_request("/api/v1/index", "POST", index_data)
                    elapsed = time.time() - start_time
                    
                    progress.progress(100)
                    
                    if response_data:
                        st.success(f"✅ Completed in {elapsed:.1f}s")
                        
                        metadata = response_data.get('metadata', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("✅ Success", metadata.get('successfully_indexed', 0))
                        with col2:
                            st.metric("❌ Failed", metadata.get('failed', 0))
                        with col3:
                            st.metric("📄 Chunks", metadata.get('new_documents_added', 0))
                        
                        if response_data.get('indexed_url'):
                            st.markdown("**✅ Indexed:**")
                            for url in response_data['indexed_url']:
                                st.markdown(f"- {url}")
                    else:
                        st.error(f"Indexing failed: {error}")
                    
                    progress.empty()

def show_metrics():
    """Show performance metrics"""
    st.markdown("### 📊 Performance Metrics")
    
    metrics_data, error = make_api_request("/api/v1/metrics")
    
    if metrics_data:
        # Cache performance
        st.markdown("#### 🗄️ Cache Performance")
        cache = metrics_data.get('cache_performance', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Hits", cache.get('hits', 0))
        with col2:
            st.metric("Misses", cache.get('misses', 0))
        with col3:
            st.metric("Hit Rate", f"{cache.get('hit_rate_percent', '0')}%")
        with col4:
            st.metric("Cache Size", cache.get('response_cache_size', 0))
        
        # System health
        st.markdown("#### 🏥 System Health")
        health = metrics_data.get('system_health', {})
        indices = metrics_data.get('indices', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", health.get('status', 'unknown').upper())
        with col2:
            st.metric("Active Conversations", health.get('conversations_active', 0))
        with col3:
            total_docs = indices.get('static_documents', 0) + indices.get('dynamic_documents', 0)
            st.metric("Total Documents", total_docs)
        
        # Documents breakdown
        st.markdown("#### 📚 Knowledge Base")
        doc_col1, doc_col2 = st.columns(2)
        with doc_col1:
            st.metric("Static Documents", indices.get('static_documents', 0))
        with doc_col2:
            st.metric("Dynamic Documents", indices.get('dynamic_documents', 0))
    
    else:
        st.error(f"Cannot fetch metrics: {error}")
    
    if st.button("🔄 Refresh Metrics"):
        st.rerun()

def main():
    """Main application"""
    
    # Check authentication
    if not authenticate_user():
        return
    
    # Sidebar
    st.sidebar.title("🧭 Navigation")
    st.sidebar.caption(f"👤 Authenticated")
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = None
        st.session_state.last_user_message = ""
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Select:",
        ["🏠 Dashboard", "💬 Chat", "📚 Content", "📊 Metrics"]
    )
    
    # Route to pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "💬 Chat":
        show_chat()
    elif page == "📚 Content":
        show_content_management()
    elif page == "📊 Metrics":
        show_metrics()
    
    # Footer
    st.markdown("---")
    st.caption("🚀 Enhanced RAG System v4.0 | Secure & Optimized")

if __name__ == "__main__":
    main()
