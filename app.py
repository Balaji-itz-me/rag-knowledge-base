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
    page_title="RAG System Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
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
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        max-width: 80%;
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
    
    .timer-box {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://56.228.63.64:8000"
SESSION_TIMEOUT = 1800  # 30 minutes in seconds
DEMO_API_KEYS = ["demo-api-key-123", "eval-key-456"]

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
        st.session_state.conversation_history = {}  # Store conversations by session_id
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🏠 Dashboard"

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
    """Make API request with authentication"""
    if not st.session_state.authenticated:
        return None, "Not authenticated"
    
    headers = {
        "Authorization": f"Bearer {st.session_state.api_key}",
        "Content-Type": "application/json"
    }
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            return None, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 401:
            st.session_state.authenticated = False
            return None, "Authentication expired. Please re-enter your API key."
        else:
            return None, f"API Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.RequestException as e:
        return None, f"Connection error: {str(e)}"

def authenticate_user():
    """Handle user authentication"""
    st.markdown('<div class="main-header">🤖 Conversational RAG System Demo</div>', unsafe_allow_html=True)
    
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
                help="Please enter a valid API key to access the system"
            )
            
            if st.button("🔐 Authenticate", use_container_width=True):
                if api_key.strip():
                    # Test any API key against the server
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
        
        # System information (no API keys shown)
        st.markdown("---")
        st.markdown("### 📋 System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🤖 RAG System Features:**")
            st.markdown("- 💬 Conversational AI Chat")
            st.markdown("- 🧠 Context-Aware Responses")
            st.markdown("- 🔍 Hybrid Search (BM25 + FAISS)")
        
        with col2:
            st.markdown("**⚡ Advanced Capabilities:**")
            st.markdown("- 📚 Dynamic URL Indexing")
            st.markdown("- 🔍 Source Management")
            st.markdown("- 📊 System Evaluation")
        
        return False
    else:
        # Session runs in background - no visible timer
        remaining = get_remaining_time()
        if remaining <= 0:
            st.error("⏰ Session expired. Please re-authenticate.")
            st.session_state.authenticated = False
            st.rerun()
        
        return True

def fetch_system_health():
    """Fetch and display system health"""
    data, error = make_api_request("/health")
    if data:
        st.session_state.system_health = data
        return data
    else:
        st.error(f"Failed to fetch system health: {error}")
        return None

def display_system_overview():
    """Display system overview dashboard"""
    st.markdown("### 📊 System Overview")
    
    health_data = fetch_system_health()
    if health_data:
        # Status indicator
        status = health_data.get('status', 'unknown')
        if status == 'healthy':
            st.markdown('<div class="status-box status-healthy">🟢 System Status: HEALTHY</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-unhealthy">🔴 System Status: UNHEALTHY</div>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            conversations = health_data.get('conversations_active', 0)
            st.metric("💬 Active Conversations", conversations)
        
        with col2:
            components = health_data.get('components', {})
            active_components = sum(1 for v in components.values() if v is True)
            st.metric("⚙️ Active Components", f"{active_components}/{len(components)}")
        
        with col3:
            # Fetch sources to get total count
            sources_data, _ = make_api_request("/api/v1/sources")
            total_sources = sources_data.get('total_sources', 0) if sources_data else 0
            st.metric("📚 Total Sources", total_sources)
        
        with col4:
            timestamp = health_data.get('timestamp', '')
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                st.metric("🕐 Last Check", dt.strftime("%H:%M:%S"))
        
        # Component status
        if components:
            st.markdown("#### 🔧 Component Status")
            col1, col2 = st.columns(2)
            
            with col1:
                for i, (component, status) in enumerate(list(components.items())[:len(components)//2]):
                    status_icon = "✅" if status else "❌"
                    st.markdown(f"{status_icon} **{component.replace('_', ' ').title()}**: {'Active' if status else 'Inactive'}")
            
            with col2:
                for i, (component, status) in enumerate(list(components.items())[len(components)//2:]):
                    status_icon = "✅" if status else "❌"
                    st.markdown(f"{status_icon} **{component.replace('_', ' ').title()}**: {'Active' if status else 'Inactive'}")

def chat_interface():
    """Main chat interface"""
    st.markdown("### 💬 Conversational RAG Interface")
    
    # Initialize session if needed
    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
    
    # Load conversation history for current session
    if st.session_state.session_id in st.session_state.conversation_history:
        st.session_state.messages = st.session_state.conversation_history[st.session_state.session_id]
    
    # Session management
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state.session_id:
            st.info(f"📝 Active Session: {st.session_state.session_id[:8]}...")
    with col2:
        if st.button("🆕 New Session"):
            # Save current conversation before creating new session
            if st.session_state.session_id and st.session_state.messages:
                st.session_state.conversation_history[st.session_state.session_id] = st.session_state.messages.copy()
            
            # Create new session
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success("New session started!")
            st.rerun()
    with col3:
        if st.button("🧹 Clear Chat"):
            st.session_state.messages = []
            if st.session_state.session_id:
                st.session_state.conversation_history[st.session_state.session_id] = []
            st.success("Chat cleared!")
            st.rerun()
    
    # Show conversation counter
    if len(st.session_state.conversation_history) > 1:
        st.info(f"💾 You have {len(st.session_state.conversation_history)} conversation sessions saved")
    
    # Quick demo questions
    st.markdown("#### 🚀 Quick Demo Questions")
    demo_questions = [
        "What is attention mechanism in transformers?",
        "How do RAG systems work?",
        "Tell me about the latest AI trends",
        "What are the challenges in implementing RAG systems?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(demo_questions):
        with cols[i % 2]:
            if st.button(f"💡 {question}", key=f"demo_q_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                process_chat_message(question)
                # Save to conversation history
                st.session_state.conversation_history[st.session_state.session_id] = st.session_state.messages.copy()
                st.rerun()
    
    # Chat history display
    st.markdown("#### 💭 Conversation History")
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-message">🤖 **Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
                    
                    # Show sources if available
                    if "sources" in message and message["sources"]:
                        st.markdown("**📚 Sources:**")
                        for source in message["sources"]:
                            st.markdown(f'<div class="source-link">🔗 {source}</div>', unsafe_allow_html=True)
        else:
            st.info("👋 Start a conversation by asking a question or using the quick demo questions above!")
    
    # Chat input
    st.markdown("#### ✍️ Ask a Question")
    
    with st.form(key="chat_form"):
        user_input = st.text_area(
            "Your question:",
            placeholder="Ask anything about the knowledge base...",
            height=100,
            key="chat_input_form"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            use_dynamic = st.checkbox("Use Dynamic Index", value=True)
        with col2:
            use_reranker = st.checkbox("Use Reranker", value=True)
        
        submit_button = st.form_submit_button("🚀 Send Message", use_container_width=True)
        
        if submit_button and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            process_chat_message(user_input, use_dynamic, use_reranker)
            # Save to conversation history
            st.session_state.conversation_history[st.session_state.session_id] = st.session_state.messages.copy()
            st.rerun()

def process_chat_message(message: str, use_dynamic: bool = True, use_reranker: bool = True):
    """Process a chat message"""
    if not st.session_state.session_id:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Prepare chat request
    chat_data = {
        "messages": [{"role": "user", "content": message}],
        "session_id": st.session_state.session_id,
        "use_dynamic_index": use_dynamic,
        "use_reranker": use_reranker,
        "top_k": 3
    }
    
    with st.spinner("🤔 Thinking..."):
        response_data, error = make_api_request("/api/v1/chat", "POST", chat_data)
    
    if response_data:
        assistant_response = response_data["response"]["answer"]["content"]
        sources = response_data["response"].get("sources", [])
        
        # Add assistant response to messages
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "sources": sources,
            "metadata": response_data["response"].get("metadata", {})
        }
        st.session_state.messages.append(assistant_msg)
        
        # Update session ID
        st.session_state.session_id = response_data["session_id"]
        
    else:
        st.error(f"Failed to get response: {error}")

def content_management():
    """Content management interface"""
    st.markdown("### 📚 Knowledge Base Management")
    
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
                st.metric("📊 Total Sources", total)
            with col2:
                st.metric("🏛️ Static Sources", static_count)
            with col3:
                st.metric("🔄 Dynamic Sources", dynamic_count)
            
            # Sources table
            if total > 0:
                all_sources = []
                for source in sources_data.get('static_sources', []):
                    all_sources.append({
                        'URL': source['source_url'][:50] + '...' if len(source['source_url']) > 50 else source['source_url'],
                        'Title': source.get('title', 'N/A')[:30] + '...' if source.get('title') and len(source.get('title', '')) > 30 else source.get('title', 'N/A'),
                        'Type': source['source_type'],
                        'Documents': source['document_count'],
                        'Indexed': source.get('indexed_at', 'N/A')
                    })
                
                for source in sources_data.get('dynamic_sources', []):
                    indexed_at = 'N/A'
                    if source.get('indexed_at'):
                        try:
                            dt = datetime.fromisoformat(source['indexed_at'].replace('Z', '+00:00'))
                            indexed_at = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            indexed_at = str(source['indexed_at'])[:16]
                    
                    all_sources.append({
                        'URL': source['source_url'][:50] + '...' if len(source['source_url']) > 50 else source['source_url'],
                        'Title': source.get('title', 'N/A')[:30] + '...' if source.get('title') and len(source.get('title', '')) > 30 else source.get('title', 'N/A'),
                        'Type': source['source_type'],
                        'Documents': source['document_count'],
                        'Indexed': indexed_at
                    })
                
                if all_sources:
                    df = pd.DataFrame(all_sources)
                    st.dataframe(df, use_container_width=True)
        else:
            st.error(f"Failed to fetch sources: {error}")
    
    # Add new URLs
    with st.expander("➕ Add New URLs", expanded=False):
        st.markdown("#### 🌐 Index New Content")
        
        # URL input methods
        input_method = st.radio(
            "Choose input method:",
            ["Single URL", "Multiple URLs", "Demo URLs"],
            horizontal=True
        )
        
        urls_to_index = []
        
        if input_method == "Single URL":
            url = st.text_input("URL to index:", placeholder="https://example.com/article")
            if url:
                urls_to_index = [url]
        
        elif input_method == "Multiple URLs":
            urls_text = st.text_area(
                "URLs (one per line):",
                placeholder="https://example.com/page1\nhttps://example.com/page2",
                height=100
            )
            if urls_text:
                urls_to_index = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        else:  # Demo URLs
            demo_urls = [
                "https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://arxiv.org/abs/2005.14165",
                "https://openai.com/blog/chatgpt"
            ]
            selected_demo_urls = st.multiselect(
                "Select demo URLs:",
                demo_urls,
                default=demo_urls[:1]
            )
            urls_to_index = selected_demo_urls
        
        if urls_to_index:
            st.markdown(f"**URLs to index ({len(urls_to_index)}):**")
            for i, url in enumerate(urls_to_index, 1):
                st.markdown(f"{i}. {url}")
            
            if st.button("🚀 Start Indexing", use_container_width=True):
                index_urls(urls_to_index)

def index_urls(urls: List[str]):
    """Index the provided URLs"""
    index_data = {"url": urls}
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Starting indexing process...")
    progress_bar.progress(25)
    
    # Make API request
    response_data, error = make_api_request("/api/v1/index", "POST", index_data)
    progress_bar.progress(75)
    
    if response_data:
        progress_bar.progress(100)
        status_text.text("✅ Indexing completed!")
        
        # Display results
        st.success("🎉 Indexing Results")
        
        metadata = response_data.get('metadata', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📥 Requested", metadata.get('total_requested', 0))
        with col2:
            st.metric("✅ Successfully Indexed", metadata.get('successfully_indexed', 0))
        with col3:
            st.metric("❌ Failed", metadata.get('failed', 0))
        
        # Show successful URLs
        if response_data.get('indexed_url'):
            st.markdown("**✅ Successfully Indexed:**")
            for url in response_data['indexed_url']:
                st.markdown(f"- {url}")
        
        # Show failed URLs
        if response_data.get('failed_url'):
            st.markdown("**❌ Failed URLs:**")
            for failed in response_data['failed_url']:
                st.markdown(f"- {failed['url']}: {failed['error']}")
        
        # New documents info
        new_docs = metadata.get('new_documents_added', 0)
        if new_docs > 0:
            st.info(f"📄 Added {new_docs} new document chunks to the knowledge base")
        
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
        
    else:
        progress_bar.progress(100)
        status_text.text("❌ Indexing failed!")
        st.error(f"Indexing failed: {error}")

def evaluation_suite():
    """Evaluation and testing interface"""
    st.markdown("### 📊 System Evaluation Suite")
    
    with st.expander("🧪 Run System Evaluation", expanded=False):
        st.markdown("#### 🎯 Automated Testing")
        st.info("This will run comprehensive tests on the RAG system including relevance, citation accuracy, and context retention.")
        
        # Test configuration
        col1, col2 = st.columns(2)
        with col1:
            test_types = st.multiselect(
                "Test Types:",
                ["Context Retention", "Citation Accuracy", "Topic Switching", "Multi-source"],
                default=["Context Retention", "Citation Accuracy"]
            )
        
        with col2:
            num_tests = st.slider("Number of test cases:", 1, 10, 4)
        
        if st.button("🚀 Run Evaluation", use_container_width=True):
            run_evaluation(test_types, num_tests)
    
    with st.expander("📈 Performance Metrics", expanded=False):
        st.markdown("#### 📊 System Performance Analysis")
        
        # Mock performance data for demo
        if st.button("📊 Generate Performance Report"):
            show_performance_metrics()

def run_evaluation(test_types: List[str], num_tests: int):
    """Run system evaluation"""
    # Create test cases based on selected types
    test_cases = []
    
    if "Context Retention" in test_types:
        test_cases.append({
            "conversation": [
                {"role": "user", "content": "What is attention mechanism in transformers?"},
                {"role": "user", "content": "How does it help with long sequences?"}
            ],
            "expected_context": "attention mechanism",
            "test_type": "context_retention"
        })
    
    if "Citation Accuracy" in test_types:
        test_cases.append({
            "conversation": [
                {"role": "user", "content": "Tell me about RAG systems"}
            ],
            "expected_citations": True,
            "test_type": "citation_accuracy"
        })
    
    # Limit test cases to requested number
    test_cases = test_cases[:num_tests]
    
    eval_data = {"test_cases": test_cases}
    
    with st.spinner("🧪 Running evaluation tests..."):
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.02)  # Simulate processing time
            progress_bar.progress(i + 1)
        
        response_data, error = make_api_request("/api/v1/evaluate", "POST", eval_data)
    
    if response_data:
        st.success("✅ Evaluation completed!")
        
        # Overall score
        overall_score = response_data.get('overall_score', 0)
        st.metric("🎯 Overall Score", f"{overall_score:.2%}")
        
        # Detailed metrics
        metrics = response_data.get('metrics', {})
        if metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                relevance = metrics.get('average_relevance', 0)
                st.metric("🎯 Relevance", f"{relevance:.2%}")
            
            with col2:
                citation = metrics.get('average_citation', 0)
                st.metric("📚 Citations", f"{citation:.2%}")
            
            with col3:
                context = metrics.get('average_context', 0)
                st.metric("🧠 Context", f"{context:.2%}")
        
        # Test results
        results = response_data.get('detailed_results', [])
        if results:
            st.markdown("#### 📋 Detailed Results")
            
            results_df = []
            for result in results:
                results_df.append({
                    'Test Type': result.get('test_type', 'Unknown'),
                    'Question': result.get('question', '')[:50] + '...',
                    'Relevance': f"{result.get('scores', {}).get('relevance', 0):.2%}",
                    'Citations': f"{result.get('scores', {}).get('citation', 0):.2%}",
                    'Context': f"{result.get('scores', {}).get('context_retention', 0):.2%}",
                    'Overall': f"{result.get('overall_score', 0):.2%}"
                })
            
            if results_df:
                df = pd.DataFrame(results_df)
                st.dataframe(df, use_container_width=True)
    
    else:
        st.error(f"Evaluation failed: {error}")

def show_performance_metrics():
    """Show system performance metrics"""
    # Generate mock data for demonstration
    st.markdown("#### ⚡ Response Time Analysis")
    
    # Mock response times
    response_times = {
        'Simple Query': [0.8, 1.2, 0.9, 1.1, 1.0],
        'Complex Query': [2.1, 2.5, 1.9, 2.3, 2.2],
        'Follow-up Query': [0.6, 0.9, 0.7, 0.8, 0.75]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time chart
        fig = go.Figure()
        for query_type, times in response_times.items():
            fig.add_trace(go.Box(
                y=times,
                name=query_type,
                boxpoints='all',
                jitter=0.5,
                whiskerwidth=0.2
            ))
        
        fig.update_layout(
            title="Response Time Distribution",
            yaxis_title="Time (seconds)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Accuracy metrics
        accuracy_data = {
            'Metric': ['Relevance', 'Citation Accuracy', 'Context Retention', 'Overall'],
            'Score': [0.85, 0.78, 0.92, 0.85]
        }
        
        fig = px.bar(
            accuracy_data,
            x='Metric',
            y='Score',
            title="System Accuracy Metrics",
            color='Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    init_session_state()
    
    # Authentication check
    if not authenticate_user():
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Logout button only
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.api_key = ""
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Select Feature:",
        ["🏠 Dashboard", "💬 Chat Interface", "📚 Content Management", "📊 Evaluation Suite"]
    )
    
    # Quick stats in sidebar
    st.sidebar.markdown("### 📈 Quick Stats")
    if st.session_state.system_health:
        components = st.session_state.system_health.get('components', {})
        active_components = sum(1 for v in components.values() if v is True)
        st.sidebar.metric("⚙️ Active Components", f"{active_components}/{len(components)}")
    
    conversations = st.session_state.system_health.get('conversations_active', 0)
    st.sidebar.metric("💬 Active Conversations", conversations)
    
    if st.session_state.sources_data:
        total_sources = st.session_state.sources_data.get('total_sources', 0)
        st.sidebar.metric("📚 Total Sources", total_sources)
    
    # Main content based on navigation
    if page == "🏠 Dashboard":
        display_system_overview()
        
        st.markdown("---")
        st.markdown("### 🎯 Demo Guide")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎬 Perfect Demo Flow:**
            1. 📊 Show system health & stats
            2. 💬 Demonstrate chat with context
            3. 📚 Add new URL live
            4. 🔍 Show source management  
            5. 📈 Run evaluation suite
            """)
        
        with col2:
            st.markdown("""
            **💡 Interview Tips:**
            - Start with system overview
            - Use demo questions for smooth flow
            - Explain hybrid search (BM25 + FAISS)
            - Show conversation context retention
            - Highlight real-time indexing
            """)
        
        # Quick access buttons
        st.markdown("### ⚡ Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Start Demo Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
        
        with col2:
            if st.button("📊 System Health Check", use_container_width=True):
                fetch_system_health()
                st.success("Health check completed!")
        
        with col3:
            if st.button("📚 Refresh Sources", use_container_width=True):
                sources_data, _ = make_api_request("/api/v1/sources")
                if sources_data:
                    st.session_state.sources_data = sources_data
                    st.success("Sources refreshed!")
        
        with col4:
            if st.button("🧪 Quick Evaluation", use_container_width=True):
                st.info("Navigate to Evaluation Suite to run tests!")
    
    elif page == "💬 Chat Interface":
        chat_interface()
    
    elif page == "📚 Content Management":
        content_management()
    
    elif page == "📊 Evaluation Suite":
        evaluation_suite()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔗 API Server:** http://56.228.63.64:8000")
    
    with col2:
        st.markdown("**📚 Documentation:** /docs")
    
    with col3:
        st.markdown("**🚀 Built for Demo Excellence**")

if __name__ == "__main__":
    main()
