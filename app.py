import streamlit as st
import requests

# 🔑 API details
API_URL = "https://269d4b2a64b7.ngrok-free.app"
API_KEY = "demo-api-key-123"  # or whichever key matches your permission

# 🎯 Helper: Send query to backend
def ask_question(query):
    headers = {"x-api-key": API_KEY}
    payload = {"query": query}
    response = requests.post(f"{API_URL}/api/v1/chat", json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# 🎯 Helper: Index a URL
def index_url(url):
    headers = {"x-api-key": API_KEY}
    payload = {"urls": [url]}
    response = requests.post(f"{API_URL}/api/v1/index", json=payload, headers=headers)
    return response.json()

# 🚀 Streamlit UI
st.title("RAG Web Q&A System")

st.sidebar.header("Index a Website")
new_url = st.sidebar.text_input("Enter URL to index")
if st.sidebar.button("Index URL"):
    with st.spinner("Indexing..."):
        result = index_url(new_url)
    st.sidebar.write(result)

query = st.text_input("Ask a question about indexed sites:")
if st.button("Send"):
    with st.spinner("Fetching answer..."):
        answer = ask_question(query)
    st.write("**Answer:**", answer.get("answer", "No answer"))
    
    if "citations" in answer:
        st.write("**Citations:**")
        for c in answer["citations"]:
            st.markdown(f"- [{c['title']}]({c['url']})")

