import streamlit as st
import requests

# --------------------
# CONFIG
# --------------------
API_URL = "https://22fe0b5754d9.ngrok-free.app"   # replace with your current ngrok/URL
API_KEY = "demo-api-key-123"                      # replace with the right key

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

# --------------------
# STREAMLIT APP
# --------------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("💬 RAG-Powered Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})

    try:
        # Call backend API
        response = requests.post(
            f"{API_URL}/api/v1/chat",
            headers=headers,
            json={"messages": st.session_state["messages"]}
        )

        if response.status_code == 200:
            data = response.json()
            answer = data["response"]["answer"]["content"]
            citations = data["response"].get("citation", [])

            st.session_state["messages"].append({"role": "assistant", "content": answer})

            # Display assistant response
            with st.chat_message("assistant"):
                st.write(answer)
                if citations:
                    st.caption(f"📚 Sources: {', '.join(citations)}")

        else:
            st.error(f"❌ Error {response.status_code}: {response.text}")

    except Exception as e:
        st.error(f"⚠️ Request failed: {e}")

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
