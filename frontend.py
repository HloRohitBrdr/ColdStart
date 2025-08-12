import streamlit as st
import requests
import json
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="🛍️ AI E-commerce Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Simple title
st.title("🛍️ AI Shopping Assistant")
st.markdown("*Your intelligent companion for the perfect shopping experience*")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if 'user_id' not in st.session_state:
    st.session_state.user_id = "user123"

# Sidebar for user settings
with st.sidebar:
    st.header("⚙️ Settings")
    user_id = st.text_input("User ID:", value=st.session_state.user_id)
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared!")

# Main chat area
st.header("💬 Chat")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show recommendations if available
        if message["role"] == "assistant" and "recommendations" in message:
            if message["recommendations"]:
                st.subheader("🎯 Recommendations")
                for rec in message["recommendations"]:
                    st.write(f"**{rec['name']}** - {rec['category']} - ₹{rec['price']:.2f}")

# Chat input
if prompt := st.chat_input("Ask me anything about shopping..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post("http://localhost:8000/chat",
                                         json={
                                             "user_id": st.session_state.user_id,
                                             "message": prompt,
                                             "session_id": st.session_state.session_id
                                         })

                if response.status_code == 200:
                    result = response.json()
                    st.write(result["reply"])

                    # Add to session state
                    assistant_message = {
                        "role": "assistant",
                        "content": result["reply"],
                        "recommendations": result.get("recommendations", [])
                    }
                    st.session_state.messages.append(assistant_message)

                    # Show recommendations
                    if result.get("recommendations"):
                        st.subheader("🎯 Recommendations")
                        for rec in result["recommendations"]:
                            st.write(f"**{rec['name']}** - {rec['category']} - ₹{rec['price']:.2f}")

                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Intent", result.get("intent", "general"))
                    with col2:
                        st.metric("Recommendations", len(result.get("recommendations", [])))
                    with col3:
                        st.metric("Status", "✅ Success")

                else:
                    st.error(f"Error: {response.status_code}")

            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# System status
st.sidebar.markdown("---")
st.sidebar.header("System Status")
try:
    health = requests.get("http://localhost:8000/", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("✅ Backend Online")
    else:
        st.sidebar.error("❌ Backend Error")
except:
    st.sidebar.error("❌ Backend Offline")