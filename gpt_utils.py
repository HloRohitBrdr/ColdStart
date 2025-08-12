import os
from openai import OpenAI

# Load Groq API key (either from env or hardcoded for development)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = "My_GROQ_API_KEY"
    print("⚠️ Using hardcoded Groq API key. Consider setting GROQ_API_KEY environment variable.")

# Initialize Groq client
try:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    print("✅ Groq API client initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Groq client: {e}")
    client = None

# You can use these models with Groq:
# - llama3-70b-8192 (best)
# - llama3-8b-8192
# - gemma-7b-it
# - mixtral-8x7b-32768
deployment = "llama3-70b-8192"  # Groq model name


def generate_gpt_response(prompt: str, user_context: dict = None):
    """
    Generate response using Groq LLaMA3
    Enhanced with user context for personalized responses
    """
    if not client:
        return "AI service is unavailable. Please try again later."

    # Build system prompt
    system_message = """You are an expert AI assistant for a premium e-commerce platform. You help customers with:

🛍️ Product recommendations  
💡 Shopping guidance  
👗 Fashion & style advice  
📦 Order tracking  
↩️ Return/refund help  
🎯 Personalized shopping

Be friendly, helpful, and always excited to assist users in finding what they want."""

    if user_context:
        system_message += f"\n\n🔍 User Context: {user_context}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq Error: {e}")
        return f"⚠️ AI error: {str(e)}"


def generate_conversational_response(messages: list, user_context: dict = None):
    """
    Multi-turn conversation support using Groq's LLaMA3
    """
    if not client:
        return "AI service is unavailable. Please try again later."

    system_message = """You are an expert conversational assistant for a premium e-commerce platform.

You:
🧠 Remember conversation context  
📦 Assist with orders & returns  
💡 Suggest products & outfits  
🎯 Offer smart shopping advice  

Stay helpful, warm, and always focus on user satisfaction."""

    if user_context:
        system_message += f"\n\n🔍 User Context: {user_context}"

    # Prepend system message
    full_messages = [{"role": "system", "content": system_message}] + messages

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=full_messages,
            max_tokens=800,
            temperature=0.7,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq Error: {e}")
        return f"⚠️ AI error: {str(e)}"
