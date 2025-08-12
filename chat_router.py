from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from recommender import recommend_for_user, semantic_search, initialize_recommendation_system
from gpt_utils import generate_gpt_response, generate_conversational_response
from utils import detect_intent, get_order_status, get_return_policy
import json
import asyncio
import time

router = APIRouter()

# In-memory session storage (in production, use Redis or database)
conversation_sessions = {}

# Initialize recommendation system on startup
print("🚀 Initializing recommendation system...")
try:
    initialize_recommendation_system()
    print("✅ Recommendation system initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize recommendation system: {e}")


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatInput(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[Message]] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None
    intent: Optional[str] = None
    recommendations: Optional[List] = None
    processing_time: Optional[float] = None


@router.post("/chat", response_model=ChatResponse)
def chat(input: ChatInput):
    start_time = time.time()

    try:
        intent = detect_intent(input.message)
        session_id = input.session_id or input.user_id

        # Get conversation history
        if session_id in conversation_sessions:
            conversation_history = conversation_sessions[session_id]
        elif input.conversation_history:
            conversation_history = [msg.dict() for msg in input.conversation_history]
        else:
            conversation_history = []

        # Add current message to history
        conversation_history.append({"role": "user", "content": input.message})

        response_data = ChatResponse(
            reply="",
            session_id=session_id,
            intent=intent
        )

        if intent == "recommendation":
            print(f"🎯 Processing recommendation request for user {input.user_id}")
            # Get recommendations
            try:
                recommendations = recommend_for_user(input.user_id, input.message)
                response_data.recommendations = recommendations

                # Generate contextual response about recommendations
                rec_context = f"Based on the user's query '{input.message}', here are the recommended products: {json.dumps(recommendations)[:500]}..."
                reply = generate_gpt_response(
                    f"The user asked: '{input.message}'. Please provide a helpful response explaining these product recommendations in a conversational way. Be enthusiastic and highlight the best features!",
                    user_context={"recommendations": recommendations, "user_id": input.user_id}
                )
                response_data.reply = reply
                print(f"✅ Generated {len(recommendations)} recommendations")

            except Exception as e:
                print(f"❌ Error getting recommendations: {e}")
                response_data.reply = "I'm having trouble accessing our product recommendations right now. Let me help you in other ways! What type of products are you looking for?"
                response_data.recommendations = []

        elif intent == "order_status":
            print(f"📦 Processing order status request for user {input.user_id}")
            order_status = get_order_status(input.user_id)
            reply = generate_gpt_response(
                f"The user is asking about their order status. The current status is: '{order_status}'. Please provide a helpful, conversational response with enthusiasm!",
                user_context={"user_id": input.user_id}
            )
            response_data.reply = reply

        elif intent == "return_policy":
            print("↩️ Processing return policy request")
            return_policy = get_return_policy()
            reply = generate_gpt_response(
                f"The user is asking about return policy. The policy is: '{return_policy}'. Please explain this in a conversational and helpful way, making it sound customer-friendly!",
                user_context={"user_id": input.user_id}
            )
            response_data.reply = reply

        else:
            print("💬 Processing general conversation")
            # Use conversational AI for general queries with context
            if len(conversation_history) > 1:
                reply = generate_conversational_response(
                    conversation_history,
                    user_context={"user_id": input.user_id, "platform": "premium_e-commerce", "intent": intent}
                )
            else:
                reply = generate_gpt_response(
                    input.message,
                    user_context={"user_id": input.user_id, "platform": "premium_e-commerce", "intent": intent}
                )
            response_data.reply = reply

        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": response_data.reply})

        # Store conversation history (keep last 20 messages to manage memory)
        conversation_sessions[session_id] = conversation_history[-20:]

        # Calculate processing time
        processing_time = round(time.time() - start_time, 2)
        response_data.processing_time = processing_time

        print(f"✅ Chat response generated in {processing_time}s")
        return response_data

    except Exception as e:
        print(f"❌ Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/conversation/{session_id}")
def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "conversation": conversation_sessions.get(session_id, [])
    }


@router.delete("/conversation/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
    return {"message": "Conversation history cleared", "session_id": session_id}