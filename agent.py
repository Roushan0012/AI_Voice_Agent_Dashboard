import asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import openai, silero
from utils import add_call, mark_call_connected, end_call 

load_dotenv()

class FinanceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are an AI voice agent named Raj from Bajaj Finance. "
                "You are calling regarding a financial backup plan. "
                "Keep your responses concise, natural, and conversational. "
                "Your goal is to introduce the Flexi Overdraft facility. "
                "Ask the user for their current employment status and their net salary to determine their limit."
            )
        )

async def entrypoint(ctx: JobContext):
    # 1. Create the call record in the database immediately
    from app import app
    with app.app_context():
        call_id = add_call(status='active')

    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")
    
    # 2. Update status to connected
    with app.app_context():
        mark_call_connected(call_id)

    # Initialize the Voice Session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
    )

    await session.start(room=ctx.room, agent=FinanceAgent())

    await session.say(
        "Namaste! Good afternoon, am I speaking with Mr. Sharma? This is Raj from Bajaj Finance.", 
        allow_interruptions=True
    )

    # ==========================================
    # 3. BULLETPROOF DISCONNECT TRIGGER
    # Automatically extracts and saves the conversation when the call ends
    # ==========================================
    @ctx.room.on("disconnected")
    def on_disconnect():
        print("📞 Call disconnected! Extracting transcript and saving to database...")
        
        transcript = ""
        try:
            # Safely grab the chat context from your 'session' variable
            chat_history = getattr(session, "chat_ctx", None) or getattr(session, "_chat_ctx", None)

            if chat_history and hasattr(chat_history, "messages"):
                for msg in chat_history.messages:
                    if msg.role != "system":
                        role_name = "AI (Raj)" if msg.role == "assistant" else "Customer"
                        
                        # LiveKit sometimes uses msg.text and sometimes msg.content
                        msg_data = getattr(msg, "content", None) or getattr(msg, "text", "")
                        
                        if msg_data:
                            transcript += f"{role_name}: {msg_data}\n"
            else:
                transcript = "Could not locate chat history."

        except Exception as e:
            print(f"⚠️ Error extracting transcript: {e}")
            transcript = "Error reading conversation."

        # Fallback if no talking happened
        if not transcript.strip():
            transcript = "No conversation recorded."

        # Send to utils.py to trigger Groq AI and update the Database!
        from utils import end_call
        end_call(call_id, transcript)
        print("✅ Call saved successfully!")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))