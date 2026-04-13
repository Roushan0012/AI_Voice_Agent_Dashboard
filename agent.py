import asyncio
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli, Agent, AgentSession
from livekit.plugins import openai, silero
from utils import add_call, mark_call_connected, end_call  # <-- CRITICAL: IMPORT YOUR HELPERS

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
    # We wrap this in a thread because Flask-SQLAlchemy needs an app context
    from app import app
    with app.app_context():
        call_id = add_call(status='active')

    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")
    
    # 2. Update status to connected
    with app.app_context():
        mark_call_connected(call_id)

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

    # 3. Wait for the call to end, then save everything
    @ctx.room.on("participant_disconnected")
    def on_disconnected(participant):
        # Get the full transcript from the session
        final_transcript = ""
        for msg in session.chat_ctx.messages:
            if msg.role in ["user", "assistant"]:
                final_transcript += f"{msg.role}: {msg.text}\n"
        
        # Save transcript and run AI Analysis (Name, Sentiment, Outcome)
        with app.app_context():
            end_call(call_id, final_transcript)
        print(f"Call {call_id} ended and analyzed successfully!")

    @ctx.room.on("disconnected")
    def on_disconnect():
            print("📞 Call disconnected! Extracting transcript and saving to database...")
            
            # 1. Extract the full conversation from the AI's memory
            transcript = ""
            for msg in agent.chat_ctx.messages:
                # We skip the system prompt so it doesn't clutter the dashboard
                if msg.role != "system":
                    role_name = "AI (Raj)" if msg.role == "assistant" else "Customer"
                    transcript += f"{role_name}: {msg.content}\n"
            
            # 2. If the transcript is empty, just put a placeholder
            if not transcript.strip():
                transcript = "No conversation recorded."

            # 3. Send it to utils.py to analyze the sentiment and save to SQLite!
            from utils import end_call # Ensure it's imported
            end_call(call_id, transcript)
            print("✅ Call saved successfully!")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))