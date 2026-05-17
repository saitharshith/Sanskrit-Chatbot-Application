import os
import asyncio
import nest_asyncio
import edge_tts
from langchain_core.tools import tool

# This line prevents "Event loop is already running" crashes in Jupyter/Streamlit
nest_asyncio.apply()

@tool
def generate_sanskrit_audio(text: str) -> str:
    """
    Use this tool ONLY when the user explicitly asks to hear how a Sanskrit word,
    Shloka, or phrase is pronounced, chanted, or spoken.
    Input: The exact Sanskrit text (must be in Devanagari script) to be spoken.
    """
    if not text or not isinstance(text, str):
        return "Error: Invalid text provided for TTS."

    # Edge-TTS outputs MP3 natively
    os.makedirs("outputs", exist_ok=True)
    output_filepath = os.path.abspath(os.path.join("outputs", "guru_spoken_response.mp3"))

    async def _generate_audio():
        # Using pure Python API, bypassing Windows command line entirely!
        communicate = edge_tts.Communicate(
            text=text,
            voice="hi-IN-MadhurNeural", # Hindi male voice for Sanskrit approximation
            rate="-20%",             # Slightly slower rate for clarity and classical feel
            pitch="-2Hz"             # Slightly lower pitch for a more resonant, authoritative tone
        )
        await communicate.save(output_filepath)

    try:
        # Run the generation
        asyncio.run(_generate_audio())

        return (
            f"SUCCESS. The audio file was generated at: {output_filepath}\n"
            f"Instruction for Guru: Tell the student 'Listen closely to my pronunciation, my child...' "
            f"and inform them the audio is ready to play."
        )
    except Exception as e:
        return f"An illusion (Maya) interrupted my voice. Error: {str(e)}"