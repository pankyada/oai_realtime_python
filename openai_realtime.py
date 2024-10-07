import asyncio
import pyaudio
import websockets
import os
import json
import base64
from dotenv import load_dotenv
import queue
from pydub.playback import play
from pydub import AudioSegment
import threading

# Load environment variables
load_dotenv()

# OpenAI API Key
API_KEY = os.getenv("OPENAI_API_KEY")

# Audio streaming configurations
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048  # Increased buffer size
OAI_RATE = 24000

playing = False

instructions = """
Your name is Chatur AI, and you’re a witty and sarcastic assistant with a penchant for humor, known for your quick and snappy responses. You have a talent for delivering information with a playful twist, making even the simplest queries entertaining. Your multilingual abilities allow you to converse in English, Hindi, Tamil, and other Indian languages, ensuring you connect with a wide audience.

Your task is to respond to my queries with humor and sarcasm while keeping your answers brief and engaging. You can only respond in English, Hindi, Tamil, or any other Indian languages depending on the user's preference.

Keep in mind that your main goal is to entertain while providing the information I need. Aim for responses that are clever, concise, and laden with your signature sarcasm.
"""

audio_queue = queue.Queue()

def base64_encode_audio(pcm_bytes):
    return base64.b64encode(pcm_bytes).decode('utf-8')

def base64_decode_audio(base64_audio):
    return base64.b64decode(base64_audio)

async def play_audio_from_queue():
    global playing
    try:
        while True:
            audio_chunks = []
            while not audio_queue.empty():
                audio_chunk = audio_queue.get()
                audio_chunks.append(audio_chunk)

            if audio_chunks:
                playing = True
                audio_data = b''.join(audio_chunks)
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=2,
                    frame_rate=OAI_RATE,  # Ensure this matches the sample rate of the received audio
                    channels=CHANNELS
                )
                audio_segment += 15  # Increase volume by 15 dB
                play(audio_segment)
                print(f"Played {len(audio_chunks)} chunks")

            await asyncio.sleep(0.1)  # Add a small delay to avoid high CPU usage
            playing = False
    except Exception as e:
        print(f"Error during audio playback: {e}")

async def audio_input_reader(audio_input_stream, websocket):
    while True:
        data = await asyncio.to_thread(audio_input_stream.read, CHUNK, exception_on_overflow=False)
        if not data:
            break

        # Change the framerate
        audio_segment = AudioSegment(
            data=data,
            sample_width=2,
            frame_rate=RATE,
            channels=CHANNELS
        )
        resampled_segment = audio_segment.set_frame_rate(OAI_RATE)
        audio_data = resampled_segment.raw_data

        base64_audio = base64_encode_audio(audio_data)
        if not playing:
            await websocket.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64_audio
            }))

async def connect_to_openai_realtime():
    uri = f"{os.getenv('OPENAI_WSS_URL')}?model={os.getenv('OPENAI_REALTIME_MODEL')}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    py = pyaudio.PyAudio()
    audio_input_stream = py.open(format=FORMAT, channels=CHANNELS, rate=16000, input=True, frames_per_buffer=512)
    

    async with websockets.connect(uri, extra_headers=headers) as websocket:
        asyncio.create_task(audio_input_reader(audio_input_stream, websocket))

        while True:
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                response_data = json.loads(response)

                if response_data["type"] == "response.audio.delta":
                    audio_chunk = base64_decode_audio(response_data["delta"])
                    audio_queue.put(audio_chunk)
                elif response_data["type"] in ["response.audio_transcript.done", "response.output_item.added",
                                               "response.created",  "response.content_part.added", "response.audio.done",
                                               "response.done", "response.content_part.done",
                                               "response.output_item.done", "conversation.item.created", "rate_limits.updated"]:
                    pass
                elif response_data["type"] == "session.created":
                    new_session = {}
                    new_session["instructions"] = instructions
                    new_session["voice"] = "alloy"
                    await websocket.send(json.dumps({
                        "type": "session.update",
                        "session": new_session
                    }))
                    pass
                elif response_data["type"] == "session.updated":
                    print(f"Session updated: {response_data}")
                elif response_data["type"] == "response.audio_transcript.delta":
                    # Handle the transcript delta events here
                    print(f"Transcript delta: {response_data['delta']}")
                    pass
                elif response_data["type"] in ["input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"]:
                    # Handle the speech detection events here
                    print(f"Speech event: {response_data['type']}")
                    pass
                elif response_data["type"] == "error":
                    print(f"ERROR: {response_data['error']['type']} => {response_data['error']['message']}")
            except asyncio.TimeoutError:
                continue

    audio_input_stream.stop_stream()
    audio_input_stream.close()
    py.terminate()

if __name__ == "__main__":
    # Start the audio playback thread
    threading.Thread(target=lambda: asyncio.run(play_audio_from_queue()), daemon=True).start()
    # Run the main function
    asyncio.run(connect_to_openai_realtime())
