from pydoc import text
from django.conf import settings
import json
import os

from ai.utils.open_ai_manager import OpenAIManager

def test_get_response():
    manager = OpenAIManager(model="gpt-4o", api_key=settings.OPEN_AI_SECRET_KEY)
    manager.add_message("system", text="You are a helpful assistant, that receives a text and will generate a json including user_message and a random id")
    manager.add_message("system", text="Format of the json is like {'user_message': <user_message>, 'id': <random_id>}")
    manager.add_message("user", text="Hello, world!")
    response = manager.generate_response()
    cost = manager.get_cost()
    json_response = json.loads(response)
    print(json_response['id'])
    print(f"Response: {json.dumps(json_response, indent=2)}")
    print(f"Cost: {cost}")

def test_convert_html_to_text():
    html_file_path = os.path.join(settings.MEDIA_ROOT, 'index.html')
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    manager = OpenAIManager(model="gpt-4o", api_key=settings.OPEN_AI_SECRET_KEY)
    simple_text = manager.build_simple_text_from_html(html_content)
    with open(os.path.join(settings.MEDIA_ROOT, 'simple_text.txt'), 'w', encoding='utf-8') as file:
        file.write(simple_text)
    print(f"Successfully Done")

def test_chunking():
    html_file_path = os.path.join(settings.MEDIA_ROOT, 'index.html')
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    manager = OpenAIManager(model="gpt-4o", api_key=settings.OPEN_AI_SECRET_KEY)
    chunks = manager.build_chunks(text=html_content, max_chunk_size=1000)
    for i, chunk in enumerate(chunks):
        html_src = chunk["html"]
        simple_text = chunk["text"]
        with open(os.path.join(settings.MEDIA_ROOT, f'chunk_{i}.html'), 'w', encoding='utf-8') as file:
            file.write(html_src)
        with open(os.path.join(settings.MEDIA_ROOT, f'chunk_{i}.txt'), 'w', encoding='utf-8') as file:
            file.write(simple_text)
    print(f"Successfully Done")

def test_ai_tts():
    manager = OpenAIManager(model="gpt-4o", api_key=settings.OPEN_AI_SECRET_KEY)
    # Simulate SSML tags for OpenAI TTS
    my_var = "HEY GUYS!!!!!!"
    text = f"""
        "{my_var} ... "
        "I am SO EXCITED to speak with you today. "
        "    This is a demonstration (with a higher pitch) of OpenAI's text-to-speech capabilities. "
        "Can you hear the happiness in my voice? "
        "Let's make this a WONDERFUL EXPERIENCE together!"
    """
    audio_bytes = manager.tts(text=text, voice="nova", audio_format="mp3")
    audio_file_path = os.path.join(settings.MEDIA_ROOT, 'audio.mp3')
    with open(audio_file_path, 'wb') as file:
        file.write(audio_bytes)
    print(manager.get_cost())
    print(f"Successfully Done")

def test_ai_stt():
    manager = OpenAIManager(model="gpt-4o", api_key=settings.OPEN_AI_SECRET_KEY)
    audio_file_path = os.path.join(settings.MEDIA_ROOT, 'audio.mp3')
    with open(audio_file_path, 'rb') as file:
        audio_bytes = file.read()
    text = manager.stt(audio_input=audio_bytes, input_type="bytes")
    print(f"Transcribed Text: {text}")

def test_openai_manager():
    test_ai_stt()