from django.conf import settings
import openai
import wave
import contextlib
import io
import requests
import random
import re

from ai.utils.chunk_manager import ChunkPipeline

class OpenAIManager:
    def __init__(self, model, api_key):
        """
        Initialize the OpenAIManager.
        
        Args:
            model (str): The OpenAI model name (e.g., 'gpt-4o', 'gpt-3.5-turbo').
            api_key (str): Your OpenAI API key.
        
        Returns:
            None
        
        Example:
            manager = OpenAIManager(model="gpt-4o", api_key="sk-...")
        """
        self.OPENAI_PRICING = {
            "gpt-3.5-turbo": {
                "input_per_1k_token": 0.0005,
                "output_per_1k_token": 0.0015,
            },
            "gpt-4-turbo": {
                "input_per_1k_token": 0.001,
                "output_per_1k_token": 0.003,
                "image_per_1_image": 0.00765,
            },
            "gpt-4o": {
                "input_per_1k_token": 0.0005,
                "output_per_1k_token": 0.0015,
                "audio_stt_per_1_minute": 0.006,
                "image_per_1_image": 0.00765,
                "tts_standard_per_1k_char": 0.015,
                "tts_premium_per_1k_char": 0.030,
            },
            "gpt-4": {
                "input_per_1k_token": 0.03,
                "output_per_1k_token": 0.06,
            },
            "text-embedding-3-small": {
                "input_per_1k_token": 0.00002,
            },
            "text-embedding-3-large": {
                "input_per_1k_token": 0.00013,
            },
            "whisper": {
                "audio_stt_per_1_minute": 0.006,
            },
        }
        self.OPEN_AI_CLIENT = openai.OpenAI(api_key=api_key)
        self.model = model
        self.messages = []
        self.memorized_conversations = ""
        self.cost = 0

    def _random_generator(self, length=16):
        """
        Generate a random string of specified length.
        
        Args:
            length (int): Length of the random string. Default is 16.
        
        Returns:
            str: Randomly generated string.
        
        Example:
            token = self._random_generator(8)
        """
        characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(random.choice(characters) for _ in range(length))

    def _clean_code_block(self, response_text):
        # Remove code block markers like ```json, ```html, and ending ```
        pattern = r"^```(?:json|html)?\n?(.*)```$"
        match = re.match(pattern, response_text.strip(), re.DOTALL)
        if match:
            return match.group(1).strip()
        return response_text.strip()
    
    def get_cost(self):
        """
        Get the total accumulated cost of API calls.
        
        Returns:
            float: Total cost in USD.
        
        Example:
            total = manager.get_cost()
        """
        return self.cost

    def clear_cost(self):
        """
        Reset the accumulated cost to zero.
        
        Returns:
            None
        
        Example:
            manager.clear_cost()
        """
        self.cost = 0

    def clear_messages(self):
        """
        Clear the message history.
        
        Returns:
            None
        
        Example:
            manager.clear_messages()
        """
        self.messages = []
    
    def add_message(self, role, text=None, img_url=None, max_history=5):
        """
        Add a message to the conversation history. Supports multimodal (text + image) messages.
        
        Args:
            role (str): One of 'system', 'user', 'assistant'.
            text (str): The text content.
            img_url (str): The image URL (optional).
            max_history (int): Maximum number of messages to keep in history. Default is 5.
        
        Returns:
            None
        
        Example:
            manager.add_message("user", text="Describe this image.", img_url="https://example.com/image.png")
            manager.add_message("user", text="Hello!")
        """
        if role not in ["system", "user", "assistant"]:
            return
        content = []
        if text is not None:
            content.append({"type": "text", "text": text})
        if img_url is not None:
            content.append({"type": "image_url", "image_url": {"url": img_url}})
        if role == "system":
            sys_text = text if text is not None else ""
            if self.messages and self.messages[0]["role"] == "system":
                self.messages[0]["content"] += f"\n{sys_text}"
            else:
                self.messages.insert(0, {"role": "system", "content": sys_text})
        else:
            msg_content = content if content else text
            self.messages.append({"role": role, "content": msg_content})
            if len(self.messages) > max_history + 2:
                old_msgs = self.messages[1:-max_history] if self.messages and self.messages[0]["role"] == "system" else self.messages[:-max_history]
                if old_msgs:
                    summary_lines = []
                    for msg in old_msgs:
                        if msg["role"] == "user":
                            summary_lines.append(f"User said: {msg['content']}")
                        elif msg["role"] == "assistant":
                            summary_lines.append(f"Assistant said: {msg['content']}")
                    summary_text = "\n".join(summary_lines)
                    summarized = self.summarize(summary_text)
                    if self.messages and self.messages[0]["role"] == "system":
                        self.messages[0]["content"] += f"\n{summarized}"
                        self.messages = [self.messages[0]] + self.messages[-max_history:]
                    else:
                        self.messages = [{"role": "system", "content": summarized}] + self.messages[-max_history:]

    def build_chunks(self, text, max_chunk_size=1000):
        """
        Chunk text into manageable pieces for processing.
        
        Args:
            text (str): The input text to chunk.
            max_chunk_size (int): Maximum size of each chunk. Default is 1000.
        
        Returns:
            list: List of chunk dicts with 'html' and 'text' keys.
        
        Example:
            chunks = manager.build_chunks(long_text, max_chunk_size=500)
        """
        chunk_pipeline = ChunkPipeline()
        chunks = chunk_pipeline.process(text, max_chunk_size)
        for i in range(len(chunks) - 1):
            head, tail = chunk_pipeline.chunker.get_incomplete_end(chunks[i]["html"])
            if tail:
                chunks[i]["html"] = head
                chunks[i]["text"] = head
                chunks[i + 1]["html"] = tail + chunks[i + 1]["html"]
                chunks[i + 1]["text"] = tail + chunks[i + 1]["text"]
        return chunks

    def generate_response(self, max_token=2000, messages=None):
        """
        Generate a response from the OpenAI chat model.
        
        Args:
            max_token (int): Maximum number of tokens in the response. Default is 2000.
            messages (list): List of message dicts. If None, uses internal history.
        
        Returns:
            str: The assistant's response text.
        
        Example:
            reply = manager.generate_response(max_token=500)
        """
        if messages is None:
            messages = self.messages
        response = self.OPEN_AI_CLIENT.chat.completions.create(
            model=self.model,
            messages=messages if messages else self.messages,
            max_tokens=max_token
        )
        tokens_used = response.usage
        prompt_tokens = tokens_used.prompt_tokens
        completion_tokens = tokens_used.completion_tokens
        pricing = self.OPENAI_PRICING.get(self.model, {})
        input_price = pricing.get("input_per_1k_token", 0)
        output_price = pricing.get("output_per_1k_token", 0)
        
        cost = (prompt_tokens / 1000) * input_price + (completion_tokens / 1000) * output_price
        self.cost += cost
        
        raw_response = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
        return self._clean_code_block(raw_response)

    def summarize(self, text, max_summary_input=15000, max_length=1000, max_chunk_size=1000):
        """
        Summarize a long text using the chat model, chunking if needed.
        
        Args:
            text (str): The text to summarize.
            max_summary_input (int): Max input size for a single summary. Default 15000.
            max_length (int): Max tokens for each summary. Default 1000.
            max_chunk_size (int): Chunk size for splitting text. Default 1000.
        
        Returns:
            str: Summarized text.
        
        Example:
            summary = manager.summarize(long_text)
        """
        def recursive_summarize(text):
            if len(text) <= max_summary_input:
                messages = [
                    {"role": "system", "content": "You are a summarization expert. Summarize the following text."},
                    {"role": "user", "content": text}
                ]
                response = self.generate_response(max_token=max_length, messages=messages)
                return response
            else:
                chunks = self.build_chunks(text, max_chunk_size=max_chunk_size)
                summaries = []
                for chunk in chunks:
                    self.clear_messages()
                    self.add_message("system", "You are a summarization expert. Summarize the following text.")
                    self.add_message("user", chunk["text"])
                    response = self.generate_response(max_token=max_length)
                    summaries.append(response)
                combined = " ".join(summaries)
                return recursive_summarize(combined)
        return recursive_summarize(text)
    
    def stt(self, audio_input, response_format="text", language=None, input_type="url"):
        """
        Transcribe speech to text using OpenAI Whisper.
        
        Args:
            audio_input: Audio data (bytes, file path, or URL).
            response_format (str): Output format. Options: 'text', 'json', 'srt', 'verbose_json'. Default 'text'.
            language (str): Language code (e.g., 'en'). Optional.
            input_type (str): Type of input. Options: 'bytes', 'url', 'file'. Default 'url'.
        
        Returns:
            str or dict: Transcription result in the requested format.
        
        Example:
            # Using bytes
            with open('audio.wav', 'rb') as f:
                audio_bytes = f.read()
            text = manager.stt(audio_bytes, input_type='bytes')
            # Using URL
            text = manager.stt('https://example.com/audio.wav', input_type='url')
            # Using file path
            text = manager.stt('/path/to/audio.wav', input_type='file')
        """
        if input_type == "bytes":
            audio_file = io.BytesIO(audio_input)
            audio_file.name = f"{self._random_generator()}.wav"
            file_for_api = audio_file
            try:
                audio_file.seek(0)
                with contextlib.closing(wave.open(audio_file, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration_seconds = frames / float(rate)
            except Exception:
                duration_seconds = 0
        elif input_type == "url":
            response_url = requests.get(audio_input)
            audio_bytes = response_url.content
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"{self._random_generator()}.wav"
            file_for_api = audio_file
            try:
                audio_file.seek(0)
                with contextlib.closing(wave.open(audio_file, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration_seconds = frames / float(rate)
            except Exception:
                duration_seconds = 0
        else:
            file_for_api = audio_input
            try:
                with contextlib.closing(wave.open(audio_input, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration_seconds = frames / float(rate)
            except Exception:
                duration_seconds = 0
        response = self.OPEN_AI_CLIENT.audio.transcriptions.create(
            model="whisper-1",
            file=file_for_api,
            response_format=response_format,
            language=language
        )
        duration_minutes = duration_seconds / 60
        pricing = self.OPENAI_PRICING.get("whisper", {})
        input_price = pricing.get("audio_stt_per_1_minute", 0)
        cost = duration_minutes * input_price
        self.cost += cost
        if response_format == "text":
            return response.text
        elif response_format == "json":
            return response.json["text"]
        elif response_format == "srt":
            return response.srt
        elif response_format == "verbose_json":
            return response.verbose_json
        else:
            return response

    def tts(self, text, voice="en-US-Wavenet-D", audio_format="mp3", model="tts-1"):
        """
        Convert text to speech using OpenAI TTS.
        
        Args:
            text (str): The text to synthesize.
            voice (str): Voice name (e.g., 'en-US-Wavenet-D'). Default is 'en-US-Wavenet-D'.
            audio_format (str): Output format. Options: 'mp3', 'wav', 'ogg'. Default 'mp3'.
            model (str): TTS model. Options: 'tts-1', 'tts-1-hd'. Default 'tts-1'.
        
        Returns:
            bytes: Audio content in the requested format.
        
        Example:
            audio = manager.tts("Hello world!", voice="en-US-Wavenet-D", audio_format="mp3")
            with open("output.mp3", "wb") as f:
                f.write(audio)
        """
        response = self.OPEN_AI_CLIENT.audio.speech.create(
            model=model,
            input=text,
            voice=voice,
            response_format=audio_format
        )
        pricing = self.OPENAI_PRICING.get("gpt-4o", {})
        if model == "tts-1-hd":
            input_price = pricing.get("tts_premium_per_1k_char", 0)
        else:
            input_price = pricing.get("tts_standard_per_1k_char", 0)
        char_count = len(text)
        cost = (char_count / 1000) * input_price
        self.cost += cost
        return response.content

    def generate_image(self, prompt, size="1024x1024"):
        """
        Generate an image from a text prompt using OpenAI's DALL-E model.

        Args:
            prompt (str): The text prompt to generate the image.
            size (str): The size of the generated image. Default is "1024x1024".

        Returns:
            bytes: The generated image in bytes.

        Example:
            image = manager.generate_image("A futuristic cityscape", size="512x512")
            with open("output.png", "wb") as f:
                f.write(image)
        """
        response = self.OPEN_AI_CLIENT.images.generate(
            model="dall-e",
            prompt=prompt,
            size=size
        )
        image_url = response.data[0].url
        image_bytes = requests.get(image_url).content
        pricing = self.OPENAI_PRICING.get("gpt-4o", {})
        image_price = pricing.get("image_per_1_image", 0)
        self.cost += image_price
        return image_bytes
