"""
        This file is part of Outbound Phone GPT.

        Outbound Phone GPT is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        Outbound Phone GPT is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with Outbound Phone GPT.  If not, see <https://www.gnu.org/licenses/> 
"""

# Import configs
from __config__ import ELEVEN_LABS_API_KEY, STREAMING_LATENCY_VAL, VOICE_ID, MODEL_ID, VOICE_SETTINGS, AGENT_AUDIO_FOLDER, END_OF_STREAM_SIGNAL, DEFAULT_MESSAGE, LABEL_TO_FILLER, OUTPUT_FORMAT

# General Imports
import asyncio
import requests
import os
import io
import re
import logging
import aiohttp
import pickle
import subprocess
from collections import OrderedDict
from urllib.parse import urlencode

# Set up logging configuration
from logger_config import setup_logger
logger = setup_logger("my_app_logger", level=logging.DEBUG)

################################################ ASYNC FUNCTIONS ####################################################
async def generate_audio_file(message: str, file_name : str, type : str):
    """
    Generates mp3 audio files from a given input `message`.

    Args:
    message : The text content of the audio to be generated through text to speech
    file_name : The preferred name of the file where the generated audio will be saved. The name should be given without an extension e.g 
        file_name = myaudio is acceptable but file_name=myaudio.mp3 is not acceptable
    type: choose from 'starter' or 'filler'. 
    'starter' indicates that the audio file is to be used for playing welcome messages on the start of the call
    'filler' indicates that the audio file is to be used as a filler between user speech input and GPT audio output, to reduce response latency
    
    """
    _file_path_starter = os.path.join(AGENT_AUDIO_FOLDER, file_name + "-starter.wav")
    _file_path_filler = os.path.join(AGENT_AUDIO_FOLDER, file_name + "-filler.wav")

    if os.path.exists(_file_path_starter):
        logger.info(f"File already exists: {_file_path_starter}")
        return _file_path_starter
    elif os.path.exists(_file_path_filler):
        logger.info(f"File already exists: {_file_path_filler}")
        return _file_path_filler
    else:
        if type == "starter":
            file_name = file_name + "-starter.mp3"
        elif type == "filler":
            file_name = file_name + "-filler.mp3"
        else:
            return "Invalid type. You can only choose type `starter` or `filler`"
        
        file_path = os.path.join(f'{AGENT_AUDIO_FOLDER}', file_name)

        CHUNK_SIZE = 1024
        # Prepare query parameters
        query_params = {'optimize_streaming_latency': STREAMING_LATENCY_VAL}
        query_string = urlencode(query_params)

        # Include query parameters in the URL
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream?{query_string}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVEN_LABS_API_KEY
        }

        data = {
            "text": message,
            "model_id": MODEL_ID,
            "voice_settings": VOICE_SETTINGS
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    print(f"HTTP Error: {response.status}")
                    return None

                try:
                    with open(file_path, "wb") as audio_file:
                        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                            audio_file.write(chunk)
                        wav_file = convert_to_mulaw_with_ffmpeg(file_path)
                        os.remove(file_path)
                        return wav_file  # Return the file path where audio is saved
                except Exception as e:
                    print("Exception occurred:", str(e))
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return None

async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""

    async for text in chunks:
        if text is not None:
            if buffer.endswith(splitters):
                yield buffer + " "
                buffer = text
            elif text.startswith(splitters):
                yield buffer + text[0] + " "
                buffer = text[1:]
            else:
                buffer += text

    if buffer:
        yield buffer + " "

async def get_cached_streaming_generator(cached_response: list):
        """
        Converts a list of words into a streaming generator

        Args:
        cached_response : List of words from a pre-generated GPT response 

        Returns:
        None : This method doesn't return anything
        """
        for response_content in cached_response:
            chunk = {
            "choices": [
                {
                    "delta": {
                        "content": response_content 
                    }
                }
            ]
        }
            print(f"Caching text chunk: {response_content}")
            yield chunk

async def _asend_text_chunk_to_eleven_labs(text_chunk):
    """
    Sends a text chunk to the Eleven Labs TTS API asynchronously and returns the generated speech audio data 
    as a byte stream. It constructs the request using various predefined parameters and the provided text_chunk.
    
    Args:
    text_chunk (str): The text content to be converted into speech by the Eleven Labs TTS API.

    Returns:
    response_bytes (bytes): The audio data generated from the text_chunk by the Eleven Labs TTS API, returned as a byte stream.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    querystring = {"optimize_streaming_latency": STREAMING_LATENCY_VAL, "output_format": OUTPUT_FORMAT}

    payload = {
        "model_id": MODEL_ID,
        "text": text_chunk,
        "voice_settings": VOICE_SETTINGS
    }
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, params=querystring) as response:
            response_bytes = await response.read()
            return response_bytes  
     
################################################ SYNC FUNCTIONS #####################################################
def _send_text_chunk_to_eleven_labs(text_chunk):
        """
        Sends a text chunk to the Eleven Labs TTS API synchronously and returns the generated 
        speech response as a string. The function constructs the request using various predefined 
        parameters and the provided text_chunk.

        Args:
        text_chunk (str): The text content to be converted into speech by the Eleven Labs TTS API.

        Returns:
        response_text (str): The response from the Eleven Labs TTS API, typically the generated speech 
        data or a reference to it, returned as a string.
        """  
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
        querystring = {"optimize_streaming_latency":STREAMING_LATENCY_VAL,"output_format":OUTPUT_FORMAT}

        payload = {
            "model_id": MODEL_ID,
            "text": text_chunk,
            "voice_settings": VOICE_SETTINGS
        }
        headers = {
            "xi-api-key": ELEVEN_LABS_API_KEY,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", url, json=payload, headers=headers, params=querystring)

        return response.text

def add_to_list(item, my_list):
    """ Adds an item to a list with a maximum capacity of 10 items"""
    if len(my_list) >= 10:
        my_list.clear()  # Clear the list if it already has 10 items
    my_list.append(item)  # Add the new item

def convert_to_mulaw_with_ffmpeg(input_file_path):
    """ 
    Convert an audio file from `MP3` format to `MULAW`. This ensures the compatibility of the audio file to be played 
    through Twilio.

    Args:
    input_file_path : The path to the .mp3 file to be convreted

    Returns:
    output_file_path: The path of the converted .mp3 file, written as a .wav file
    """
    # Replace the file extension from .mp3 (or any other) to .wav
    base_file_path = os.path.splitext(input_file_path)[0]
    output_file_path = f"{base_file_path}.wav"

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print(f"Output file already exists: {output_file_path}")
        return output_file_path  # Return the existing file path

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the ffmpeg command to convert the audio
    command = [
        'ffmpeg',
        '-i', input_file_path,  # Input file
        '-ar', '8000',  # Set sample rate to 8000 Hz
        '-acodec', 'pcm_mulaw',  # Set audio codec to μ-law
        output_file_path  # Output file
    ]

    # Run the ffmpeg command
    subprocess.run(command, check=True)

    # Return the output file path for further use
    return output_file_path

def normalize_sentence(sentence):
    """ 
    Normalises a sentence by converting it to lowercase, removing punctuation and collapsing spaces
    
    Args:
    sentence : text input to be normalised

    Returns:
    This method returns the normalised sentence after conversion is complete 

    """
    cleaned_sentence = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', sentence.lower())).strip()
    return cleaned_sentence

def get_filler(label):
    """ 
    A helper method used to fetch the filler audio file corresponding to the output label from
    the filler prediction model

    Args:
    label : output label from the filler prediction model

    Returns:
    filler_path = the corresponding filler path as per the `value` assigned to the `label` in the 
    LABEL_TO_FILLER dictionary. It returns 'None' if the model the output lable from the model is None.
    
    """
    filler_path = LABEL_TO_FILLER.get(label)
    if filler_path:
        return filler_path
    else:
        return None
    
################################################ CUSTOM CLASSES #####################################################
class LRUCache:
    """
    A class that implements an LRU cache. It stores a limited amount of key-value pairs and ensures that the least 
    recently used item is discarded when the capacity is exceeded. It provides methods to get and put items 
    in the cache and to save and load the cache state to and from a file.
    """
    def __init__(self, cache_file_path, capacity=1000):
        """ Initializes a new instance of the LRUCache class. """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.cache_file_path = cache_file_path
        self.load_cache_from_file()

    def save_cache_to_file(self):
        """
        Saves the current state of the cache to a file specified by self.cache_file_path. 
        This method is called internally, typically when the cache is updated.
        Args: None.
        Returns: None.
        """
        try:
            with open(self.cache_file_path, 'wb') as file:
                pickle.dump(self.cache, file)
        except Exception as e:
            print(f"Error saving cache to file: {str(e)}")

    def load_cache_from_file(self):
        """
        Loads the cache's state from a file specified by self.cache_file_path. If the file does not exist, 
        it initializes an empty cache and notifies the user. This method is called during the initialization of the cache.
        Args: None.
        Returns: None.
        """
        try:
            with open(self.cache_file_path, 'rb') as file:
                self.cache = pickle.load(file)
        except FileNotFoundError:
            print("Cache file not found. Creating a new cache.")
        except Exception as e:
            print(f"Error loading cache from file: {str(e)}")
    
    def get(self, key):
        """
        Retrieves the value associated with a given key from the cache. If the key exists, it moves the key 
        to the end of the cache to mark it as recently used. If the key is a default message or not found, 
        it logs a warning or info message respectively and returns None.
        
        Args:
        key (str): The key for which to retrieve the corresponding value.
        
        Returns: The value associated with the given key if found; otherwise, None.
        """
        if key == DEFAULT_MESSAGE:
            logger.warning("Key is default message. Cannot retreive from cache.")
            return None
        key = normalize_sentence(key)    
        if key in self.cache:
            # Move the key to the end to indicate it was recently used
            logger.info("Key found in cache...")
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            logger.info("Key not found in cache...")
            return None  # If the key is not found, return None

    def put(self, key, value):
        """
        Adds a key-value pair to the cache. If the key is not a default message, it updates the cache with the key-value pair, 
        moves the key to the end to mark it as recently used, and saves the cache to a file. If adding the key-value pair 
        exceeds the cache's capacity, the least recently used item is removed from the cache. If the key is a default message, 
        it logs a warning and does not add it to the cache.
        
        Args:
        key (str): The key associated with the item to add to the cache.
        value: The value associated with the key to add to the cache.
        
        Returns: None.
        """
        if not key == DEFAULT_MESSAGE:
            key = normalize_sentence(key)
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            logger.info(f"Cache updated: Key='{key}' Value='{self.cache[key]}'")
            self.save_cache_to_file()  
        else:
            logger.warning("Key is default message. Not adding to cache.")


################################################ NOT USED ###########################################################
class CustomAsyncBufferedQueue: 
    """ Custom asyncio.Queue() class with an integrated audio data buffer """
    def __init__(self, max_buffer_size=4000):
        self.queue = asyncio.Queue()
        self.buffer = io.BytesIO()
        self.max_buffer_size = max_buffer_size
        self.end_of_stream_count = 0
        self.end_of_stream_enqueued = False
        self.item_available_event = asyncio.Event()
        self.trigger_end_signal_event = asyncio.Event()
    
    async def put_end_stream_signal(self):
        await self.queue.put(END_OF_STREAM_SIGNAL)
        self.end_of_stream_enqueued = True
        self.item_available_event.set()
        
    async def put(self, item):
        # Add regular items to buffer
        self.buffer.write(item)
        logger.info(f"Buffer size after writing: {self.buffer.tell()} bytes")

        # Process buffer and enqueue chunks when buffer exceeds max size
        while self.buffer.tell() >= self.max_buffer_size:
            self.buffer.seek(0)
            chunk_data = self.buffer.read(self.max_buffer_size)
            await self.queue.put(chunk_data)
            #self.items.append(item)
            self.item_available_event.set()
            if item == END_OF_STREAM_SIGNAL:
                self.end_of_stream_count += 1
                self.end_of_stream_enqueued = False
            logger.info(f"Chunk enqueued: {chunk_data[:10]}... (length: {len(chunk_data)} bytes)")

            # Handle remaining data
            remaining_data = self.buffer.read()
            self.buffer = io.BytesIO()
            self.buffer.write(remaining_data)
            logger.info(f"Remaining data in buffer: {remaining_data[:10]}... (length: {len(remaining_data)} bytes)")
    
    async def flush(self):
        """Enqueues any remaining data in the buffer."""
        remaining_data = self.buffer.getvalue()
        if remaining_data:
            await self.queue.put(remaining_data)
            self.item_available_event.set() # Signal that an item is available
            self.trigger_end_signal_event.set() # Signal that END_OF_STREAM_SIGNAL should be enqueued
            logger.info(f"Flushed remaining data: {remaining_data[:10]}... (length: {len(remaining_data)} bytes)")
            self.buffer = io.BytesIO()  # Reset the buffer
        await self.put_end_stream_signal()

    async def get(self):
        while self.queue.empty():
            logger.info("Queue is empty, waiting for data...")
            await self.item_available_event.wait()
            self.item_available_event.clear()

        item = await self.queue.get()
        if item == END_OF_STREAM_SIGNAL:
            self.end_of_stream_count -= 1
        return item
        
    def empty(self):
        return self.queue.empty()
    
    async def reset(self):
        # Resets the queue by creating a new empty buffer 
        # and emptying the queue by dequeuing all items.
        self.buffer = io.BytesIO()
        while not self.queue.empty():
            await self.queue.get()
    
    async def is_only_two_left(self):
        return self.end_of_stream_enqueued and self.queue.qsize() == 2
    
async def test_CustomAsyncBufferedQueue():
    """ Used to test the functionality of the `CustomAsyncBufferedQueue` class"""
    queue = CustomAsyncBufferedQueue(max_buffer_size=3)

    # Enqueue some audio data
    await queue.put(b"AudioData1")
    await queue.put(b"AudioData2")
    await queue.put(b"AudioData3")
    await queue.flush()  # This should also enqueue END_OF_STREAM_SIGNAL

    print("Completed enqueue operation...")

    # Dequeue and test
    while not queue.empty():
        print(f"Queue size before get: {queue.queue.qsize()}, is_only_two_left: {await queue.is_only_two_left()}")
        item = await queue.get()
        print(f"Dequeued: {item}, Queue size after get: {queue.queue.qsize()}")

 # Run the test
#asyncio.run(test_CustomAsyncBufferedQueue())
