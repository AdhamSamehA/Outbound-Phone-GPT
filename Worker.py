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
import json
import uuid
import logging
import base64
import asyncio
import time
import os
from typing import Optional, AsyncGenerator

# Import utils
from __utils__ import text_chunker, get_cached_streaming_generator, LRUCache

# Import configs
from __config__ import AGENT_CACHE_FILE

# Import sales agent framework (SalesGPT)
from ConversationModel.agents import ConversationalModel

# Import filler prediction model (FillerPredictor)
from FillerPredictionModel import GPTPredictor

# Import FastAPI libraries
from fastapi import WebSocket, WebSocketDisconnect

# Import libraries to handle eleven labs websocket
from websockets.legacy.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed

# Set up logging configuration
from logger_config import setup_logger
logger = setup_logger("my_app_logger", level=logging.DEBUG)

# AI Agent class 
class AIAgent():
    """
    AIAgent instances are workers specialised in processing streaming generators and audio streams
    """

    def __init__(self, call_sid=None):
        """
        Initializes a new AI agent instance.

        Args:
        call_sid (str): A string representing the Call SID (Session Identifier) that uniquely identifies the call session.
        """
        self.gpt : ConversationalModel = ConversationalModel().init_agent()
        self.model : GPTPredictor = GPTPredictor()

        self.call_sid : str = call_sid or str(uuid.uuid4())
        self.ai_response : str = ""
        self.websocket_url : str = ""
        self.stream_sid : str = ""
        self.welcome_file_path : str = ""
        self.filler_file_path : str = ""

        self.user_transcribed_input : str = ""
        self.transcripts = []

        self.lru_cache = LRUCache(capacity=1000, cache_file_path=AGENT_CACHE_FILE)

        self.streaming_generator_future: Optional[asyncio.Future]
        self.mark_event_future = Optional[asyncio.Future]

        self.post_audio_tasks = []
        self.responses = []

        self.streaming_generator: Optional[AsyncGenerator]

        self.audio_start_event = asyncio.Event()

        self.is_first_audio_chunk_sent : bool = True
        self.use_cache : bool = False
        self.stop_signal : bool = False

        self.response_latency_start : time 
        self.response_latency_end : time 
        self.streaming_gen_retrieval_start_time : time 
        self.streaming_gen_retrieval_end_time : time 
        self.eleven_labs_websocket_connect_start_time : time 
        self.eleven_labs_websocket_connect_end_time : time

################################################ TWILIO BI-DIRECTIONAL STREAM HANDLING #############################################
    async def connect_to_twilio_bidirectional_stream(self, websocket : WebSocket):
        """
        an async method responsible for connecting to Twilio's bidirectional stream in order to enable audio input/output streaming.

        Args:
        websocket: Our FastAPI server's websocket instance which is responsible for handling twilio's media streams

        Returns:
        None: This method does not return anything
        """
        try:
            twilio_to_websocket_connection_time_start = time.time()
            # Wait for the 'connected' message from Twilio
            connected_message = await websocket.receive_json()
            if connected_message.get('event') != 'connected':
                logger.error("connect_to_twilio_bidirectional_stream: Expected 'connected' message, received something else.")
                return

            logger.info("connect_to_twilio_bidirectional_stream: Connection between custom websocket and Twilio is successful...")

            # Wait for the 'start' message from Twilio
            start_message = await websocket.receive_json()
            if start_message.get('event') != 'start':
                logger.error("connect_to_twilio_bidirectional_stream: Expected 'start' message, received something else.")
                return
            logger.info("Twilio bi-directional media streaming started...")

            self.stream_sid = start_message.get('streamSid')
            logger.info("Stream SID Received...")

            twilio_to_websocket_connection_time_end = time.time()
            logger.info(f"Time taken to connect Twilio to custom websocket: {twilio_to_websocket_connection_time_end - twilio_to_websocket_connection_time_start} seconds")

        except WebSocketDisconnect:
            logger.error(f"connect_to_twilio_bidirectional_stream: Error -> Websocket Disconnected")
            #TODO: Implement better exception handling
        except Exception as e:
            logger.error(f"connect_to_twilio_bidirectional_stream: Exception -> {e}")

    async def send_mark_message(self, websocket_server : WebSocket):
            """
            Called when the entire audio output stream has been pushed to Twilio through the `websocket_server`. This will send
            a mark message to the Twilio and wait for a response to signal that the audio output stream finished playing on the call.

            Args:
            websocket_server: Our FastAPI server's websocket instance which is responsible for handling twilio's media streams

            Returns:
            None: This method does not return anything
            """
            mark_name = str(uuid.uuid4())
            logger.info("Sending mark message now...")
            await websocket_server.send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": mark_name}
            })
            logger.info("Waiting for mark message response...")
            while True:
                response = await websocket_server.receive_json()
                if response.get('event') == 'mark' and response.get('mark', {}).get('name') == mark_name:
                    logger.info(f"Mark event received for {mark_name}. Audio chunk has been transmitted successfully.")
                    break
                elif response.get('event') == 'stop':
                    logger.info("Call Ended...")
                    self.stop_signal = True
                    break

################################################ STREAM HANDLING ###################################################################
    async def send_text_stream(self, eleven_labs_websocket, cached_response=None):
        """
        Takes a new (is self_use_cache=False) or old (is self_use_cache=True) generated stream from the GPT model, filters out it's contents for compatibility with voice generation, and then streams it 
        asynchronously to the Eleven Labs API for converting it into audio.
        
        Args:
        cached_response : This is a list of words from a pre-generated GPT response. It is stored as a list to enable compatibility with the
        `get_cached_streaming_generator` method, which converts the list of words into a streaming generator.
        eleven_labs_websocket (WebSocket): The WebSocket connection to Eleven Labs API.

        Returns:
        None: This method does not return anything
        """
        async def output_iterator():
            if self.use_cache:
                logger.info("Sending pre-generated text stream to Eleven Labs API...")
                async for chunk in get_cached_streaming_generator(cached_response):
                    chunk_content = chunk["choices"][0]["delta"]["content"]
                    if chunk_content is not None:
                        self.ai_response += chunk_content
                        logger.info(f"Retrieving text chunk: {chunk_content}")
                        yield chunk_content
            else:
                logger.info("Sending newly generated text stream to Eleven Labs API...")
                async for chunk in self.streaming_generator:
                    chunk_content = chunk.choices[0].delta.content
                    if chunk_content is not None:
                        self.ai_response += chunk_content
                        logger.info(f"Retrieving text chunk: {chunk_content}")
                        #self.responses.append(chunk_content)  # Collect the content
                        yield chunk_content
                #self.lru_cache.put(user_input, self.responses)
                #self.responses.clear()
                        
                ###
    #           If you'd like to store newly generated streams automatically, then uncomment the logic for adding content into the LRU cache.
    #           Otherwise, you can leave it, and update the cache manulaly through the API (preferred method, as you can control what is stored)
                ### 
                
        try:
            async for text_chunk in text_chunker(output_iterator()):
                try:
                    logger.info("Sending text chunk...")
                    await eleven_labs_websocket.send(json.dumps({"text": text_chunk, "try_trigger_generation": True}))
                except ConnectionClosed as e:
                    logger.error("Eleven Labs WebSocket connection closed unexpectedly: {}".format(e))
                    break  # Exit loop if connection is closed
                except Exception as e:
                    logger.error(f"Exception in sending text chunk: {e}")

            # Now that the text stream has ended, send the final WebSocket message to Eleven Labs
            await eleven_labs_websocket.send(json.dumps({"text": ""}))
            logger.info("Final WebSocket message sent to Eleven Labs.")
        except Exception as e:
            logger.error(f"Exception in send_text_stream: {e}")    

    async def post_audio_to_websocket(self, audio_data, websocket_server : WebSocket):
            """
            Posts the audio stream to the agent's WebSocket server.

            This method is responsible for streaming the audio data, received from 
            Eleven Labs, to the client via the agent's WebSocket server.

            Args:
            audio_stream (AsyncGenerator): An asynchronous generator yielding audio chunks.

            Returns:
            None: This method streams audio data and does not return anything.
            """               
            logger.info("Sending audio data now...")
            await websocket_server.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(audio_data).decode('utf-8')}
            })
            logger.info("Audio data sent...")

            if self.is_first_audio_chunk_sent == False:
                self.response_latency_end = time.time()
                logger.info(f"Agent Utterance Response Latency ~ {self.response_latency_end - self.response_latency_start} seconds")
                self.is_first_audio_chunk_sent = True

    async def get_streaming_generator(self):
        """
        Utilises `self.streaming_generator_future` which is an `asyncio.Future` instance. The method
        is responsible for handling the retrieval of the streaming generator from the instance. 

        The future instance is used to mimimise the time taken to wait for the completion of the streamimg generator in order to reduce 
        response latency.
        
        Arguments:
        None: This method doesn't take in any arguments

        Returns:
        None: This method doesn't return anything
        """
        if self.streaming_generator_future:
            logger.info("Verified streaming generator future coroutine...")
            # Check if the future is already done
            if self.streaming_generator_future.done():
                # Get the result directly without awaiting
                self.streaming_generator = self.streaming_generator_future.result()
                logger.info(f"Streaming generator created and retrieved in: {self.streaming_gen_retrieval_end_time-self.streaming_gen_retrieval_start_time} seconds") 
            else:
                # If the future is not done, await it
                logger.info("Awaiting streaming generator...")
                await_streaming_gen_start = time.time()
                self.streaming_generator = await self.streaming_generator_future
                await_streaming_gen_end = time.time()
                logger.info("Streaming generator awaited successfully...")
                logger.info(f"Streaming generator awaited for: {await_streaming_gen_end - await_streaming_gen_start} seconds")

            self.streaming_gen_retrieval_end_time = time.time()
            logger.info(f"Streaming generator created and retrieved in: {self.streaming_gen_retrieval_end_time-self.streaming_gen_retrieval_start_time} seconds")
        
################################################ INPUT PROCESSING ##################################################################     
    async def process_input(self, cached_response, eleven_labs_websocket : WebSocketClientProtocol, websocket_server : WebSocket, user_input : str):
        """
        Processing the text and audio stream asynchronously to ensure that

        Args:
        eleven_labs_websocket (WebSocket): The WebSocket connection to Eleven Labs API.
        websocket (WebSocket): The active WebSocket connection for audio streaming.
        user_input (str): The user input text to be processed.
        cached_response: list of words from a pre-generated GPT stream, retrived from the agent's cache

        Returns:
        None: This method does not return anything but sends data over a websocket.
        """ 
        try:
            if eleven_labs_websocket: 
                async def listen_eleven_labs():
                    """
                    This is the primary handler of the speech to text response stream generated by Eleven Labs API. 
                    It calls `post_audio_to_websocket` as soon as a valid audio chunk has been recieved then adds that to a list of 
                    `post_audio_future` instances. This technique replaces the standard buffer to enable taking full advantage of asynchronous programming
                    by pushing data to our websocket server as soon as its available.

                    It then waits for the mark message to be recieved from Twilio, to signal successful playback of the voice generated.
                    
                    """
                    try:
                        while True:
                            message = await eleven_labs_websocket.recv()
                            data = json.loads(message)
                            if data.get("audio"):
                                audio_chunk = base64.b64decode(data["audio"])
                                if audio_chunk is not None:
                                    logger.info(f"listen_eleven_labs: Audio data : {audio_chunk[:10]} - Length: {len(audio_chunk)} bytes")
                                    post_audio_future = asyncio.ensure_future(self.post_audio_to_websocket(audio_chunk, websocket_server))
                                    self.post_audio_tasks.append(post_audio_future)
                                else:
                                    logger.warning("listen_eleven_labs: Recieved empty audio chunk")
                                    continue

                            elif data.get('isFinal'):
                                await asyncio.gather(*self.post_audio_tasks)
                                await self.send_mark_message(websocket_server)
                                break
                    except Exception as e:
                        logger.error(f"listen_eleven_labs: Exception-> {e}")
                
                if self.use_cache:
                    logger.info(f"Retrieved from cache for '{user_input}': {cached_response}") 
                else:
                    await self.get_streaming_generator()
                    
                listen_task = asyncio.create_task(listen_eleven_labs())
                text_stream_task = asyncio.create_task(self.send_text_stream(eleven_labs_websocket=eleven_labs_websocket, cached_response=cached_response))

                await asyncio.gather(text_stream_task, listen_task)

                self.mark_event_future = asyncio.ensure_future(self.send_mark_message(websocket_server))

            else:
                logger.error(f"Error with eleven labs websocket. Type: {type(eleven_labs_websocket)}")
                raise WebSocketDisconnect
        except Exception as e:
            logger.error(f"Exception in process_input: {e}")

################################################ UPDATING AGENT FOR CONVERSATION FLOW ###############################################
    def update_agent_response(self):
        """
        Updates the agent's response after processing user input.

        Args:
        None: This method doesn't take in any arguments

        Returns:
        None: This method updates internal state and does not return anything.
        """
        logger.info(f"Most recent response: {self.ai_response}")
        logger.info("Updating Agent...")
        agent_name = self.gpt.get_attribute('agent_name')
        self.ai_response = f"{agent_name}: {self.ai_response.strip()}"
        if "<END_OF_TURN>" not in self.ai_response:
            self.ai_response += " <END_OF_TURN>"
        self.gpt.conversation_history.append(self.ai_response)
        logger.info("Agent Update Complete...")
        logger.info(self.ai_response.replace("<END_OF_TURN>", ""))
        logger.info(f"Conversation History: {self.gpt.conversation_history}")
        self.ai_response=""

################################################ RESETING AGENT FOR NEXT CONVERSATION ###############################################
    async def reset_after_interaction(self):
        """Resets the agent's state after each interaction round"""
        self.audio_start_event.clear()  # Reset the event for the next interaction
        self.is_first_audio_chunk_sent = False
        self.post_audio_tasks.clear()
        self.use_cache = False
        # Reset other state variables or events as necessary