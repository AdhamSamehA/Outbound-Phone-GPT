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

import logging
import asyncio
import websockets
import uvicorn
import time
import base64
import json
import uuid
import pickle
from typing import List, Tuple

# Import utils.py
from __utils__ import add_to_list, generate_audio_file

# Import config.py
from __config__ import ACCOUNT_SID, AUTH_TOKEN, TWILIO_NUM, HTTP_SERVER_PORT, WEBSOCKET_SUBDOMAIN, BASE_WEBSOCKET_URL, SECRET_KEY, LOCAL_HOST, AGENT_AUDIO_FOLDER, DEEPGRAM_URI, HEADERS, ELEVEN_LABS_URI, VOICE_SETTINGS, ELEVEN_LABS_API_KEY, DEFAULT_MESSAGE, AGENT_CACHE_FILE, LABEL_TO_FILLER

# Import FastAPI libraries
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# Import Twilio libraries
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# Import session management middleware
from starlette.middleware.sessions import SessionMiddleware

# Set up logging configuration
from logger_config import setup_logger
logger = setup_logger("my_app_logger", level=logging.INFO)

# Import AIAgent class for phone handling
from Worker import AIAgent

# Pydantic base classes:
from pydantic import BaseModel

class CallRequest(BaseModel): 
    """Handles JSON data sent along with the `/make-call` POST request """
    welcome_message: str = ""

class SentenceFiller(BaseModel):
    """
    Handles requests to generate pre-set sentence fillers through the `/generate-filler` POST request.
    
    Example JSON for the POST request: (`filename` should be provided without an extension. This is internally handled.)
        {
    "fillers": [
        ["Filler text 1", "file_name_1"],
        ["Filler text 2", "file_name_2"],
        ["Filler text 3", "file_name_3"]
    ]
    }
    """
    fillers: List[Tuple[str, str]]  # Each item in the list is a tuple with (filler text, file name)

class KeyValueInput(BaseModel):
    """
    Handles requests manually generate and add pre-defined question and answer pairs to the agent's cache
    
    Example JSON for the POST request:
    [
        {
            "key": "Hi",
            "value": "Hey, how is it going?"
            },
        {
            "key": "Who are you?",
            "value": "I'm Myra, the recruitement director at Escade Networks."
            },
        {
            "key" : "Why are you calling?",
            "value": "I'm calling to ask you some questions regarding your application for one of our job openings. This is a quick, preliminary routine which helps us qualify prospects before scheduling interviews. Is that fine with you?"
            }
    ]
    """
    key: str
    value: str

################################################ GLOBAL VARIABLES #####################################################
agents = {} # Dictionary to store agent instances and their associated call sids
call_sids = [None] # Indexed list to store call_sids

################################################ HELPER FUNCTIONS #####################################################
def get_agent(call_sid: str) -> AIAgent:
    """
    Function to retrieve or create an AIAgent instance using the provided call identification number

    Args:
    call_sid : Unique identifier for calls

    Returns:
    None: This method' doesn't return anything
    """
    agent_retrieval_start_time = time.time()
    if call_sid not in agents:
        agents[call_sid] = AIAgent(call_sid)
        logger.info(f"New AI Agent Initialized As Previous Instance with - Call SID: {call_sid} - wasn't found\n")
    logger.info("Agent Found. Retrieving agent now...")
    agent_retrieval_end_time = time.time()
    logger.info(f"AIAgent retrieved in: {agent_retrieval_end_time-agent_retrieval_start_time} seconds")
    return agents[call_sid]

def reset_for_next_call():
    """ Resets agent configurations so that the same agent can take another call"""
    global is_first_transcript_recieved, is_first_eleven_labs_connection
    is_first_transcript_recieved = False # Default state
    is_first_eleven_labs_connection = True # Default state
    logger.info("Session variables have been reset...")

################################################ INITIALISE THE APP ###################################################
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

################################################ SESSION VARIABLES ####################################################
call_count : int = 0 # Keep track of call_count
is_first_eleven_labs_connection : bool = True # Default state

################################################ WEBSOCKET ############################################################
@app.websocket("/audiostream/{call_sid}")
async def audio_stream(websocket: WebSocket, call_sid: str):
    """
    Handles the WebSocket connection for streaming audio data. It manages the entire lifecycle of the call.

    Args:
    websocket: Our FastAPI server's websocket instance responsible for handling Twilio's bidirectional media streams
    call_sid: A string representing the Call SID (Session Identifier) that uniquely identifies the call session.
    
    Returns:
    This function does not return a value but operates asynchronously to manage the WebSocket connection and audio stream processing.
    
    """
    global call_count, is_first_eleven_labs_connection

    connection_start_time = time.time()
    await websocket.accept()
    connection_end_time = time.time()
    logger.info(f"Custom websocket connection time: {(connection_end_time - connection_start_time)} seconds")
    
    if call_sid is None:
        return {"error": "Call SID not found in session"}
    logger.info(f"Successfully retrieved call SID")
    agent = get_agent(call_sid)  # Retrieve or create an AIAgent instance

    await agent.connect_to_twilio_bidirectional_stream(websocket)

    async def play_welcome_message():
        """ Streams a welcome message audio file to the WebSocket connection. """
        chunk_size = 1024  # Adjust based on your needs
        with open(agent.welcome_file_path, 'rb') as audio_file:
            while audio_chunk := audio_file.read(chunk_size):
                if audio_chunk is not None:
                    logger.info(f"Audio data : {audio_chunk[:10]} - Length: {len(audio_chunk)} bytes")
                    post_audio_task = asyncio.ensure_future(agent.post_audio_to_websocket(audio_chunk, websocket_server=websocket))
                    agent.post_audio_tasks.append(post_audio_task)
                else:
                    logger.warning("listen_eleven_labs: Recieved empty audio chunk")
                    continue
        await asyncio.gather(*agent.post_audio_tasks)
        await agent.send_mark_message(websocket_server=websocket)
    
    welcome_message_task = asyncio.create_task(play_welcome_message())
    logger.info("Welcome message task has been created...")

    # Variables to keep track of 
    close_message = json.dumps({"type": "CloseStream"})
    last_transcript_chunk = ""
    user_transcribed_input = ""
    connect_deepgram_ws = False

    try:
        async with websockets.connect(uri=DEEPGRAM_URI, extra_headers=HEADERS) as deepgram_ws:
            logger.info("Deepgram websocket connection has been established...")
            async with websockets.connect(uri=ELEVEN_LABS_URI) as eleven_labs_ws:
                logger.info("Eleven Labs websocket connection has been established...")
                async def receive_deepgram_transcripts():
                    """ Receives and processes transcripts from the Deepgram speech-to-text service. """
                    nonlocal last_transcript_chunk
                    logger.info("Entered receive_deepgram_transcripts")
                    try:
                        #async for message in deepgram_ws:
                        while True:  # Keep listening for new messages
                            message = await deepgram_ws.recv()  # Wait for the next message. The message should be a string.
                            if message is None:
                                continue
                            logger.info(f"Deepgram message received: {message}")
                            data = json.loads(message)
                            if data.get('type') == 'SpeechStarted':
                                logger.info("Speech started signal received...")
                            if data.get('type') == 'UtteranceEnd':
                                logger.info("End of speech signal received...")
                                sender_task.cancel()
                                logger.info("Signaling Deepgram websocket to close...")
                                try:
                                    await sender_task
                                except asyncio.CancelledError:
                                    logger.info("sender_task was cancelled")
                                except Exception:
                                    logger.info(f"Skipping exception: {e}")
                                finally:
                                    #await deepgram_ws.close()
                                    break  # Exit the coroutine after handling UtteranceEnd
                            elif data.get("type") == "Results" and data.get("is_final"):
                                if data["channel"]["alternatives"][0]["confidence"] > 0.8:
                                    transcript = data["channel"]["alternatives"][0]["transcript"]
                                    logger.info("Recieved transcript from Deepgram websocket...")
                                    #if transcript.strip():
                                    agent.transcripts.append(transcript)
                                    logger.info(f"Transcribed input: {transcript}")
                                else:
                                    logger.info(f"Confidence score of {data['channel']['alternatives'][0]['confidence']} is too low...")
                    except asyncio.CancelledError:
                        logger.info("receive_deepgram_transcripts task was cancelled as signaled by Deepgram VAD")
                    except Exception as e:
                        logger.error(f"Error in receive_deepgram_transcripts: {e}")
                        await deepgram_ws.send(close_message)  # Close the WebSocket in case of an exception
                        raise Exception

                async def forward_audio_to_deepgram():
                    """ Forwards incoming audio data from the WebSocket connection to the Deepgram service for speech-to-text conversion. """
                    logger.info("Entered forward_audio_to_deepgram")
                    buffer = bytearray()
                    buffer_size = 20 * 160  # Buffer size in bytes
                    try:
                        #async for message in client_ws:
                        while True:
                            message = await websocket.receive() # Wait for the next message. The message should be a dictionary.
                            if message is None:
                                continue
                            twilio_json = message['text']
                            data = json.loads(twilio_json)
                            event = data['event']
                            if event == 'media':
                                chunk = base64.b64decode(data['media']['payload'])
                                buffer.extend(chunk)
                                # If the buffer reaches the specified size, send it to Deepgram
                                while len(buffer) >= buffer_size:
                                    # Send the first buffer_size bytes to Deepgram
                                    await deepgram_ws.send(buffer[:buffer_size])
                                    # Remove the sent bytes from the buffer
                                    buffer = buffer[buffer_size:]

                            elif event == 'stop':
                                # If there's any remaining audio in the buffer, send it
                                if buffer:
                                    await deepgram_ws.send(buffer)
                                    buffer.clear()
                                # Close the WebSocket connection as the stream has ended
                                await websocket.close()
                                #await deepgram_ws.send(close_message)
                                await deepgram_ws.close()
                                agent.stop_signal = True
                                break  # Exit the loop as the stream has ended
                            else:
                                logger.error("Can't parse the message...")
                                continue
                    
                    except Exception as e:
                        logger.error(f"Error processing audio stream: {e}")
                        await websocket.close()
                        await deepgram_ws.send(close_message)
                        agent.stop_signal = True
                
                async def initialise_eleven_labs_websocket():
                    """ Initializes the connection with Eleven Labs's websocket """
                    nonlocal eleven_labs_ws
                    if eleven_labs_ws.closed:
                        logger.warning("Eleven Labs websocket connection is closed...")
                        eleven_labs_ws = await websockets.connect(ELEVEN_LABS_URI)
                    
                    await eleven_labs_ws.send(json.dumps({
                        "text": " ",
                        "voice_settings": VOICE_SETTINGS,
                        "xi_api_key": ELEVEN_LABS_API_KEY,
                    }))
                    logger.info("Initiation message sent to Eleven Labs...")
                    
                    return eleven_labs_ws
                
                async def initialise_deepgram_ws():
                    """ Initializes the connection with Deepgram's websocket"""
                    nonlocal deepgram_ws
                    if deepgram_ws.closed:
                        logger.warning("Eleven Labs websocket connection is closed...")
                        deepgram_ws = await websockets.connect(uri=DEEPGRAM_URI, extra_headers=HEADERS)
                    return deepgram_ws

                async def play_filler():
                        """ Streams a filler audio file to the WebSocket connection, used when waiting for user input or processing delays."""
                        chunk_size = 1024  # Adjust based on your needs
                        with open(agent.filler_file_path, 'rb') as audio_file:
                            while audio_chunk := audio_file.read(chunk_size):
                                if audio_chunk is not None:
                                    logger.info(f"Audio data : {audio_chunk[:10]} - Length: {len(audio_chunk)} bytes")
                                    post_audio_task = asyncio.ensure_future(agent.post_audio_to_websocket(audio_chunk, websocket_server=websocket))
                                    agent.post_audio_tasks.append(post_audio_task)
                                else:
                                    logger.warning("listen_eleven_labs: Recieved empty audio chunk")
                                    continue
                        await asyncio.gather(*agent.post_audio_tasks)
                        await agent.send_mark_message(websocket_server=websocket)

                if is_first_eleven_labs_connection:
                    await welcome_message_task
                    logger.info("Welcome message has played...")
                    
                eleven_labs_ws = await initialise_eleven_labs_websocket()

                while True:
                    if connect_deepgram_ws:
                        deepgram_ws = await deepgram_connection_task
                    user_transcribed_input = ""
                    logger.info("Transcribing...")
                    try:
                        receiver_task = asyncio.create_task(receive_deepgram_transcripts())
                        sender_task = asyncio.create_task(forward_audio_to_deepgram())
                        await asyncio.gather(receiver_task, sender_task)
                        is_first_eleven_labs_connection = False
                    except asyncio.CancelledError:
                        logger.info("A task was cancelled upon EOS detection")
                    except WebSocketDisconnect:
                        logger.error(f"Error in WebSocket: Websocket Disconnected")
                        #TODO: Handle WebSocket disconnection (e.g., clean up, logging) -> NOT YET IMPLEMENTED
                    except Exception as e:
                        logger.error(f"An error occurred: {e}")
                        agent.stop_signal = True
                    finally:
                    # Ensure any necessary cleanup happens here
                        if not sender_task.done():
                            sender_task.cancel()
                            try:
                                await sender_task
                            except asyncio.CancelledError:
                                logger.info("sender_task cleanup cancellation")
                        if not receiver_task.done():
                            receiver_task.cancel()
                            try:
                                await receiver_task
                            except asyncio.CancelledError:
                                logger.info("receiver_task cleanup cancellation")

                        logger.info("Transcription completed...")

                        
                    user_transcribed_input = " ".join(agent.transcripts) or DEFAULT_MESSAGE
                    logger.info(f'Complete transcription collected: {user_transcribed_input}')
                   
                    # Handling speech generation
                    logger.info("Responding...")
 
                    eleven_labs_connection_task = asyncio.create_task(initialise_eleven_labs_websocket())

                    agent.is_first_audio_chunk_sent = False
                    agent.response_latency_start = time.time()

                    ### METHOD 1: Agent cache dismissed ###

                    try:
                        agent.transcripts.clear()
                        human_step_start = time.time()
                        agent.gpt.human_step(user_transcribed_input)
                        human_step_end = time.time()
                        logger.info(f"`human_step` took: {human_step_end-human_step_start} seconds")
                        
                        try:
                            agent.streaming_gen_retrieval_start_time = time.time()
                            agent.streaming_generator_future = asyncio.ensure_future(agent.gpt._astreaming_generator())
                            eleven_labs_ws = await eleven_labs_connection_task
                            agent.use_cache = False
                            await agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=None)
                        except WebSocketDisconnect:
                            logger.error(f"Error in WebSocket: Websocket Disconnected")
                            raise WebSocketDisconnect
                        except Exception as e:
                            logger.error(f"Error occured while processing input: {e}")
                            raise Exception
                            #TODO: Handle Exception -> NOT YET IMPLEMENTED
                            
                    except WebSocketDisconnect:
                        logger.error(f"Error in WebSocket: Websocket Disconnected")
                        raise WebSocketDisconnect
                        #TODO: Handle WebSocket disconnection (e.g., clean up, logging) -> NOT YET IMPLEMENTED
                    except Exception as e:
                        logger.error(f"Error in WebSocket: {e}")
                        await websocket.close()
                        await deepgram_ws.send(close_message)
                        agent.stop_signal = True
                    
                    ### Method 2: Utilising Agent cache ###
                        
#                     try:
#                        get_cache_start = time.time()
#                        cached_response = agent.lru_cache.get(user_transcribed_input)
#                        get_cache_end = time.time()
#                        logger.info(f"Time to scan through cache: {get_cache_end-get_cache_start} seconds")
#                        agent.transcripts.clear()
#                        human_step_start = time.time()
#                        agent.gpt.human_step(user_transcribed_input)
#                        human_step_end = time.time()
#                        logger.info(f"`human_step` took: {human_step_end-human_step_start} seconds")
#                        
#                        # Implement code here to check if we need to use a cached response or generate a new response 
#                        if cached_response:
#                            logger.info(f"Cache hit for input: {user_transcribed_input}")
#                            agent.use_cache = True
#                            try:
#                                eleven_labs_ws = await eleven_labs_connection_task
#                                await agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=cached_response)
#                            except Exception as e:
#                                logger.error(f"Error occured while processing input: {e}")
#                                raise Exception
#                                #TODO: Handle Exception -> NOT YET IMPLEMENTED
#                        else:
#                            logger.info(f"Cache miss for input: {user_transcribed_input}. Generating response...")
#                            agent.use_cache = False
#                            agent.streaming_gen_retrieval_start_time = time.time()
#                            agent.streaming_generator_future = asyncio.ensure_future(agent.gpt.astep(stream=True))
#                            try:
#                                eleven_labs_ws = await eleven_labs_connection_task                          
#                                await agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=cached_response)
#                            except Exception as e:
#                               logger.error(f"Error occured while processing input: {e}")
#                                raise Exception
#                                #TODO: Handle Exception -> NOT YET IMPLEMENTED
#                            
#                    except WebSocketDisconnect:
#                        logger.error(f"Error in WebSocket: Websocket Disconnected")
#                        raise WebSocketDisconnect
#                        #TODO: Handle WebSocket disconnection (e.g., clean up, logging) -> NOT YET IMPLEMENTED
#                    except Exception as e:
#                        logger.error(f"Error in WebSocket: {e}")
#                        await websocket.close()
#                        await deepgram_ws.send(close_message)
#                        agent.stop_signal = True
#
                    ### Method 3: Utilise Agent cache in addition to a filler mechanism to minimise response latency ###
                    # NOTE: The filler prediction method is incomplete and can add approx. 1 second to the response time latency if it return's None. In the future, I might integrate a text classification model with fast inference using OpenVino.

#                    try:
#                        get_cache_start = time.time()
#                        cached_response = agent.lru_cache.get(user_transcribed_input)
#                        get_cache_end = time.time()
#                        logger.info(f"Time to scan through cache: {get_cache_end-get_cache_start} seconds")
#                        agent.transcripts.clear()
#                        human_step_start = time.time()
#                        agent.gpt.human_step(user_transcribed_input)
#                        human_step_end = time.time()
#                        logger.info(f"`human_step` took: {human_step_end-human_step_start} seconds")
                        
#                        if cached_response:
#                            logger.info(f"Cache hit for input: {user_transcribed_input}")
#                            agent.use_cache = True
#                            try:
#                                eleven_labs_ws = await eleven_labs_connection_task
#                                await agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=cached_response)
#                            except Exception as e:
#                                logger.error(f"Error occured while processing input: {e}")
#                                raise Exception
#                                #TODO: Handle Exception -> NOT YET IMPLEMENTED
#                        else:
#                            logger.info(f"Cache miss for input: {user_transcribed_input}. Generating response...")
#                            agent.use_cache = False
#                            agent.streaming_gen_retrieval_start_time = time.time()
#                            agent.streaming_generator_future = asyncio.ensure_future(agent.gpt.astep(stream=True))
#                            try:
#                                intent_classifier_start = time.time()
#                                agent.filler_file_path = agent.model.classify_intent(user_transcribed_input)
#                                intent_classifier_end = time.time()
#                                logger.info(f"Intent Classification Result: {agent.filler_file_path}. Result obtained in {intent_classifier_end-intent_classifier_start} seconds")
#                                eleven_labs_ws = await eleven_labs_connection_task
#                                if agent.filler_file_path:
#                                    filler_task = asyncio.create_task(play_filler())
#                                    processing_task = asyncio.create_task(agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=cached_response))
#                                    await asyncio.gather(filler_task, processing_task)
#
#                                else:
#                                    await agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=cached_response)
#                                #await agent.process_input(eleven_labs_websocket=eleven_labs_ws, websocket_server=websocket, user_input=user_transcribed_input, cached_response=cached_response)
#                            except Exception as e:
#                                logger.error(f"Error occured while processing input: {e}")
#                                raise Exception
#                                #TODO: Handle Exception -> NOT YET IMPLEMENTED
#                            
#                    except WebSocketDisconnect:
#                        logger.error(f"Error in WebSocket: Websocket Disconnected")
#                        raise WebSocketDisconnect
#                        #TODO: Handle WebSocket disconnection (e.g., clean up, logging) -> NOT YET IMPLEMENTED
#                    except Exception as e:
#                        logger.error(f"Error in WebSocket: {e}")
#                        await websocket.close()
#                        await deepgram_ws.send(close_message)
#                        agent.stop_signal = True
                
                   # After all tasks are completed, update agent response and reset queue manager for next conversation
                    deepgram_connection_task = asyncio.create_task(initialise_deepgram_ws())
                    await agent.mark_event_future
                    update_agent_response_start_time = time.time()
                    agent.update_agent_response()
                    update_agent_response_end_time = time.time()
                    logger.info(f"Updated agent in: {update_agent_response_end_time-update_agent_response_start_time} seconds")
                    reset_interaction_start_time = time.time()
                    await agent.reset_after_interaction()
                    reset_interaction_end_time = time.time()
                    logger.info(f"Reset interaction for next conversation: {reset_interaction_end_time-reset_interaction_start_time} seconds")
                    connect_deepgram_ws = True
                    
                    if agent.stop_signal:
                        reset_for_next_call()
                        logger.info("Stop signal recieved. Stopping...")
                        break  

    except WebSocketDisconnect:
        logger.error(f"Error in WebSocket: Websocket Disconnected")
        #TODO: Handle WebSocket disconnection (e.g., clean up, logging) -> NOT YET IMPLEMENTED
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
    finally:
        reset_for_next_call()
        logger.info("Closing WebSocket connection...")
        disconnection_start_time = time.time()
        await websocket.close()
        disconnection_end_time = time.time()
        logger.info("Websocket Connection time: " + str(disconnection_end_time - disconnection_start_time))

################################################ POST REQUESTS ########################################################
@app.post('/make-call')
async def make_call(call_request: CallRequest):
    global call_count
    """
    Initiates a phone call using Twilio and sets up the necessary session and agent.

    This endpoint triggers a new phone call, stores the call SID in the session,
    and initializes a new AI agent associated with the call.

    Args:
    call_request (CallRequest): Responsble for recieving call data that is to be sent with the POST request as in JSON format. 
    Currently it only supports 'welcome_message' (See the CallRequest class declaration above) but this is highly customisable as per your need.

    Returns:
    JSON: A JSON object containing the call SID.
    """
    start_time = time.time()
    twml = VoiceResponse()
    
    # Initialise agent and websocket connection
    ai_agent = AIAgent()
    add_to_list(ai_agent.call_sid, call_sids)
    logger.info("Call SID saved as a session variable")
    agents[ai_agent.call_sid] = ai_agent
    ai_agent.websocket_url = f"wss://{WEBSOCKET_SUBDOMAIN}/audiostream/{ai_agent.call_sid}"

    welcome_message = call_request.welcome_message
    if welcome_message != "":
        ai_agent.ai_response = welcome_message
        ai_agent.update_agent_response()
        logger.info(f"Received welcome message: {welcome_message}")
        try:
            ai_agent.welcome_file_path = await generate_audio_file(welcome_message, file_name=f'{str(uuid.uuid4())}', type='starter')
            logger.info(f"Recieved file : {ai_agent.welcome_file_path}")
        except IOError as e:
            logger.error(f"IOError: {e}\n")
        except KeyError as key_error:
            logger.error(f"KeyError: {key_error}\n")
        except Exception as e:
            logger.error(f"Error: {str(e)}\n")
    else:
        welcome_message = "Hello, this is Myra calling from Escade Networks, how are you?"
        ai_agent.ai_response = welcome_message
        ai_agent.update_agent_response()
        logger.info("Playing default intro message...")
        ai_agent.welcome_file_path = await generate_audio_file(welcome_message, file_name='default', type='starter')
    
    connect = Connect()
    connect.stream(url=ai_agent.websocket_url)
    twml.append(connect)

    start_xml = str(twml.to_xml())
    logger.info(f"start_xml : {start_xml}")

    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    try:
        call = client.calls.create(
            twiml= start_xml,
            to='+971547055538',
            from_=TWILIO_NUM
        )
        call_count += 1
        logger.info(f"Call Count: {call_count}")
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution time for /make-call: {execution_time} seconds\n")
        return JSONResponse(content={"call_sid": call.sid})
    except TwilioRestException as e:
        logger.error(f"TwilioRestException: {e}\n")
        return f"TwilioRestException: {e}"
    except Exception as ex:
        logger.error(f"Error: {ex}\n")
        return f"Error: {ex}"

@app.post('/generate-filler')
async def generate_filler(sentence_filler: SentenceFiller):
    """
    Processes a list of sentence fillers and generates corresponding audio files.

    Args:
    sentence_filler: An instance of SentenceFiller, which includes a list of tuples. 
    Each tuple contains a filler sentence (filler) and a corresponding file name (file_name).
    
    Returns:
    A JSON response with a status indicating the success of the operation and a message providing a summary. 
    """
    for filler, file_name in sentence_filler.fillers:
        logger.info(f"Received sentence filler: {filler}")
        logger.info(f"Received file name: {file_name}")

        if not filler.strip():  # Check if filler is not empty or just whitespace
            logger.error("Empty or invalid sentence filler received.")
            continue  # Skip this filler and move to the next

        try:
            audio_file = await generate_audio_file(message=filler, file_name=file_name, type='filler')
            if audio_file:
                logger.info(f"Successfully generated audio file for filler: {filler}")
            else:
                logger.error(f"Failed to generate the audio file for filler: {filler}")
        except Exception as e:
            logger.error(f"An error occurred while generating the audio file for filler: {filler}, error: {e}")

    return {"status": "Success", "message": "Processed all fillers"}

@app.post("/update-cache")
async def update_cache(kv_data_list: list[KeyValueInput]):
    """
    Updates a cache with new key-value pairs provided in the request.

    Args:
    kv_data_list: A list of KeyValueInput objects, where each object contains a key and a value. 
    The value is expected to be a string that will be split into a list before being stored in the cache.
   
    Returns:
    A JSON response indicating the success of the cache update operation and includes the new cache state. 
    """
    try:
        # Load the existing cache from the .pkl file
        with open(AGENT_CACHE_FILE, 'rb') as file:
            cache = pickle.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty cache
        cache = {}

    # Iterate through the list of key-value pairs in the JSON input
    for kv_data in kv_data_list:
        # Split the value into a list and assign it to the key in the cache
        value_list = kv_data.value.split()
        cache[kv_data.key] = value_list

    # Save the updated cache back to the .pkl file
    with open(AGENT_CACHE_FILE, 'wb') as file:
        pickle.dump(cache, file)

    return {"message": "Cache updated successfully", "new_cache": cache}


if __name__ == '__main__':   
    logger.info(f"Server listening on: {LOCAL_HOST}")
    logger.info(f"WebSocket server active at: {BASE_WEBSOCKET_URL}/audiostream")
    uvicorn.run(app, host='0.0.0.0', port=HTTP_SERVER_PORT)
