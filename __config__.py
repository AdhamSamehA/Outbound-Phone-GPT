import os
from dotenv import load_dotenv

# Load `.env` content file
load_dotenv()

# Set up the directory paths within your project
AGENT_CONFIGS_FOLDER = f'{os.getcwd()}/example_agent_configs'
AGENT_CACHE_FOLDER = f'{os.getcwd()}/example_cache_files'
AGENT_AUDIO_FOLDER = f"{os.getcwd()}/example_audio_files" 

# Configure Twilio
ACCOUNT_SID : str = os.getenv('TWILIO_ACCOUNT_SID')
AUTH_TOKEN : str = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_NUM : str = os.getenv('TWILIO_PHONE_NUMBER')

# Configure Open AI:
OPENAI_API_KEY : str = os.getenv('OPENAI_API_KEY')
MAX_TOKENS : int = 100
BASE_GPT_TURBO_MODEL : str = "gpt-3.5-turbo-0125"

# Configure Eleven Labs
ELEVEN_LABS_API_KEY : str = os.getenv('ELEVEN_LABS_API_KEY')
VOICE_ID : str = os.getenv('ELEVEN_LABS_VOICE_ID')
MODEL_ID : str = os.getenv('ELEVEN_LABS_TURBO_MODEL_ID')
STREAMING_LATENCY_VAL : str = '4'
ENABLE_SSML_PARSE : bool = True
VOICE_SETTINGS : dict = {"stability": 0.71, "similarity_boost": 0.5}
OUTPUT_FORMAT : str  = 'ulaw_8000'
ELEVEN_LABS_URI : str = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id={MODEL_ID}&enable_ssml_parsing={ENABLE_SSML_PARSE}&optimize_streaming_latency={STREAMING_LATENCY_VAL}&output_format={OUTPUT_FORMAT}"
END_OF_STREAM_SIGNAL = b"END"

# Configure Deepgram
DEEPGRAM_API_KEY : str = os.getenv('DEEPGRAM_API_KEY')
DEEPGRAM_MODEL : str = os.getenv("DEEPGRAM_MODEL_ID")
VERSION : str = "latest"
LANGUAGE : str = "en-US"
PUNCTUATE : str = "true"
INTERIM_RESULTS : str = "true"
ENDPOINTING : str = "true"
UTTERANCE_END_MS : str = "1000"
VAD_EVENTS : str = "true"
ENCODING : str = "mulaw"
SAMPLE_RATE: str = 8000
DEEPGRAM_URI: str = f"wss://api.deepgram.com/v1/listen?model={DEEPGRAM_MODEL}&language={LANGUAGE}&version={VERSION}&punctuate={PUNCTUATE}&interim_results={INTERIM_RESULTS}&endpointing={ENDPOINTING}&utterance_end_ms={UTTERANCE_END_MS}&sample_rate={SAMPLE_RATE}&encoding={ENCODING}&vad_events={VAD_EVENTS}"
HEADERS : dict = {'Authorization': f'Token {DEEPGRAM_API_KEY}'}
DEFAULT_MESSAGE : str = "Sorry, can you repeat that again?" # This will the default transcription output


# Configure your server using Ngrok: https://ngrok.com/docs/getting-started/
HTTP_SERVER_PORT : int = 3000 
NGROK_HTTPS_URL : str = ''
WEBSOCKET_SUBDOMAIN : str = NGROK_HTTPS_URL.replace("https://", "")
BASE_WEBSOCKET_URL : str = f"wss://{WEBSOCKET_SUBDOMAIN}"

# Define session management key
SECRET_KEY : str = "secret!"

# You can add multiple agent configurations here:
MYRA_CONFIG_PATH : str = f'{AGENT_CONFIGS_FOLDER}/Myra_config.json'
MYRA_CACHE_PATH : str = f'{AGENT_CACHE_FOLDER}/myra.pkl'
CHRIS_CONFIG_PATH : str = f'{AGENT_CONFIGS_FOLDER}/Chris_config.json'
CHRIS_CACHE_PATH : str = f'{AGENT_CACHE_FOLDER}/chris.pkl'


# Agent configuration (These will configure your ConversationalModel worker. In the example below I chose Myra):
AGENT_CACHE_FILE : str = MYRA_CACHE_PATH
AGENT_CONFIG_PATH : str = MYRA_CONFIG_PATH


# This is a key value pair to match the output of the filler predictor model to a filler audio file to be played on the call
#          -->  'key' : output_label
#          -->  'value' : file_path
LABEL_TO_FILLER = {
    'General-Inquiry' : f"{AGENT_AUDIO_FOLDER}/General-Inquiry-filler.wav",
    'Company-Inquiry' : f"{AGENT_AUDIO_FOLDER}/Company-Inquiry-filler.wav",
    'Concern' : f"{AGENT_AUDIO_FOLDER}/Concern-filler.wav",
    'Confused' : f"{AGENT_AUDIO_FOLDER}/Confused-filler.wav",
    'Positive-Intent' : f"{AGENT_AUDIO_FOLDER}/Positive-Intent-filler.wav",
    'Dont-Understand' : f"{AGENT_AUDIO_FOLDER}/Dont-Understand-filler.wav",
    'None' : None
}