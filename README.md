![Banner Alt Text](/resources/Project_Banner.png)

# Outbound Phone GPT

Outbound Phone GPT is a sophisticated prototype for a context-aware agent designed to autonomously handle outbound phone calls through Twilio. The system is built on core principles of asynchronous programming to enable real-time speech-to-speech communication.

## License

This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details.


## Features

- Context-Aware Conversations: Engages in meaningful dialogues, tailored to the specifics of each phone call.
- Conversation Management: Manages the flow of conversations, ensuring coherent and goal-oriented interactions.
- Twilio Integration: Utilizes Twilio's robust API for handling outbound phone calls.
- STT and TTS: Incorporates Deepgram for Speech-to-Text (STT) and Eleven Labs for Text-to-Speech (TTS) capabilities.
- Response Time: Typically between 1 to 2 seconds, based on how quickly OpenAI's system processes and streams the response back.


**Note on Response Times:** 

- In the US: If you're using the service in the United States, you can expect faster response times for the entire speech to speech process. This is due to the closer proximity of the servers, reducing data travel time. For example, the time it takes from when you ask something to when Open AI sends the first part of the response (we call this "inference time") can reach about 250 milliseconds.

- Outside the US: If you're not in the US, like in the Middle East and North Africa (MENA) region, response times might be a bit longer. For instance, in the MENA region, it can take anywhere between 0.7-1.4 seconds just for the first part of the response from Open AI to start coming through ([Text-To-Text Latency In MENA](./resources/images/GPT-Response-Latency.png))


- Overall Performance: Even though the time it takes to get the first part of Open AI's response can vary depending on your location, the Outbound Phone GPT system as a whole is built to streamline the speech to speech process with a big emphasis on latency optimisation. This means that, despite the extra steps involved in delivering a realistic human-like outbound call, the system is able to provide responses in as little as ~1.5 seconds ([Speech-To-Speech Latency In MENA](./resources/images/Agent-On-Call-Response-Latency.png))


## Technologies and External APIs

- FastAPI: For creating Outbound Phone GPT's API
- WebSockets: For real-time bidirectional communication.
- Twilio's API: For phone call management.
- Deepgram: For real-time speech recognition.
- Eleven Labs: For dynamic and natural-sounding voice synthesis.
- Langchain: For building the core logic of the conversational agent
- Baby AGI framework inspired by Filip Michalsky's SalesGPT -> https://github.com/filip-michalsky/SalesGPT

## Setup and Installation

### 1) Create a Python Virtual Environment:

``` python
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


### 2) Install Dependencies:

``` python
pip install -r requirements.txt
```


### 3) Set up your Ngrok server on a new terminal:

Ensure Ngrok is installed: https://ngrok.com/docs/getting-started/. Then set up the server through the terminal:
``` bash
ngrok http <HHTP_SERVER_PORT>
```

### 4) Set up Environment Variables:

Create your `.env` file using `.env.example` as a template and add your keys


### 5) Setting Up Your Agent:

- **Step 1: Go to /ConversationModel/prompts.py**
    
    Write a custom `STAGE_ANALYZER_INCEPTION_PROMPT` as per the examples given, to suit your use case

- **Step 2: Go to /ConversationModel/stages.py**

    Write a custom set of `CONVERSATION_STAGES` as per the examples given, to suit your use case

- **Step 3: Create the Agent's Configuration File**

    First, you need to make a configuration file for your agent in JSON format. This file should include two main pieces of information: `prompt` and `agent_name`.

    - `prompt` is like an instruction that tells your agent what to do and how to manage the call. Remember to include 
    `{conversation_history}` in your prompt, to enable the agent's context awareness as well as the conversation stages
        which you've defined earlier.
    - `agent_name` is simply the name you want to give your agent.
    
    You can add more details to this file if you want (examples can be found at [here](./example_agent_configs)). These details will be treated as the agent's characteristics and can be used anytime throughout the call.

    When setting up your prompt, use placeholders (like `{agent_name}`) instead of fixed text. This way, the agent can automatically use the specific details you've set in the JSON file, making your prompts more flexible and relevant to each call.

    Example of using placeholders in a prompt:

    For example:
    ``` json
    "agent_name": "Myra",
    "agent_role": "BDR",
    "prospect_name": "John Doe",
    "call_purpose": "to explore potential interest in our products",
    "prompt": "Your name is {agent_name} and your role is {agent_role}. You are calling {prospect_name} for {call_purpose}. You will find the conversation history below:\n\n{conversation_history}"
    ```

- **Step 4: Go to /ConversationModel/playground.py**

    Test your agent's configuration through a chat-like interface in the program's terminal. This is important as it allows you
    to refine your agent's configuration before setting up your agent to take phone calls.

### 6) App Configuration:
    
Edit the `__config__.py` file to set up OpenAI, Eleven Labs, Deepgram keys, and the ngrok server settings. Use
the example `__config__.py` file for guidance.


### 7) Choose your implementation:

Go to `app.py` and choose your implementation by uncommenting the method of choice, and commenting out the others. Here's a breakdown of the methods you can choose from:
    - Method 1: Doesn't utilise the agent's cache or the agent's filler prediction capabalities. This is the default method.
    - Method 2: Utilises the agent's cache but not the agent's filler prediction capabalities
    - Method 3: Incoporates both the agent's cache and filler prediction capabilities

**Keep in mind that the filler prediction model is experimental for the time being.**


### 8) Start the Project:

Run `app.py` to initiate the server



## Endpoints:

**/make-call**

Initiates an outbound call. Can accept a JSON with a custom 'welcome_message'. If you send an empty JSON, make sure 
that a 'default-starter.wav' file exists within your audio files directory. 
``` json
{}  // For default welcome message
{"welcome_message": "Hello, this is Myra from Escade Networks."}
```

**/generate-filler**

Generates audio files from a list of sentence fillers. Can accept a JSON of the following format:
``` json
{
    "fillers": [
        ["Filler text 1", "file_name_1"],
        ["Filler text 2", "file_name_2"],
        ["Filler text 3", "file_name_3"]
    ]
}
```

**/update-cache/**

Adds predefined questions and answers to the agent's cache. Can accept a JSON of the following format:
``` json
[
    {"key": "Hi", "value": "Hey, how is it going?"},
    {"key": "Who are you?", "value": "I'm Myra, the recruitment director at Escade Networks."},
    {"key": "Why are you calling?", "value": "I'm calling to discuss your application for one of our job openings..."}
]
```