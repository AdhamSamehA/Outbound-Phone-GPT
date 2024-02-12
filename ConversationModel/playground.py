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

from agents import ConversationalModel
import asyncio
import time

agent = ConversationalModel().init_agent()
ai_response = ""

def update_agent(agent : ConversationalModel):
    """
    Updates the agent's response after processing user input.

    Args:
    None: This method doesn't take in any arguments

    Returns:
    None: This method updates internal state and does not return anything.
    """
    global ai_response
    agent_name = agent.get_attribute('agent_name')
    ai_response = f"{agent_name}: {ai_response.strip()}"
    if "<END_OF_TURN>" not in ai_response:
        ai_response += " <END_OF_TURN>"
    agent.conversation_history.append(ai_response)
    ai_response=""

async def simulate():
    """ This method allows you to interact with an example agent instance in order to test out your custom agent configuration
    before taking your agent into production """
    global ai_response
    while True:
        user_input = input("Enter message: ")
        if user_input == "Bye":
            break
        time_start = time.time()
        measure_time = True
        agent.human_step(user_input)
        streaming_gen = await agent._astreaming_generator()
        async for chunk in streaming_gen:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                if measure_time:
                    time_end = time.time()
                    print(f"Time taken to first chunk: {time_end-time_start} seconds")
                    measure_time = False
                ai_response+=chunk_content
        print(f"{agent.get_attribute('agent_name')}: {ai_response}")
        update_agent(agent)
        print(f"Conversation history: {agent.conversation_history}")

asyncio.run(simulate())



