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

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatLiteLLM

from ConversationModel.logger import time_logger
from ConversationModel.prompts import (HR_STAGE_ANALYZER_INCEPTION_PROMPT, SALES_AGENT_DEFAULT_INCEPTION_PROMPT)


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        stage_analyzer_inception_prompt_template = HR_STAGE_ANALYZER_INCEPTION_PROMPT
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt_template,
            input_variables=[
                "conversation_history",
                "conversation_stage_id",
                "conversation_stages",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class ConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    @time_logger
    def from_llm(
        cls,
        llm: ChatLiteLLM,
        inputs : list,
        custom_prompt: str,
        verbose: bool = True,
        use_custom_prompt: bool = True
    ) -> LLMChain:
        """Get the response parser."""
        if use_custom_prompt:
            sales_agent_inception_prompt = custom_prompt
            prompt = PromptTemplate(
                template=sales_agent_inception_prompt,
                input_variables=[
                    "conversation_history",
                ] + inputs,
            )
        else:
            sales_agent_inception_prompt = SALES_AGENT_DEFAULT_INCEPTION_PROMPT
            prompt = PromptTemplate(
                template=sales_agent_inception_prompt,
                input_variables=[
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
