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

from openai import OpenAI
from __config__ import OPENAI_API_KEY, LABEL_TO_FILLER
from __utils__ import normalize_sentence, get_filler

class GPTPredictor():
    """
    The GPTPredictor class is designed to classify user input into predefined categories using Open AI's GPT-3.5
    and suggest appropriate filler content based on the classification.
    """
    # Function to classify user input
    def __init__(self):
        """ Initializes the FillerPredictor instance. """
        self.model = OpenAI(api_key=OPENAI_API_KEY)
        self.categories = list(LABEL_TO_FILLER.keys())
    
    def classify_intent(self, user_input: str):
        """
        Classifies the intent of the user input and retrieves the corresponding filler path.

        Args:
        user_input: A string containing the user's input sentence.

        Returns:
        A string representing the path to the filler content if the classification is successful and a filler is available; otherwise, None.
        """
        # Normalize the user input
        normalized_input = normalize_sentence(user_input)

        # Create the API request payload
        response = self.model.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": f"Classify the intent of 'Input' into 1 label from: {self.categories}. If input doesn't match any return None, else return label only."},
                {"role": "user", "content": f"Input: {normalized_input}"}
            ],
            max_tokens=5,
            temperature=0
        )
        # Extract and return the classification label from the response
        label = response.choices[0].message.content.strip()
        filler_path = get_filler(label)
        if filler_path:
            return filler_path
        else:
            return None


#class InputClassifier():
#    """  
#    The InputClassifier class is designed to classify user input into predefined categories using a fine-tuned model
#    and suggest appropriate filler content based on the classification. The goal is to achieve fast and accurate classification labels so that
#    we can speed up inference by 50% (~500ms)
#    
#    """
#    raise NotImplementedError("This class is not yet implemented")
