"""
Interface for interacting with language models through Ollama and OpenAI.
"""

import ollama
from ollama import chat
from openai import OpenAI
import os
from typing import Optional

def get_llm_response(prompt: str, model: str = 'qwen2.5:32b') -> str:
    """
    Get a response from an Ollama model.
    
    Args:
        prompt: The input prompt for the model
        model: The name of the Ollama model to use
        
    Returns:
        The model's response text
        
    Raises:
        RuntimeError: If the model is not available or there's an error getting the response
    """
    try:
        response = chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'num_predict': 1024}
        )
        return response['message']['content']
    except Exception as e:
        raise RuntimeError(f"Error getting response from Ollama model {model}: {str(e)}")

def get_openai_response(prompt: str, model: str = 'gpt-4') -> str:
    """
    Get a response from an OpenAI model.
    
    Args:
        prompt: The input prompt for the model
        model: The name of the OpenAI model to use
        
    Returns:
        The model's response text
        
    Raises:
        RuntimeError: If the API key is not set or there's an error getting the response
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
        
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error getting response from OpenAI model {model}: {str(e)}")