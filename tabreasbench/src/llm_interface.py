"""
Interface for interacting with language models.

This module uses vLLM (if available) for local/OSS model inference and
keeps the OpenAI-backed helper for remote models. The original Ollama
integration has been replaced with vLLM per project preference.
"""

import os
from typing import Optional

try:  # Optional dependency: vllm
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None

from openai import OpenAI


def get_llm_response(prompt: str, model: str = "qwen2.5:32b") -> str:
    """
    Get a response from a local model using vLLM.

    Args:
        prompt: The input prompt for the model
        model: The name or path of the model vLLM should load

    Returns:
        The model's response text

    Raises:
        RuntimeError: If vLLM is not installed or there's an error getting the response
    """
    if LLM is None:
        raise RuntimeError(
            "vLLM is not installed. Install it with: pip install vllm"
        )

    try:
        # Use a context manager to ensure resources are released.
        with LLM(model=model) as llm:
            sampling_params = SamplingParams(temperature=0.0)
            outputs = llm.generate([prompt], sampling_params=sampling_params, max_tokens=1024)
            # vLLM generate yields batches; return first sequence text
            for batch in outputs:
                for seq in batch:
                    return seq.text
        raise RuntimeError("vLLM returned no output")
    except Exception as e:
        raise RuntimeError(f"Error getting response from vLLM model {model}: {str(e)}")


from typing import List, Optional

def get_llm_responses(
    prompts: List[str],
    llm,
    sampling_params=None,
    max_tokens: int = 1024,
    world_size: Optional[int] = None,
) -> List[str]:
    """
    Generate responses for a list of prompts using vLLM in batch mode.

    Args:
        prompts: List of prompt strings to generate for (keeps order).
        llm: An initialized vLLM LLM instance.
        sampling_params: vLLM SamplingParams object (temperature, max_tokens, etc.).
        max_tokens: Maximum tokens to generate per prompt (optional; you can
                    also set this directly on sampling_params).
        world_size: Currently unused; the LLM should be initialized with this
                    beforehand if needed.

    Returns:
        A list of generated texts (one per prompt) in the same order.

    Raises:
        RuntimeError: If vLLM generation fails.
    """

    try:
        # If you want max_tokens enforced here, uncomment this:
        # sampling_params.max_tokens = max_tokens

        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        responses: List[str] = []
        for out in outputs:
            # vLLM: out.outputs is a list of candidate generations
            if getattr(out, "outputs", None):
                # Take the first candidate for each prompt
                responses.append(out.outputs[0].text)
            else:
                # No outputs for this prompt
                responses.append("")

        # If for some reason vLLM produced fewer outputs than prompts,
        # pad with empty strings to keep the contract stable.
        if len(responses) < len(prompts):
            responses.extend([""] * (len(prompts) - len(responses)))

        return responses

    except Exception as e:
        raise RuntimeError(f"Error generating batch responses from vLLM model: {e}")


def get_openai_response(prompt: str, model: str = "gpt-4") -> str:
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Error getting response from OpenAI model {model}: {str(e)}")