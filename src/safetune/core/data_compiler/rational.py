"""
Rational Formatting Data Compiler.
Reference: "Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety"

Transforms standard instruction-response pairs into reasoning-enhanced 
"rational formatting" templates. This forces the LLM to output a chain-of-thought 
rationale explicitly analyzing safety *before* providing its final judgement.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

class RationalFormatter:
    """
    Transforms standard datasets into a Rational Formatting template.
    
    The paper demonstrates that when safety tuning data includes deliberate 
    chain-of-thought reasoning before an act of refusal (or safety compliance),
    the model transfers safety guardrails to downstream tasks much more effectively
    and degrades far less during future SFT.
    """
    
    SYSTEM_PROMPT = (
        "You are an AI assistant. Before responding to the user's request, "
        "you must analyze it for safety. First output your rationale inside <rationale> tags, "
        "then output your final response."
    )
    
    def __init__(self, include_system_prompt: bool = True):
        self.include_system_prompt = include_system_prompt
        
    def format_record(self, prompt: str, rationale: str, response: str) -> Dict[str, Any]:
        """
        Formats a single record with the rational structure.
        """
        messages = []
        if self.include_system_prompt:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
            
        messages.append({"role": "user", "content": prompt})
        
        # The key reasoning structure
        assistant_content = f"<rationale>\n{rationale}\n</rationale>\n\n{response}"
        messages.append({"role": "assistant", "content": assistant_content})
        
        return {"messages": messages, "is_rational": True}

    def compile_dataset(
        self, 
        records: List[Dict[str, str]], 
        prompt_key: str = "prompt", 
        rationale_key: str = "rationale", 
        response_key: str = "response"
    ) -> List[Dict[str, Any]]:
        """
        Compiles an entire dataset into rational formatting.
        
        Args:
            records: A list of dicts representing the raw dataset.
            prompt_key: Key for the user instruction.
            rationale_key: Key for the chain-of-thought safety analysis.
            response_key: Key for the final response (refusal or compliance).
            
        Returns:
            A list of dictionary records containing conversation arrays in the Rational format.
        """
        logger.info(f"Compiling dataset of {len(records)} records into Rational Formatting.")
        compiled = []
        for i, rec in enumerate(records):
            try:
                formatted = self.format_record(
                    prompt=rec[prompt_key],
                    rationale=rec[rationale_key],
                    response=rec[response_key]
                )
                compiled.append(formatted)
            except KeyError as e:
                logger.warning(f"Record {i} missing key {e}. Skipping.")
                
        return compiled
