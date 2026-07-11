"""
Data compilers and filters for safety-aligned fine-tuning.

Includes representation-based dataset filters (Similarity Risk, LARF) designed
to prune downstream fine-tuning data that threatens safety guardrails.
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_hidden_states(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    device: Any,
    batch_size: int = 16,
) -> torch.Tensor:
    """Compute normalized final-token hidden states for a list of texts."""
    hiddens = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            # Forward pass asking for hidden states
            outputs = model(**inputs, output_hidden_states=True)
            # (batch, seq_len, hidden_dim)
            if hasattr(outputs, "hidden_states"):
                last_layer = outputs.hidden_states[-1]
            else:
                # fallback for generic encoders
                last_layer = outputs.last_hidden_state
            
            # Extract representation of the final token in each sequence
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(last_layer.size(0), device=device)
            final_states = last_layer[batch_indices, seq_lens, :]
            
            # Normalize for cosine similarity
            final_states = F.normalize(final_states, p=2, dim=1)
            hiddens.append(final_states.cpu())
            
    return torch.cat(hiddens, dim=0)


def filter_by_similarity_risk(
    downstream_data: List[Dict[str, Any]],
    safety_data: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    target_tokens_to_keep: int,
    strategy: str = "low_sim",
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Similarity Risk Filter (https://hsiung.cc/llm-similarity-risk/).
    
    Filters downstream data based on its representation similarity to upstream
    safety-alignment data. Retaining "low_sim" data reduces catastrophic
    forgetting of safety guardrails.
    
    Args:
        downstream_data: List of dicts with 'text' or 'prompt' keys.
        safety_data: Reference safety dataset.
        model: Base LLM.
        tokenizer: Tokenizer.
        target_tokens_to_keep: Max number of examples to keep (budget).
        strategy: "low_sim" (recommended for safety) or "high_sim".
    """
    device = next(model.parameters()).device
    
    downstream_texts = [d.get("text", d.get("prompt", "")) for d in downstream_data]
    safety_texts = [s.get("text", s.get("prompt", "")) for s in safety_data]
    
    logger.info("Computing downstream hidden states...")
    downstream_hiddens = compute_hidden_states(model, tokenizer, downstream_texts, device, batch_size)
    
    logger.info("Computing safety hidden states...")
    safety_hiddens = compute_hidden_states(model, tokenizer, safety_texts, device, batch_size)
    
    # Compute max cosine similarity for each downstream example against ANY safety example
    # downstream_hiddens: (N, D), safety_hiddens: (M, D)
    # Cosine sim matrix: (N, M)
    logger.info("Computing representation similarities...")
    sim_scores = []
    
    # Chunked matmul to save memory if datasets are large
    chunk_size = 500
    for i in range(0, downstream_hiddens.size(0), chunk_size):
        dh_chunk = downstream_hiddens[i : i + chunk_size]
        # (chunk_size, D) @ (D, M) -> (chunk_size, M)
        sim_matrix = torch.matmul(dh_chunk, safety_hiddens.T)
        # Max sim for each downstream example
        max_sims, _ = sim_matrix.max(dim=1)
        sim_scores.extend(max_sims.tolist())
        
    for item, score in zip(downstream_data, sim_scores):
        item["_safety_sim"] = score
        
    # Sort downstream data based on similarity
    ranked_data = sorted(
        downstream_data,
        key=lambda x: x["_safety_sim"],
        reverse=(strategy == "high_sim"),
    )
    
    # Take top K fitting budget
    kept_data = ranked_data[:target_tokens_to_keep]
    logger.info(
        "Similarity filter retained %d/%d examples (%s strategy).",
        len(kept_data), len(downstream_data), strategy
    )
    return kept_data


def filter_by_larf(
    downstream_data: List[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    safety_layers: List[int],
    keep_ratio: float = 0.5,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Layer-Aware Representation Filtering (LARF).
    [EMNLP 2025] "Layer-Aware Representation Filtering: Purifying Finetuning Data
    to Preserve LLM Safety Alignment".
    
    Measures bidirectional interference at safety-sensitive layers. Data that
    causes large shifts in safety-layer representations is pruned.
    
    Args:
        downstream_data: FT dataset to filter.
        model: Model.
        tokenizer: Tokenizer.
        safety_layers: Indices of recognized safety-sensitive layers.
        keep_ratio: Fraction of dataset to keep (lowest interference).
    """
    device = next(model.parameters()).device
    texts = [d.get("text", d.get("prompt", "")) for d in downstream_data]
    
    interference_scores = []
    model.eval()
    
    # LARF approximates interference by comparing intermediate activations
    # We collect activations at the specified safety_layers.
    
    logger.info("Computing LARF interference scores for %d samples...", len(texts))
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            if not hasattr(outputs, "hidden_states"):
                # fallback for generic models without intermediate states exposed
                # If we can't do layer-aware, assign 0 interference.
                interference_scores.extend([0.0] * len(batch_texts))
                continue
                
            hidden_states = outputs.hidden_states
            # Calculate standard deviation/norm of activations across the specific 
            # safety-sensitive layers as a proxy for representational drift/interference.
            # (In the paper, it uses a bidirectional divergence metric, approximated here 
            # by the representation magnitude at sensitive layers).
            
            batch_scores = torch.zeros(len(batch_texts), device=device)
            for layer_idx in safety_layers:
                if layer_idx < len(hidden_states):
                    layer_acts = hidden_states[layer_idx]
                    # Average activation norm per sequence
                    batch_scores += torch.norm(layer_acts.float(), dim=-1).mean(dim=1)
            
            interference_scores.extend(batch_scores.cpu().tolist())
            
    for item, score in zip(downstream_data, interference_scores):
        item["_larf_interference"] = score
        
    ranked_data = sorted(downstream_data, key=lambda x: x["_larf_interference"])
    
    target_count = max(1, int(len(downstream_data) * keep_ratio))
    kept_data = ranked_data[:target_count]
    
    logger.info(
        "LARF filter retained %d/%d examples (keep_ratio=%.2f).",
        len(kept_data), len(downstream_data), keep_ratio
    )
    return kept_data
