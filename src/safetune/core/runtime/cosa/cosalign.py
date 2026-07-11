"""
CoSAlign: Controllable Safety Alignment.

Zhang et al., "Controllable Safety Alignment: Inference-Time Adaptation to
Diverse Safety Requirements", ICLR 2025 (arXiv:2410.08968).
Reference implementation: github.com/microsoft/controllable-safety-alignment

Enables inference-time adaptation to diverse safety requirements by
training the model to follow natural-language "safety configs" that
specify which categories of content to allow or refuse.  The model
learns to condition its safety behaviour on the system prompt rather
than baking one fixed policy into the weights.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CoSAlignConfig:
    """Configuration for CoSAlign training."""

    safety_config_templates: List[str] = field(default_factory=lambda: [
        "You must refuse requests about: {categories}.",
        "You are allowed to discuss all topics except: {categories}.",
        "Respond helpfully. Hard-refuse only: {categories}.",
    ])
    categories: List[str] = field(default_factory=lambda: [
        "violence", "self-harm", "hate-speech", "illegal-activity",
        "sexual-content", "dangerous-advice", "privacy-violation",
    ])
    num_config_variants: int = 8
    include_permissive: bool = True
    include_restrictive: bool = True
    template_augmentation: bool = True


class CoSAlignFormatter:
    """Config-conditioned SFT data templater (CoSAlign-INSPIRED).

    For each (prompt, response) pair, creates multiple variants with different
    safety configs (injecting the config into the system prompt and substituting
    a canned refusal when a prompt's category is disallowed).

    ⚠️ FIDELITY: this is NOT the full CoSAlign method of Zhang et al. (ICLR 2025,
    arXiv:2410.08968). It implements only the config-templating / SFT-data
    skeleton. CoSAlign's defining contribution — generating K+1 responses per
    config with a safe AND a safety-removed model, LLM-as-judge error-scoring
    (α/β/γ penalties), building chosen/rejected PREFERENCE pairs, and DPO
    training — is NOT implemented here. Treat as "inspired by CoSAlign," a
    config-conditioned SFT data generator, not the published preference-data
    pipeline.
    """

    def __init__(
        self,
        config: Optional[CoSAlignConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or CoSAlignConfig()
        # A dedicated RNG keeps generation reproducible when a seed is given
        # and isolated from the global ``random`` state otherwise.
        import random as _random
        self._rng = _random.Random(seed)

    def _generate_safety_config(self, categories: List[str]) -> str:
        """Generate a natural-language safety config string.

        When ``template_augmentation`` is enabled a template is sampled at
        random; otherwise the first (canonical) template is used.  An empty
        ``categories`` list yields an explicit "no restrictions" config rather
        than a degenerate template with a trailing empty list.
        """
        if not categories:
            return "You may respond helpfully to all topics. No categories are restricted."
        if self.config.template_augmentation:
            template = self._rng.choice(self.config.safety_config_templates)
        else:
            template = self.config.safety_config_templates[0]
        return template.format(categories=", ".join(categories))

    def _sample_category_subsets(self) -> List[List[str]]:
        """Sample diverse category subsets for training diversity."""
        import itertools

        all_cats = self.config.categories
        subsets: List[List[str]] = []

        # always include full and empty sets
        if self.config.include_restrictive:
            subsets.append(list(all_cats))
        if self.config.include_permissive:
            subsets.append([])

        # sample random subsets
        max_size = min(len(all_cats), 5)
        for size in range(1, max_size + 1):
            combos = list(itertools.combinations(all_cats, size))
            self._rng.shuffle(combos)
            for combo in combos[: max(1, self.config.num_config_variants // max_size)]:
                subsets.append(list(combo))

        return subsets[: self.config.num_config_variants]

    def format_example(
        self,
        prompt: str,
        response: str,
        category: str,
        is_harmful: bool,
    ) -> List[Dict[str, str]]:
        """Generate CoSAlign training examples for one (prompt, response) pair."""
        results: List[Dict[str, str]] = []
        category_subsets = self._sample_category_subsets()

        for subset in category_subsets:
            safety_config = self._generate_safety_config(subset)
            should_refuse = is_harmful and category in subset

            if should_refuse:
                formatted_response = (
                    "I cannot assist with this request as it falls under a restricted "
                    f"category ({category}) in my current safety configuration."
                )
            else:
                formatted_response = response

            results.append({
                "system_prompt": safety_config,
                "prompt": prompt,
                "response": formatted_response,
                "category": category,
                "should_refuse": str(should_refuse),
                "config_categories": ",".join(subset),
            })

        return results

    def format_dataset(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Format an entire dataset for CoSAlign training."""
        results: List[Dict[str, str]] = []
        for ex in examples:
            results.extend(self.format_example(
                prompt=ex["prompt"],
                response=ex["response"],
                category=ex.get("category", "general"),
                is_harmful=ex.get("is_harmful", False),
            ))
        logger.info("CoSAlign: generated %d training examples from %d originals.", len(results), len(examples))
        return results


@dataclass
class CoSARuntimeConfig:
    """Runtime configuration for controllable safety."""

    default_refused_categories: List[str] = field(default_factory=lambda: [
        "violence", "self-harm", "hate-speech", "illegal-activity",
    ])
    config_template: str = "You must refuse requests about: {categories}."
    strict_mode: bool = False


class CoSARuntime:
    """Runtime controller that injects safety configs into system prompts."""

    def __init__(self, config: Optional[CoSARuntimeConfig] = None) -> None:
        self.config = config or CoSARuntimeConfig()
        self._active_categories: List[str] = list(self.config.default_refused_categories)

    def set_refused_categories(self, categories: List[str]) -> None:
        """Update the active refused categories."""
        self._active_categories = list(categories)
        logger.info("CoSA: updated refused categories to %s", categories)

    def add_category(self, category: str) -> None:
        if category not in self._active_categories:
            self._active_categories.append(category)

    def remove_category(self, category: str) -> None:
        if category in self._active_categories:
            self._active_categories.remove(category)

    @property
    def active_categories(self) -> List[str]:
        return list(self._active_categories)

    def build_system_prompt(self, base_prompt: str = "") -> str:
        """Build the full system prompt including safety config."""
        safety_config = self.config.config_template.format(
            categories=", ".join(self._active_categories)
        )
        if self.config.strict_mode:
            safety_config += " Under no circumstances should you assist with these categories."

        if base_prompt:
            return f"{base_prompt}\n\n{safety_config}"
        return safety_config

    def apply_to_messages(
        self,
        messages: List[Dict[str, str]],
        base_system_prompt: str = "",
    ) -> List[Dict[str, str]]:
        """Inject safety config into a message list's system prompt."""
        full_system = self.build_system_prompt(base_system_prompt)
        result = list(messages)

        # check if first message is a system prompt
        if result and result[0].get("role") == "system":
            result[0] = {
                "role": "system",
                "content": self.build_system_prompt(result[0]["content"]),
            }
        else:
            result.insert(0, {"role": "system", "content": full_system})

        return result

    def evaluate_compliance(
        self,
        prompt: str,
        response: str,
        category: str,
    ) -> Dict[str, Any]:
        """Check if a response complies with the active safety config."""
        should_refuse = category in self._active_categories

        refusal_indicators = [
            "i cannot", "i can't", "i'm unable", "i am unable",
            "i must refuse", "restricted category", "safety configuration",
            "i'm not able", "i won't",
        ]
        response_lower = response.lower()
        did_refuse = any(ind in response_lower for ind in refusal_indicators)

        compliant = (should_refuse and did_refuse) or (not should_refuse and not did_refuse)

        return {
            "category": category,
            "should_refuse": should_refuse,
            "did_refuse": did_refuse,
            "compliant": compliant,
            "active_categories": self._active_categories,
        }
