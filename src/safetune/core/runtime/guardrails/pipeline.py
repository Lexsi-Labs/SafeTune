"""
Production Guardrail Pipeline.

InputSanitizer: Prompt injection detection, PII scrubbing, template validation.
OutputVerifier: Toxicity scan, schema enforcement, refusal verification.
AuditLogger: Structured safety event logging.
GuardrailPipeline: Composable Input → Generate → Output → Audit pipeline.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# InputSanitizer
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InputSanitizerConfig:
    """Configuration for input sanitization."""

    detect_prompt_injection: bool = True
    scrub_pii: bool = True
    max_input_length: int = 8192
    injection_patterns: List[str] = field(default_factory=lambda: [
        r"ignore\s+(all\s+)?previous\s+(instructions?|prompts?)",
        r"you\s+are\s+now\s+\w+",
        r"system\s*:\s*",
        r"SYSTEM\s+OVERRIDE",
        r"forget\s+(everything|all)",
        r"disregard\s+(everything|all|your)",
    ])
    pii_patterns: Dict[str, str] = field(default_factory=lambda: {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    })


class InputSanitizer:
    """Sanitize and validate user inputs before they reach the model.

    The default detectors are regex-based and intentionally transparent.
    For higher accuracy, pass ``injection_classifier`` — a callable
    ``str -> {"is_injection": bool, "score": float, ...}`` — which, when
    supplied, augments (logical OR) the regex result.  The regex layer is
    always kept as a deterministic floor so the sanitizer is never inert.
    """

    def __init__(
        self,
        config: Optional[InputSanitizerConfig] = None,
        injection_classifier: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        self.config = config or InputSanitizerConfig()
        self.injection_classifier = injection_classifier
        self._compiled_injection = [
            re.compile(p, re.IGNORECASE) for p in self.config.injection_patterns
        ]
        self._compiled_pii = {
            name: re.compile(pattern) for name, pattern in self.config.pii_patterns.items()
        }

    def detect_injection(self, text: str) -> Dict[str, Any]:
        """Check for prompt injection patterns.

        Combines the regex pattern bank with an optional caller-supplied
        ``injection_classifier`` (logical OR — either signal flags injection).
        """
        matches = []
        for pattern in self._compiled_injection:
            found = pattern.findall(text)
            if found:
                matches.append({"pattern": pattern.pattern, "matches": len(found)})
        regex_detected = len(matches) > 0

        classifier_detected = False
        if self.injection_classifier is not None:
            try:
                clf = self.injection_classifier(text) or {}
                classifier_detected = bool(clf.get("is_injection", False))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("InputSanitizer: injection_classifier failed — %s", exc)

        return {
            "injection_detected": regex_detected or classifier_detected,
            "patterns_matched": len(matches),
            "details": matches,
            "classifier_detected": classifier_detected,
        }

    def scrub_pii(self, text: str) -> Dict[str, Any]:
        """Replace PII with placeholders."""
        scrubbed = text
        pii_found: Dict[str, int] = {}
        for name, pattern in self._compiled_pii.items():
            matches = pattern.findall(scrubbed)
            if matches:
                pii_found[name] = len(matches)
                scrubbed = pattern.sub(f"[{name.upper()}_REDACTED]", scrubbed)
        return {
            "scrubbed_text": scrubbed,
            "pii_found": pii_found,
            "total_pii": sum(pii_found.values()),
        }

    def sanitize(self, text: str) -> Dict[str, Any]:
        """Full sanitization pipeline."""
        result: Dict[str, Any] = {"original_length": len(text)}

        # length check
        if len(text) > self.config.max_input_length:
            text = text[:self.config.max_input_length]
            result["truncated"] = True

        # injection detection
        if self.config.detect_prompt_injection:
            inj = self.detect_injection(text)
            result["injection"] = inj
            if inj["injection_detected"]:
                result["blocked"] = True
                result["block_reason"] = "prompt_injection_detected"
                return result

        # PII scrubbing
        if self.config.scrub_pii:
            pii_result = self.scrub_pii(text)
            result["pii"] = {k: v for k, v in pii_result.items() if k != "scrubbed_text"}
            text = pii_result["scrubbed_text"]

        result["sanitized_text"] = text
        result["blocked"] = False
        return result


# ═══════════════════════════════════════════════════════════════════
# OutputVerifier
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OutputVerifierConfig:
    """Configuration for output verification."""

    check_toxicity: bool = True
    check_refusal_consistency: bool = True
    max_output_length: int = 16384
    toxicity_keywords: List[str] = field(default_factory=lambda: [
        "kill", "murder", "bomb", "weapon", "hack into",
        "credit card number", "social security", "password",
    ])
    # Keywords severe enough that a *single* occurrence flags the output,
    # regardless of ``toxicity_threshold``.
    strong_toxicity_keywords: List[str] = field(default_factory=lambda: [
        "how to build a bomb", "how to make a bomb", "build a bomb",
        "how to kill", "step-by-step instructions to",
    ])
    toxicity_threshold: int = 2
    custom_deny_patterns: List[str] = field(default_factory=list)


class OutputVerifier:
    """Verify model outputs before returning to the user.

    Toxicity screening is keyword-based by default.  Pass ``toxicity_classifier``
    — a callable ``str -> {"is_toxic": bool, "score": float, ...}`` — for a real
    model-backed check; when supplied it is combined (logical OR) with the
    keyword heuristic so the verifier always has a deterministic floor.
    """

    def __init__(
        self,
        config: Optional[OutputVerifierConfig] = None,
        toxicity_classifier: Optional[Callable[[str], Dict[str, Any]]] = None,
    ) -> None:
        self.config = config or OutputVerifierConfig()
        self.toxicity_classifier = toxicity_classifier

    def check_toxicity(self, text: str) -> Dict[str, Any]:
        """Toxicity check on output.

        Output is flagged toxic when **any** of:
          * the number of matched ``toxicity_keywords`` reaches ``toxicity_threshold``;
          * a ``strong_toxicity_keyword`` appears (single occurrence is enough);
          * an optional ``toxicity_classifier`` returns ``is_toxic=True``.
        """
        text_lower = text.lower()
        matches = [kw for kw in self.config.toxicity_keywords if kw in text_lower]
        strong_matches = [
            kw for kw in self.config.strong_toxicity_keywords if kw in text_lower
        ]
        keyword_toxic = (
            len(matches) >= self.config.toxicity_threshold or len(strong_matches) > 0
        )

        classifier_toxic = False
        classifier_score = None
        if self.toxicity_classifier is not None:
            try:
                clf = self.toxicity_classifier(text) or {}
                classifier_toxic = bool(clf.get("is_toxic", False))
                classifier_score = clf.get("score")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("OutputVerifier: toxicity_classifier failed — %s", exc)

        return {
            "is_toxic": keyword_toxic or classifier_toxic,
            "keyword_matches": len(matches),
            "matched_keywords": matches,
            "strong_matched_keywords": strong_matches,
            "classifier_toxic": classifier_toxic,
            "classifier_score": classifier_score,
        }

    def check_refusal_consistency(
        self,
        prompt: str,
        response: str,
        expected_refusal: bool = False,
    ) -> Dict[str, Any]:
        """Check if the model's refusal/compliance is consistent with expectations."""
        refusal_indicators = [
            "i cannot", "i can't", "i'm unable", "i must refuse",
            "i won't", "not appropriate", "i'm not able",
        ]
        resp_lower = response.lower()
        did_refuse = any(ind in resp_lower for ind in refusal_indicators)
        consistent = (expected_refusal == did_refuse)
        return {
            "did_refuse": did_refuse,
            "expected_refusal": expected_refusal,
            "consistent": consistent,
        }

    def verify(
        self,
        prompt: str,
        response: str,
        expected_refusal: bool = False,
    ) -> Dict[str, Any]:
        """Full output verification."""
        result: Dict[str, Any] = {"response_length": len(response)}

        if len(response) > self.config.max_output_length:
            result["truncated"] = True
            response = response[:self.config.max_output_length]

        if self.config.check_toxicity:
            tox = self.check_toxicity(response)
            result["toxicity"] = tox
            if tox["is_toxic"]:
                result["blocked"] = True
                result["block_reason"] = "toxic_output"
                return result

        if self.config.check_refusal_consistency:
            ref = self.check_refusal_consistency(prompt, response, expected_refusal)
            result["refusal_check"] = ref

        # custom deny patterns
        for pattern in self.config.custom_deny_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                result["blocked"] = True
                result["block_reason"] = f"custom_pattern: {pattern}"
                return result

        result["blocked"] = False
        result["verified_response"] = response
        return result


# ═══════════════════════════════════════════════════════════════════
# AuditLogger
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AuditLoggerConfig:
    """Configuration for safety audit logging."""

    log_file: Optional[str] = None
    log_level: str = "INFO"
    include_full_text: bool = False
    max_text_preview: int = 200
    hash_pii: bool = True


class AuditLogger:
    """Structured safety event logging for audit trails."""

    def __init__(self, config: Optional[AuditLoggerConfig] = None) -> None:
        self.config = config or AuditLoggerConfig()
        self._events: List[Dict[str, Any]] = []

    def _truncate(self, text: str) -> str:
        if self.config.include_full_text:
            return text
        if len(text) > self.config.max_text_preview:
            return text[:self.config.max_text_preview] + "..."
        return text

    def log_event(
        self,
        event_type: str,
        prompt: str,
        response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log a safety event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "prompt_preview": self._truncate(prompt),
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        }
        if response is not None:
            event["response_preview"] = self._truncate(response)
            event["response_hash"] = hashlib.sha256(response.encode()).hexdigest()[:16]
        if metadata:
            event["metadata"] = metadata

        self._events.append(event)

        if self.config.log_file:
            with open(self.config.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")

        return event

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def get_safety_summary(self) -> Dict[str, Any]:
        """Summarise logged safety events."""
        type_counts: Dict[str, int] = {}
        for ev in self._events:
            t = ev["event_type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_events": len(self._events),
            "by_type": type_counts,
        }


# ═══════════════════════════════════════════════════════════════════
# GuardrailPipeline
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GuardrailPipelineConfig:
    """Configuration for the composable guardrail pipeline."""

    enable_input_sanitizer: bool = True
    enable_output_verifier: bool = True
    enable_audit: bool = True
    block_on_injection: bool = True
    block_on_toxic_output: bool = True
    refusal_response: str = (
        "I'm sorry, but I cannot process this request due to safety policy constraints."
    )


class GuardrailPipeline:
    """End-to-end production guardrail pipeline.

    Flow: Input Sanitization → Generation → Output Verification → Audit
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        config: Optional[GuardrailPipelineConfig] = None,
        sanitizer: Optional[InputSanitizer] = None,
        verifier: Optional[OutputVerifier] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.config = config or GuardrailPipelineConfig()
        self.generate_fn = generate_fn
        self.sanitizer = sanitizer or InputSanitizer()
        self.verifier = verifier or OutputVerifier()
        self.audit = audit_logger or AuditLogger()

    def process(self, prompt: str) -> Dict[str, Any]:
        """Run the full guardrail pipeline on a single prompt."""
        result: Dict[str, Any] = {"original_prompt": prompt[:200]}

        # Step 1: Input sanitization
        if self.config.enable_input_sanitizer:
            san = self.sanitizer.sanitize(prompt)
            result["input_sanitization"] = {
                k: v for k, v in san.items()
                if k not in ("sanitized_text", "original_length")
            }
            if san.get("blocked"):
                result["final_response"] = self.config.refusal_response
                result["blocked"] = True
                result["block_stage"] = "input"
                if self.config.enable_audit:
                    self.audit.log_event("input_blocked", prompt, metadata=san)
                return result
            prompt = san.get("sanitized_text", prompt)

        # Step 2: Generation
        try:
            response = self.generate_fn(prompt)
        except Exception as e:
            result["final_response"] = self.config.refusal_response
            result["blocked"] = True
            result["block_stage"] = "generation_error"
            result["error"] = str(e)
            if self.config.enable_audit:
                self.audit.log_event("generation_error", prompt, metadata={"error": str(e)})
            return result

        # Step 3: Output verification
        if self.config.enable_output_verifier:
            ver = self.verifier.verify(prompt, response)
            result["output_verification"] = {
                k: v for k, v in ver.items()
                if k not in ("verified_response",)
            }
            if ver.get("blocked"):
                result["final_response"] = self.config.refusal_response
                result["blocked"] = True
                result["block_stage"] = "output"
                if self.config.enable_audit:
                    self.audit.log_event("output_blocked", prompt, response, metadata=ver)
                return result
            response = ver.get("verified_response", response)

        # Step 4: Audit
        result["final_response"] = response
        result["blocked"] = False
        if self.config.enable_audit:
            self.audit.log_event("allowed", prompt, response)

        return result
