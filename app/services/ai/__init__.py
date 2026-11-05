"""Provider-agnostic AI interpretation layer.

Provider order is OpenAI (active) -> Qwen (reserved in-house slot, not enabled)
-> deterministic local rules (always available). Every service that wants a
structured AI interpretation builds a :class:`ProviderRequest` and calls
:func:`backend.app.services.ai.registry.generate_structured_json`; the local
calculations remain authoritative and are the final fallback.
"""
