import os


# simple router to route to the correct LLM provider
def get_provider():
	provider = os.getenv("LLM_PROVIDER", "auto").lower()

	# work based on which API key is available
	if provider == "auto":
		if os.getenv("OPENAI_API_KEY"):
			return "openai"

		if os.getenv("ANTHROPIC_API_KEY"):
			return "anthropic"

		return "offline"

	# telling the provider explicitly (no auto mode here)
	if provider not in ["ollama", "openai", "anthropic", "offline"]:
		raise ValueError(f"Unsupported LLM provider: {provider}")

	if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
		raise ValueError("OPENAI_API_KEY is not set")

	if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
		raise ValueError("ANTHROPIC_API_KEY is not set")

	return provider


# based on provider router, pick model
def get_model(provider):
	if provider == "ollama":
		return os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

	if provider == "openai":
		return os.getenv("OPENAI_MODEL", "gpt-6-luna")

	if provider == "anthropic":
		return os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")

	return "none"
