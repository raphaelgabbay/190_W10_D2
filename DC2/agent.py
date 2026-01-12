def run_agent(messages) -> str:
    from smolagents import LiteLLMModel

    model = LiteLLMModel(
        model_id="ollama_chat/qwen2:7b",     # Any Ollama-supported chat model
        api_base="http://127.0.0.1:11434",   # Ollama’s OpenAI-compatible endpoint
        num_ctx=4096,                        # Adjust per model capability
    )

    formattedMsg = normalize_messages(messages)

    return model(messages)

def normalize_messages(messages):
    normalized = []
    for m in messages:
        if isinstance(m["content"], str):
            normalized.append({
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}]
            })
        else:
            normalized.append(m)
    return normalized