batch_item = {
    "custom_id": f"{cid}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_completion_tokens": 1000,
        "temperature": 0.0
    }
}
