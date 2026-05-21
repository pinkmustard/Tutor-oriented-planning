batch_item = {
    "custom_id": f"{cid}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-5-mini",
        "messages": messages,
        "max_completion_tokens": 3000,   # reasoning 토큰 포함, 1000은 위험
        "reasoning_effort": "low"        # 평가 태스크는 low로 충분
        # temperature 제거 (gpt-5는 기본값 1만 허용)
    }
}