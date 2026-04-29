#!/usr/bin/env python3
"""Quick test: send one case to the LLM and print the raw response."""

import asyncio
import os

import anthropic

async def main():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    system = """You are an expert radiologist assistant. Determine whether each prior study is relevant to the current study. Respond ONLY with a JSON array of booleans. No other text."""

    user = '''Current study: "CT CHEST WITH CONTRAST" (date: 2025-08-26)

Prior studies to evaluate (3 total):
1. "CT CHEST WITHOUT CNTRST" (date: 2024-04-13)
2. "MRI knee LT wo con" (date: 2022-10-14)
3. "PET/CT F18 piflu skullthigh in" (date: 2023-10-16)'''

    print(f"Model: {model}")
    response = await client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    print(f"Stop reason: {response.stop_reason}")
    print(f"Content blocks: {len(response.content)}")
    for i, block in enumerate(response.content):
        print(f"  Block {i}: type={type(block).__name__}")
        if hasattr(block, "text"):
            print(f"    text: {repr(block.text)}")
        elif hasattr(block, "thinking"):
            print(f"    thinking: {repr(block.thinking[:200])}")

asyncio.run(main())
