import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

ENDPOINTS = [
    {
        "name": "gemma_2_9b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_KEY,
        "model": "google/gemma-2-9b-it",
    },
    {
        "name": "llama_3_8b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_KEY,
        "model": "meta-llama/llama-3-8b-instruct",
    },
    {
        "name": "mistral_7b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_KEY,
        "model": "mistralai/mistral-7b-instruct",
    },
    {
        "name": "qwen_2_72b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_KEY,
        "model": "qwen/qwen-2-72b-instruct",
    },
    {
        "name": "gpt-4o",
        "url": OPENAI_URL,
        "key": OPENAI_KEY,
        "model": "gpt-4o",
    },
]

PROMPT = (
    "Rewrite the following news article while keeping the core facts and meaning intact. "
    "Maintain professional journalism style without meta-commentary.\n\n"
    "Original Article:\n"
    "Federal regulators on Thursday announced new safety standards for commercial aviation."
)

async def test_endpoint(session, ep):
    headers = {
        "Authorization": f"Bearer {ep['key']}",
        "Content-Type": "application/json",
    }
    if "openrouter" in ep["url"]:
        headers["HTTP-Referer"] = "https://github.com/ai-text-detection"
        headers["X-Title"] = "AI Text Detection Research"

    payload = {
        "model": ep["model"],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    try:
        async with session.post(ep["url"], headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            status = resp.status
            data = await resp.json()
            if status == 200:
                content = data["choices"][0]["message"]["content"].strip()
                print(f"[OK] {ep['name']} ({ep['model']}): {content[:60]}...")
                return ep["name"], True, ep["model"], None
            else:
                err = data.get("error", {}).get("message", str(data))
                print(f"[FAIL {status}] {ep['name']} ({ep['model']}): {err}")
                return ep["name"], False, ep["model"], f"HTTP {status}: {err}"
    except Exception as e:
        print(f"[EXC] {ep['name']} ({ep['model']}): {e}")
        return ep["name"], False, ep["model"], str(e)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [test_endpoint(session, ep) for ep in ENDPOINTS]
        results = await asyncio.gather(*tasks)
    print("\n--- Summary ---")
    for name, ok, model, err in results:
        status_str = "AVAILABLE" if ok else f"FAILED ({err})"
        print(f"{name:15} | {model:35} | {status_str}")

if __name__ == "__main__":
    asyncio.run(main())
