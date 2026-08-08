import os
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

# ──────────────────────────────────────────────
# Load API key from .env
# ──────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────────────────────
# CONFIG: Set to True to run 50-sample test mode
# Set to False to run full 1000 samples
# ──────────────────────────────────────────────
SAMPLE_TEST = False
SAMPLE_SIZE = 50

# ──────────────────────────────────────────────
# OpenRouter - Non-GPT Models only
# :free suffix = free tier on OpenRouter
# ──────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL="https://api.openai.com/v1/chat/completions"
my_apis = [
    {
        # Gemma 3 12B - closest available to Gemma-2-9B
        "name": "gemma_2_9b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "google/gemma-3-12b-it",
    },
    {
        # Qwen 2.5 72B - successor to Qwen-2-72B
        "name": "qwen_2_72b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "qwen/qwen-2.5-72b-instruct",
    },
    {
        # LLaMA 3.1 8B - successor to LLaMA-3-8B
        "name": "llama_3_8b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "meta-llama/llama-3.1-8b-instruct",
    },
    {
        # Mistral Nemo - current available Mistral small model
        "name": "mistral_7b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "mistralai/mistral-nemo",
    },
    {
        # Since Yi-large closed we will test DeepSeek v4 Pro instead
        "name": "deepseek_pro",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "deepseek/deepseek-v4-pro",
    },
    {
        "name": "claude_haiku",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "anthropic/claude-3-haiku:beta",
    },
    {
        "name": "gpt-4o-mini",
        "url": OPENAI_URL,
        "key": OPENAI_API_KEY,
        "model": "gpt-4o-mini",
    },

]


# ──────────────────────────────────────────────
# Core fetch with retry logic + detailed error
# ──────────────────────────────────────────────
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(4))
async def fetch_article(session, api_url, api_key, model, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-text-detection",
        "X-Title": "AI Text Detection Research",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a complete, professional news article based on the following summary/abstract. "
                    "Do not add any commentary or meta-text — only output the article itself.\n\n"
                    f"Summary:\n{prompt}"
                ),
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    async with session.post(api_url, headers=headers, json=payload) as response:
        # ── Print detailed error before raising ──
        if response.status != 200:
            body = await response.text()
            print(f"\n  [HTTP {response.status}] Model={model} | Body: {body[:300]}")
            response.raise_for_status()

        data = await response.json()
        return data["choices"][0]["message"]["content"].strip()


# ──────────────────────────────────────────────
# Quick model sanity check (1 request)
# ──────────────────────────────────────────────
async def test_single_model(api):
    """Send 1 request to verify the model works before running batch."""
    headers = {
        "Authorization": f"Bearer {api['key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-text-detection",
        "X-Title": "AI Text Detection Research",
    }
    payload = {
        "model": api["model"],
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 30,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api["url"], headers=headers, json=payload) as resp:
            body = await resp.text()
            if resp.status == 200:
                print(f"  ✅ [{api['name']}] OK  (model: {api['model']})")
                return True
            else:
                print(f"  ❌ [{api['name']}] HTTP {resp.status} | {body[:200]}")
                return False


# ──────────────────────────────────────────────
# Process one API model over a DataFrame
# ──────────────────────────────────────────────
async def process_api(df, api, output_csv):
    api_name = api["name"]

    if api_name not in df.columns:
        df[api_name] = ""

    missing_idx = df[df[api_name] == ""].index

    if len(missing_idx) == 0:
        print(f"[{api_name}] Already completed. Skipping...")
        return

    print(f"[{api_name}] Generating {len(missing_idx)} articles...")

    # Limit concurrency to avoid rate-limiting on free tier
    semaphore = asyncio.Semaphore(3)

    async def bounded_fetch(idx, prompt, session):
        async with semaphore:
            try:
                content = await fetch_article(
                    session, api["url"], api["key"], api["model"], prompt
                )
                df.at[idx, api_name] = content
            except Exception as e:
                print(f"\n  [FAILED] {api_name} idx={idx}: {e}")

            # Save checkpoint every 10 rows
            if (idx + 1) % 10 == 0:
                df.to_csv(output_csv, index=False)
                print(f"  [Checkpoint] Saved at index {idx + 1}")

    async with aiohttp.ClientSession() as session:
        tasks = [
            bounded_fetch(idx, df.at[idx, "Prompt"], session)
            for idx in missing_idx
        ]
        await tqdm.gather(*tasks, desc=f"{api_name} Progress")

    df.to_csv(output_csv, index=False)
    print(f"SUCCESS: [{api_name}] Done -> {output_csv}\n")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found in .env file.")
        return

    input_file = "extra tasks/task2/cnn_dailymail_1k.csv"
    sample_file = "results/extra/task2/generated_sample50.csv"

    if SAMPLE_TEST:
        output_file = sample_file
        print(f"=== TEST MODE: Using only {SAMPLE_SIZE} samples ===\n")
    else:
        output_file = "results/extra/task2/generated_full1000.csv"
        print("=== FULL MODE: Using all 1000 samples ===\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ── Sanity check all models first ──
    print("--- Verifying all models before starting ---")
    valid_apis = []
    for api in my_apis:
        ok = await test_single_model(api)
        if ok:
            valid_apis.append(api)
    print(f"--- {len(valid_apis)}/{len(my_apis)} models passed ---\n")

    if not valid_apis:
        print("ERROR: No models available. Aborting.")
        return

    # ── Load or resume ──
    if os.path.exists(output_file):
        print(f"--> Resuming from existing file: {output_file}")
        df = pd.read_csv(output_file)
        df.fillna("", inplace=True)

    else:
        print(f"--> Starting fresh from: {input_file}")
        df = pd.read_csv(input_file)

        if SAMPLE_TEST:
            df = df.head(SAMPLE_SIZE).copy()
            print(f"--> Sliced to first {SAMPLE_SIZE} rows for test.\n")

        else:
            # ── Merge old 50-sample test results to avoid re-generating ──
            if os.path.exists(sample_file):
                print(f"--> Merging results from old test file: {sample_file}")
                sample_df = pd.read_csv(sample_file).fillna("")
                base_cols = {"Prompt", "Human_story"}
                model_cols = [c for c in sample_df.columns if c not in base_cols]

                for col in model_cols:
                    if col not in df.columns:
                        df[col] = ""
                    n = min(len(sample_df), len(df))
                    for i in range(n):
                        val = sample_df.at[i, col]
                        if val != "":
                            df.at[i, col] = val
                    filled = (df[col] != "").sum()
                    print(f"  ✅ [{col}] Imported {filled} rows from test file")
                print(f"--> Merge done. Rows 51-1000 will be generated fresh.\n")
            else:
                print("--> No old test file found. Generating all 1000 rows fresh.\n")

    print(f"Models to run: {[api['name'] for api in valid_apis]}\n")

    for api in valid_apis:
        await process_api(df, api, output_file)

    print("\n=== ALL MODELS FINISHED ===")
    print(f"Output saved to: {output_file}")
    print(f"Total rows: {len(df)}")

    # Summary of what was generated
    cols = [api["name"] for api in valid_apis]
    for col in cols:
        if col in df.columns:
            filled = (df[col] != "").sum()
            print(f"  {col}: {filled}/{len(df)} articles generated")


if __name__ == "__main__":
    asyncio.run(main())
    os.system('notify-send "APIs Task is Done"')