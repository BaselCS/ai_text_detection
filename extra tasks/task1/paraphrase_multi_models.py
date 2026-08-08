import os
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt

# ──────────────────────────────────────────────
# Load API keys from .env
# ──────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ──────────────────────────────────────────────
# CONFIG — mirrors helper/paraphrasing_attacks.ipynb exactly
# ──────────────────────────────────────────────
INPUT_FILE = "extra tasks/task1/limited_articles.csv"  # same as ../data/limited_articles.csv in notebook
NUM_SAMPLES = 2100                                      # same as NUM_SAMPLES in notebook (all rows)

# Output directory
OUTPUT_DIR = "data/processed"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"

my_apis = [
    {
        "name": "gemma_2_9b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "google/gemma-3-12b-it",
        "output_file": f"{OUTPUT_DIR}/raw_paraphrased_gemma2.csv",
    },
    {
        "name": "qwen_2_72b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "qwen/qwen-2.5-72b-instruct",
        "output_file": f"{OUTPUT_DIR}/raw_paraphrased_qwen2.csv",
    },
    {
        "name": "llama_3_8b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "meta-llama/llama-3.1-8b-instruct",
        "output_file": f"{OUTPUT_DIR}/raw_paraphrased_llama3.csv",
    },
    {
        "name": "mistral_7b",
        "url": OPENROUTER_URL,
        "key": OPENROUTER_API_KEY,
        "model": "mistralai/mistral-nemo",
        "output_file": f"{OUTPUT_DIR}/raw_paraphrased_mistral7b.csv",
    },
    {
        "name": "gpt-4o-mini",
        "url": OPENAI_URL,
        "key": OPENAI_API_KEY,
        "model": "gpt-4o-mini",
        "output_file": f"{OUTPUT_DIR}/raw_paraphrased_gpt4o_mini.csv",
    },
]


# ──────────────────────────────────────────────
# Paraphrase prompt — IDENTICAL to notebook
# (helper/paraphrasing_attacks.ipynb, Cell 23)
# ──────────────────────────────────────────────
def build_prompt(text: str) -> str:
    return (
        "Act as a New York Times writer. Paraphrase the following text while keeping \n"
        "the core meaning intact. Use natural, slightly varied sentence structures \n"
        "to make it look human-written. Do not add any introductory remarks.\n"
        "\n"
        f"Original Text: {text}"
    )


# ──────────────────────────────────────────────
# Core fetch with retry logic
# ──────────────────────────────────────────────
@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(4))
async def fetch_paraphrase(session, api_url, api_key, model, text):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-text-detection",
        "X-Title": "AI Text Detection Research",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(text)}],
        "temperature": 0.7,
        "max_tokens": 1500,
    }

    async with session.post(api_url, headers=headers, json=payload) as response:
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
# Process one API model — mirrors notebook logic
# Output columns: Writer, Original_Text, Paraphrased_Text
# Writer label:   "<original_writer>_paraphrased"  (same as notebook)
# ──────────────────────────────────────────────
async def process_api(input_df, api):
    api_name = api["name"]
    output_file = api["output_file"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize full output dataframe aligned with input_df
    df_out = pd.DataFrame({
        "Writer": input_df["Writer"] + "_paraphrased",
        "Original_Text": input_df["Article"],
        "Paraphrased_Text": ""
    })

    if os.path.exists(output_file):
        print(f"--> [{api_name}] Resuming from: {output_file}")
        try:
            existing_df = pd.read_csv(output_file)
            existing_df.fillna("", inplace=True)
            n = min(len(df_out), len(existing_df))
            if "Paraphrased_Text" in existing_df.columns:
                df_out.loc[:n-1, "Paraphrased_Text"] = existing_df.loc[:n-1, "Paraphrased_Text"].values
            print(f"    Loaded {n} rows from existing file.")
        except Exception as e:
            print(f"    Failed to read existing file: {e}")

    missing_idx = df_out[df_out["Paraphrased_Text"] == ""].index

    if len(missing_idx) == 0:
        print(f"[{api_name}] Already completed. Skipping...")
        return

    print(f"[{api_name}] Paraphrasing {len(missing_idx)} articles...")

    semaphore = asyncio.Semaphore(5)

    async def bounded_fetch(idx, text, session):
        async with semaphore:
            try:
                paraphrased = await fetch_paraphrase(
                    session, api["url"], api["key"], api["model"], text
                )
                df_out.at[idx, "Paraphrased_Text"] = paraphrased
            except Exception as e:
                print(f"\n  [FAILED] {api_name} idx={idx}: {e}")

            # Checkpoint every 10 rows
            if (idx + 1) % 10 == 0:
                df_out.to_csv(output_file, index=False)

    async with aiohttp.ClientSession() as session:
        tasks = [
            bounded_fetch(idx, df_out.at[idx, "Original_Text"], session)
            for idx in missing_idx
        ]
        await tqdm.gather(*tasks, desc=f"{api_name} Progress")

    df_out.to_csv(output_file, index=False)
    filled = (df_out["Paraphrased_Text"] != "").sum()
    print(f"SUCCESS: [{api_name}] {filled}/{len(df_out)} articles done -> {output_file}\n")


# ──────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────
async def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found in .env file.")
        return
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in .env file.")
        return

    # ── Load source dataset — same as notebook ──
    print(f"--> Loading: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Total rows: {len(df)}")
    print(df["Writer"].value_counts().to_string())
    print()

    # Use all rows up to NUM_SAMPLES (same as notebook)
    df = df.head(NUM_SAMPLES).copy()
    print(f"--> Using {len(df)} articles (NUM_SAMPLES={NUM_SAMPLES})\n")

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

    print(f"Models to run: {[api['name'] for api in valid_apis]}\n")

    for api in valid_apis:
        await process_api(df, api)

    print("=== ALL MODELS FINISHED ===")
    for api in valid_apis:
        print(f"  [{api['name']}] output: {api['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())
    os.system('notify-send "Task 1 Paraphrasing Done"')
