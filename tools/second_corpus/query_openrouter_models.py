import urllib.request
import json

url = "https://openrouter.ai/api/v1/models"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

models = data.get("data", [])
print(f"Total models: {len(models)}")

targets = ["gemma", "llama", "mistral", "qwen", "yi"]
matches = {t: [] for t in targets}

for m in models:
    mid = m.get("id", "")
    for t in targets:
        if t in mid.lower():
            matches[t].append(mid)

for t in targets:
    print(f"\n--- {t.upper()} ({len(matches[t])} models) ---")
    for mid in sorted(matches[t]):
        # print if relevant to small / 7b / 8b / 9b / 72b / yi
        if any(sub in mid.lower() for sub in ["2", "3", "7b", "8b", "9b", "72b", "nemo", "yi"]):
            print(" ", mid)
