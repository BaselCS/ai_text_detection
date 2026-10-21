"""Task 1 endpoint / checkpoint availability audit.

Writes results/phase3/T1_endpoint_audit_snapshot.txt with:

* the OpenRouter catalogue family listing;
* an endpoints probe for every CNN generator slot (six), including
  ``yi-large`` and the exact ``mistral-7b-instruct-v0.3`` id;
* a Hugging Face model-card probe for the four open-weight training
  checkpoints, so the "still publicly available" claim has an artifact.
"""

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "results/phase3/T1_endpoint_audit_snapshot.txt"
TIMEOUT = 30
UA = {"User-Agent": "phase3-endpoint-audit/1.0"}

OPENROUTER_PROBES = [
    ("qwen_2_72b", "qwen/qwen-2-72b-instruct"),
    ("gemma_2_9b", "google/gemma-2-9b-it"),
    ("llama_3_8b", "meta-llama/llama-3-8b-instruct"),
    ("mistral_7b_v0.3", "mistralai/mistral-7b-instruct-v0.3"),
    ("mistral_7b_legacy", "mistralai/mistral-7b-instruct"),
    ("yi_large", "01-ai/yi-large"),
    ("gpt_4o", "openai/gpt-4o"),
]

HF_PROBES = [
    ("Gemma-2-9B slot", "google/gemma-2-9b-it"),
    ("Llama-8B slot", "meta-llama/Meta-Llama-3-8B-Instruct"),
    ("Mistral-7B slot", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("Qwen-2-72B slot", "Qwen/Qwen2-72B-Instruct"),
]

FAMILIES = ["gemma", "llama", "mistral", "qwen", "yi"]


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main() -> None:
    lines = [
        "Task 1 endpoint / checkpoint audit snapshot",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "=== OpenRouter public model catalogue query ===",
    ]

    status, catalogue = get_json("https://openrouter.ai/api/v1/models")
    models = catalogue.get("data", [])
    lines.append(f"HTTP {status}; total models: {len(models)}")
    for fam in FAMILIES:
        ids = sorted(m["id"] for m in models if fam in m["id"].lower())
        lines.append("")
        lines.append(f"--- {fam.upper()} ({len(ids)} models) ---")
        lines.extend(f"  {i}" for i in ids)

    lines += ["", "=== OpenRouter endpoint availability probes ==="]
    lines.append("An endpoint count of 0 means the model id is catalogued but served")
    lines.append("by no provider; the legacy hosted route is therefore unavailable.")
    for label, model_id in OPENROUTER_PROBES:
        try:
            st, payload = get_json(
                f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
            )
            data = payload.get("data", {})
            eps = data.get("endpoints") or []
            providers = sorted({e.get("provider_name", "?") for e in eps})
            verdict = "AVAILABLE" if eps else "NO ENDPOINTS"
            lines.append(
                f"[{verdict}] {label:18s} {model_id:38s} HTTP {st} "
                f"endpoints={len(eps)} providers={providers}"
            )
        except urllib.error.HTTPError as e:
            lines.append(f"[HTTP {e.code}] {label:18s} {model_id:38s} probe failed")
        except Exception as e:  # noqa: BLE001
            lines.append(f"[ERROR] {label:18s} {model_id:38s} {e!r}")

    lines += ["", "=== Hugging Face model-card availability ==="]
    lines.append("Open-weight repositories are checked directly; `gated` repos are")
    lines.append("publicly downloadable after accepting the licence on the model card.")
    for label, repo in HF_PROBES:
        try:
            st, payload = get_json(f"https://huggingface.co/api/models/{repo}")
            lines.append(
                f"[HTTP {st}] {label:18s} {repo:38s} "
                f"gated={payload.get('gated')} private={payload.get('private')} "
                f"lastModified={payload.get('lastModified')} "
                f"downloads={payload.get('downloads')}"
            )
        except urllib.error.HTTPError as e:
            lines.append(f"[HTTP {e.code}] {label:18s} {repo:38s} not retrievable")
        except Exception as e:  # noqa: BLE001
            lines.append(f"[ERROR] {label:18s} {repo:38s} {e!r}")
    lines += [
        "",
        "Note: gpt-4o is a proprietary OpenAI checkpoint and has no open weights;",
        "the CNN corpus used gpt-4o-mini, which is a different OpenAI checkpoint.",
        "",
    ]

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
