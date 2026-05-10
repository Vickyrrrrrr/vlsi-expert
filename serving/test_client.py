"""SiliconSmith AI — Minimal test client.

Run this on your LOCAL PC while the SSH tunnel is open:
    ssh -L 8000:127.0.0.1:8000 root@YOUR_DROPLET_IP

Then:
    python serving/test_client.py
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",  # vLLM ignores this value
)

MODEL = "/app/vlsi-moe-yarn"

EXAMPLE_PROMPTS = [
    "You are a VLSI expert. What are the main challenges in designing a 5nm SoC for high-performance computing?",
    "Explain clock-domain crossing risks in a mixed-signal SoC and how to mitigate them.",
    "Review the concept of hold-time violations in synchronous digital design and provide a structured fix strategy.",
]

if __name__ == "__main__":
    print("SiliconSmith AI — Test Client")
    print(f"Model: {MODEL}")
    print("=" * 60)

    # Health check
    models = client.models.list()
    print(f"Available models: {[m.id for m in models.data]}")
    print("=" * 60)

    # Run first example prompt
    prompt = EXAMPLE_PROMPTS[0]
    print(f"Prompt: {prompt}\n")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are SiliconSmith, an expert VLSI and chip design assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        temperature=0.7,
    )

    print("Response:")
    print(resp.choices[0].message.content)
    print("=" * 60)
    print(f"Tokens used — prompt: {resp.usage.prompt_tokens}, completion: {resp.usage.completion_tokens}")
