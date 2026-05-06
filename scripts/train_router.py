#!/usr/bin/env python3
"""
Train a lightweight task router for the VLSI Expert MoE model.

The router maps input prompts to the appropriate expert:
  - Verilog generation tasks → Coder expert
  - Error fixing tasks → Reason/Instruct expert
  - SDC/timing tasks → Reason/Instruct expert

Architecture: Simple classifier (MLP) on top of the merged model's embeddings.
Takes the first ~100 tokens' embeddings, averages them, classifies into 3 experts.

Training data: task labels from the collected dataset (spec → type).
Time: ~30 minutes on MI300X.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────
MERGED_MODEL = str(Path(__file__).parent.parent / "models" / "vlsi-moe-merged" / "merged")
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "vlsi-moe-router"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_DIM = 5120  # Qwen 32B hidden dimension
NUM_EXPERTS = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

# Task → Expert routing rules (keyword-based, will train MLP to learn these)
TASK_PATTERNS = {
    "coder": [
        "generate verilog", "write rtl", "create module", "design a",
        "implement", "synthesize", "verilog code for", "hdl for",
        "register transfer level", "rtl for",
    ],
    "reason": [
        "fix error", "fix the", "debug", "why does", "analyze",
        "explain why", "what is wrong", "correct this", "repair",
        "timing violation", "synthesis failed",
    ],
    "instruct": [
        "sdc constraint", "timing constraint", "create_clock",
        "set_input_delay", "set_output_delay", "clock period",
        "false path", "multi cycle", "set_clock_uncertainty",
    ],
}


class TaskRouter(nn.Module):
    """Lightweight classifier that routes prompts to experts."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_experts: int = NUM_EXPERTS):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, num_experts),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Average over sequence length → single vector per batch item
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)


def classify_task(text: str) -> int:
    """Simple keyword-based classifier to generate training labels.
    Returns expert index: 0=coder, 1=reason, 2=instruct"""
    text_lower = text.lower()
    for expert_idx, (expert_name, patterns) in enumerate(TASK_PATTERNS.items()):
        for pattern in patterns:
            if pattern in text_lower:
                return expert_idx
    return 0  # Default to coder


def generate_training_data() -> tuple:
    """Generate training data: (text, expert_label) pairs."""
    data = []
    labels = []

    # From collected dataset
    data_path = Path(__file__).parent.parent / "data" / "train_pairs.jsonl"
    if data_path.exists():
        with open(data_path) as f:
            for line in f:
                item = json.loads(line)
                spec = item.get("spec", "")
                if spec:
                    label = classify_task(spec)
                    data.append(spec)
                    labels.append(label)

    # Manual examples to balance classes
    manual_examples = [
        ("Generate Verilog for an 8-bit ALU", 0),
        ("Write RTL for a SPI master controller", 0),
        ("Create a 32-bit pipelined multiplier module", 0),
        ("Fix this Verilog error: undeclared signal", 1),
        ("Why does synthesis fail with 'no matching module'?", 1),
        ("Analyze the timing violation on this path", 1),
        ("Generate SDC constraints with create_clock", 2),
        ("Write timing constraints for 500MHz operation", 2),
        ("set_input_delay for the data_in port", 2),
    ]
    for text, label in manual_examples:
        data.append(text)
        labels.append(label)

    return data, labels


def main():
    print("=" * 60)
    print("  VLSI Expert — Task Router Training")
    print(f"  Merged model: {MERGED_MODEL}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
    print()

    # Generate training data
    print("[1/3] Generating training data...")
    texts, labels = generate_training_data()

    # Balance classes
    from collections import Counter
    label_counts = Counter(labels)
    print(f"  Samples: {len(texts)} total, distribution: {dict(label_counts)}")

    # Load merged model for embeddings
    print("[2/3] Loading merged model for embeddings (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MERGED_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Extract embeddings for training
    print("  Extracting embeddings...")
    tokenizer.pad_token = tokenizer.eos_token  # Required for padding
    embeddings = []
    for i, text in enumerate(texts):
        # Pad all inputs to same length to avoid empty tensor issues
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            # Average over non-padding tokens only
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(hidden.cpu())

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(texts)}")

    # Train router
    print("[3/3] Training task router...")
    router = TaskRouter().to(model.device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    X = torch.cat(embeddings, dim=0)
    y = torch.tensor(labels).to(model.device)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i in range(0, len(X), BATCH_SIZE):
            batch_x = X[i : i + BATCH_SIZE].to(model.device)
            batch_y = y[i : i + BATCH_SIZE]

            optimizer.zero_grad()
            logits = router(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Accuracy
        with torch.no_grad():
            preds = router(X).argmax(dim=1)
            accuracy = (preds == y).float().mean().item()

        print(f"  Epoch {epoch + 1}: loss={total_loss:.3f}, accuracy={accuracy:.2%}")

    # Save
    torch.save(router.state_dict(), str(OUTPUT_DIR / "router.pt"))

    # Save expert mapping
    expert_map = {
        0: "coder (Verilog generation)",
        1: "reason (error analysis, debugging)",
        2: "instruct (SDC, constraints, timing)",
    }
    with open(OUTPUT_DIR / "expert_map.json", "w") as f:
        json.dump(expert_map, f, indent=2)

    print(f"\n  Router saved: {OUTPUT_DIR / 'router.pt'}")
    print(f"  Expert map: {OUTPUT_DIR / 'expert_map.json'}")
    print("  Done!")


if __name__ == "__main__":
    main()
