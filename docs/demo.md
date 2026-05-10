# SiliconSmith AI — Demo & Example Outputs

## Setup

Before running these examples:
1. Start vLLM server: `bash serving/launch.sh` (inside the ROCm container)
2. Open SSH tunnel: `ssh -L 8000:127.0.0.1:8000 root@YOUR_DROPLET_IP`
3. Install client: `pip install openai`

---

## Example 1: Clock Domain Crossing Analysis

**Prompt:**
```
Explain clock-domain crossing risks in a mixed-signal SoC and provide a structured mitigation strategy.
```

**Expected output:** The model should explain metastability, synchronizer chains, handshaking protocols, and provide a structured list of CDC verification steps.

---

## Example 2: RTL Design Review

**Prompt:**
```
Review this synchronous FIFO design for potential hold-time violations and suggest fixes:

module sync_fifo #(parameter DEPTH=16, WIDTH=8) (
    input clk, rst_n,
    input wr_en, rd_en,
    input [WIDTH-1:0] din,
    output reg [WIDTH-1:0] dout,
    output full, empty
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] wr_ptr, rd_ptr;
    ...
endmodule
```

**Expected output:** Analysis of pointer management, potential timing paths, and specific Verilog fixes.

---

## Example 3: Long-Context Document Q&A

The 262K token context window allows pasting an entire datasheet or design spec and asking questions:

**Prompt:**
```
[Paste 50,000 tokens of SoC datasheet here]

Based on the power specification above, what is the recommended decoupling capacitor placement strategy for the digital core domain?
```

---

## Running the Demo

```python
python serving/test_client.py
```

This runs three example prompts and prints the model's responses with token usage stats.
