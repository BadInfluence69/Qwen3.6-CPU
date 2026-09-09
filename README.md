# Quinn 3.6-CPU

A Python-based inference workflow tailored specifically for running local LLMs on modern, high-core CPUs without requiring a dedicated GPU.

## Overview

Hardware prices, scalper markups, and data center demand have pushed decent graphics cards out of reach for many. **Quinn 3.6-CPU** is built for users who are stuck without a usable discrete GPU—whether it died, got priced out, or was never installed—but still have a strong multi-core processor and plenty of RAM (e.g., 32 GB).

This repository optimizes local model execution to squeeze maximum performance directly out of system RAM and CPU instruction sets (AVX2 / AVX-512), making open-source models viable on Everyday and High-Performance CPU setups.

## Key Features

* **Zero GPU Requirement:** Built from the ground up to operate strictly on host CPU and system memory.
* **Optimized Execution:** Leverages multithreading and quantized memory models to maintain solid tokens-per-second generation speeds.
* **Low Memory Overhead:** Structured to run efficiently inside a standard 32 GB RAM budget.

## System Requirements

* **OS:** Windows / Linux / macOS
* **CPU:** Modern 6+ core processor (Intel Core i5/i7/i9 10th Gen+, AMD Ryzen 5/7/9 3000 series+) with AVX2 support
* **RAM:** 16 GB minimum (32 GB strongly recommended)
* **Python:** 3.10 or higher

## Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Quinn-3.6-CPU.git](https://github.com/YOUR_USERNAME/Quinn-3.6-CPU.git)
   cd Quinn-3.6-CPU

   # QwenBox setup — i9-9900K, 32 GB, Windows 10

One script. It downloads the model if it isn't already on disk, loads it, and
serves the chat UI. No API key, no separate server process.

## Install

```
pip install flask huggingface_hub requests beautifulsoup4
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

That second line matters on Windows. Plain `pip install llama-cpp-python` tries
to compile llama.cpp from source, which needs Visual Studio build tools and
takes a long time. The `--extra-index-url` points pip at a prebuilt CPU wheel
instead, and it installs in seconds. Your 9900K is Coffee Lake — AVX2, no
AVX-512 — and the CPU wheel covers that.

## Run

```
python qwenbox.py
```

First run downloads ~18 GB to `E:\models` with a progress bar, then loads it.
Every run after that prints `Found model:` and skips straight to loading.

Open <http://127.0.0.1:5005>.

## Disk

| Item | Size | Where |
|---|---|---|
| Qwen3.6-35B-A3B IQ4_XS | ~18 GB | `E:\models` (spinning drive) |
| Flask + llama-cpp-python + hub | ~120 MB | SSD |

Your SSD only takes the ~120 MB of Python packages. `MODEL_DIR` at the top of
the script controls where the big file goes — change it if E: isn't the drive
you want.

The script uses `use_mmap=True`, so the OS maps the file rather than copying
18 GB into RAM. First load off a hard drive takes a few minutes. Later loads
are fast while the file is still in the page cache.

## Settings at the top of the script

```python
MODEL_DIR = r"E:\models"
FILENAME  = "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
N_THREADS = 8
N_CTX     = 16384
```

- `N_THREADS = 8` — your physical core count. Don't set 16. Hyperthreading
  contends for the same memory bandwidth and usually makes it slower.
- `N_CTX = 16384` — enough to paste a file and get a full answer back. The
  model card advertises 262K; ignore that on CPU. Drop to 8192 if RAM is tight.
- Changing either needs a restart. Temperature and reply limit are in the
  Settings dialog and take effect on the next message.

## Sampling for code

Ships at temperature 0.6 / top_p 0.95 / top_k 20, reply limit 4096.

- For "write this function, get it right," drop temperature to 0.2–0.3.
- Qwen ran their own SWE-bench evaluation at temperature 1.0 / top_p 0.95 /
  top_k 20, so higher isn't wrong for open-ended work.
- At ~10 tok/s, a full 4096-token reply is roughly seven minutes. That's what
  the Stop button is for — it keeps whatever already arrived.

## Web search

Press the **Web** button next to Send before sending a message. It turns green
when it's on, and stays on until you press it again — so ordinary coding
questions never touch the network unless you ask them to.

When it's on, QwenBox searches DuckDuckGo's lite endpoint, pulls the readable
text off the top result, and hands both to the model before it answers. Sources
appear above the reply as clickable links.

No API key, no Docker, no browser automation. Just an HTTP request.

Test search on its own, without waiting on the model:

    http://127.0.0.1:5005/api/search?q=qwen3.6+release+notes

If that returns JSON results, search works. If it errors, the problem is the
network path, not the model.

Result count and how much page text to pull are `web_results` and `web_chars`
at the top of the script. Each search adds a few thousand tokens to the prompt,
and prompt processing on CPU is not free — expect a longer pause before the
first token when Web is on. That's the cost of it, and it's why it's a toggle
rather than always-on.

If DuckDuckGo changes their HTML someday and results stop parsing, `web_search`
is one function near the top of the file.

## The one free speedup

**Check XMP in your ASRock BIOS.** Token generation on CPU is bound by memory
bandwidth, not clock speed. If your DDR4 is sitting at the 2133 MHz JEDEC
default instead of its rated speed, you're losing 30–40%. It's under OC Tweaker.
You already run a CPU overclock so it may well be on, but it's worth confirming.

Expect roughly 8–14 tok/s once it's right.

## If something breaks

| What you see | What it means |
|---|---|
| `Missing llama-cpp-python` | Use the `--extra-index-url` install line above |
| Compiler errors during pip install | Same — you got the source build |
| Hangs on first run with no output | Download in progress; check E: for a growing file |
| Windows crawls once loaded | Not enough free RAM — lower `N_CTX` to 8192 |
| 2–3 tok/s | XMP is probably off, or `N_THREADS` is set too high |
| "Already generating" | One request at a time. Press Stop, then resend |
