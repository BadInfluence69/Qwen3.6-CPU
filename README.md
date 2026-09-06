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
