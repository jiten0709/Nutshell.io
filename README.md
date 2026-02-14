# Nutshell.io (In progress)

- The Problem: The "AI Hype" cycle has created a fragmentation of information. High-signal updates are buried in 15+ daily newsletters (TLDR, The Neuron, etc.), leading to "Tab Fatigue" and duplicate consumption.

- The Solution: A unified, deduplicated, and semantically indexed "Command Center" that transforms 1,000+ newsletter sentences into 10-15 actionable, technical insights per day, personalized to your specific tech stack.

# Core Functionalities (The "Moat" Features)

1. The "Smart Deduper" (Semantic Merging)
   Instead of showing you three different summaries of the "Sora Release," the system identifies they are the same event using Cosine Similarity. It then merges the insights:
   Newsletter A mentions the release date.
   Newsletter B mentions the technical architecture.
   SignalFlow presents one entry with all unique facts.

2. The "Paper-to-Verify" Engine
   When a newsletter claims a new model is "SOTA," the backend automatically triggers a worker to:
   Scrape the linked ArXiv paper or GitHub README.
   Cross-reference the newsletter's claim against the actual benchmarks.
   Flag discrepancies (e.g., "Newsletter says 90% accuracy, Paper says 82%").

3. Personal "Signal" Profiles
   The app doesn't just show "Top News." It ranks news based on your Engineer Persona:
   The Researcher: Focuses on new architectures and loss functions.
   The Builder: Focuses on SDK updates, API pricing changes, and framework releases (LangChain, LlamaIndex).

# Technical Architecture & Tech-Stack

# Made with ❤️ by Jiten !!!
