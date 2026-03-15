"""SSD Phase 2: Correctness verification.

Compare SSD (speculative) vs AR (autoregressive) outputs token-by-token
under greedy decoding (temperature=0). They should be identical.

Usage (single GPU mode):
  1. Start AR server on port 30001, run: python tests/ssd_correctness_test.py collect-ar
  2. Kill AR, start SSD server on port 30002, run: python tests/ssd_correctness_test.py collect-ssd
  3. Compare: python tests/ssd_correctness_test.py compare

Usage (two GPU mode):
  1. Start AR on port 30001, SSD on port 30002
  2. python tests/ssd_correctness_test.py
"""

import json
import os
import requests
import sys
import time

AR_URL = "http://localhost:30001"
SSD_URL = "http://localhost:30002"
AR_RESULTS_FILE = "/tmp/ssd_test_ar_results.json"
SSD_RESULTS_FILE = "/tmp/ssd_test_ssd_results.json"

PROMPTS = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a Python function to check if a number is prime:",
    "The theory of relativity states that",
    "In machine learning, gradient descent is",
    "The difference between TCP and UDP is",
    "What happens when you type a URL into a browser?",
    "Describe the water cycle in detail:",
    "The Fibonacci sequence is defined as",
    "Explain how a hash table works:",
    "The three laws of thermodynamics are",
    "In distributed systems, the CAP theorem states",
    "Write a SQL query to find duplicate rows:",
    "The human brain contains approximately",
    "Photosynthesis is the process by which",
    "The Big Bang theory suggests that",
    "In object-oriented programming, polymorphism means",
    "The fastest sorting algorithm for general cases is",
    "Explain how DNS resolution works step by step:",
    "The periodic table organizes elements by",
    "Newton's three laws of motion are",
    "A neural network consists of",
    "The difference between stack and heap memory is",
    "In economics, inflation is caused by",
    "The process of mitosis involves",
    "Explain the concept of recursion with an example:",
    "The speed of light in vacuum is",
    "In cryptography, RSA algorithm works by",
    "The main difference between Python and C++ is",
    "Climate change is primarily caused by",
    "A binary search tree has the property that",
    "The TCP three-way handshake involves",
    "Explain how garbage collection works in Java:",
    "The human heart has four chambers:",
    "In statistics, the central limit theorem states",
    "The difference between HTTP and HTTPS is",
    "Describe how a compiler works:",
    "The theory of evolution by natural selection",
    "In databases, ACID properties stand for",
    "The electromagnetic spectrum includes",
    "Explain the concept of virtual memory:",
    "The Pythagorean theorem states that",
    "In networking, a subnet mask is used to",
    "The process of protein synthesis involves",
    "Describe the architecture of a CPU:",
    "The laws of supply and demand state that",
    "In graph theory, Dijkstra's algorithm finds",
    "The difference between DNA and RNA is",
    "Explain how blockchain technology works:",
    "The universal gravitational constant is",
    "Once upon a time, in a land far away,",
    "Dear hiring manager, I am writing to express",
    "Step 1: Preheat the oven to 350°F.",
    "The year was 2050, and humanity had finally",
    "According to recent research published in Nature,",
    "To solve this differential equation, we first",
    "The United Nations was established in",
    "In functional programming, a monad is",
    "The chemical formula for water is H2O, which means",
    "During the Renaissance period, artists such as",
    "The main components of a operating system kernel are",
    "In reinforcement learning, the agent learns by",
    "The Amazon rainforest, often called the lungs of Earth,",
    "To implement a red-black tree, we need to maintain",
    "The French Revolution began in 1789 when",
    "In quantum mechanics, Heisenberg's uncertainty principle",
    "The World Wide Web was invented by Tim Berners-Lee",
    "A convolutional neural network processes images by",
    "The human genome contains approximately",
    "In microeconomics, marginal cost is defined as",
    "The Krebs cycle, also known as the citric acid cycle,",
    "To configure a Kubernetes cluster, you need to",
    "Shakespeare's most famous play, Hamlet, tells the story of",
    "In linear algebra, eigenvalues and eigenvectors",
    "The Mars rover Curiosity discovered evidence of",
    "A B+ tree differs from a B tree in that",
    "The principles of REST API design include",
    "During World War II, the Allied forces",
    "In category theory, a functor is a mapping between",
    "The human immune system fights pathogens through",
    "To optimize a SQL query, you should consider",
    "The Standard Model of particle physics describes",
    "In game theory, the Nash equilibrium occurs when",
    "The Great Wall of China was built to",
    "A distributed hash table provides",
    "The Doppler effect explains why",
    "In software engineering, the SOLID principles are",
    "The discovery of penicillin by Alexander Fleming",
    "MapReduce is a programming model that",
    "The theory of plate tectonics explains how",
    "In probability theory, Bayes' theorem relates",
    "The Apollo 11 mission successfully landed on the Moon",
    "To implement consensus in a distributed system,",
    "The Turing test was proposed by Alan Turing to",
    "In organic chemistry, functional groups determine",
    "The Internet Protocol suite consists of",
    "During the Industrial Revolution, the invention of",
    "A transformer model uses self-attention to",
    "The principles of thermodynamics govern",
    "In abstract algebra, a group is a set equipped with",
    "The discovery of DNA's double helix structure by",
    "To build a fault-tolerant system, engineers use",
]

MAX_TOKENS = 64


def get_completion(url, prompt, max_tokens=MAX_TOKENS):
    r = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["text"]
    tokens_used = data["usage"]["completion_tokens"]
    return text, tokens_used


def collect(url, name, output_file):
    print(f"Collecting {name} results from {url}...")
    results = []
    for i, prompt in enumerate(PROMPTS):
        short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        try:
            text, tokens = get_completion(url, prompt)
            results.append({"prompt": prompt, "text": text, "tokens": tokens})
            print(f"  [{i+1:3d}/{len(PROMPTS)}] {tokens:3d} tokens  {short}")
        except Exception as e:
            results.append({"prompt": prompt, "text": None, "error": str(e)})
            print(f"  [{i+1:3d}/{len(PROMPTS)}] ERROR  {short}: {e}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {output_file}")


def compare():
    print("=" * 60)
    print("SSD Phase 2: Correctness Comparison")
    print("=" * 60)

    with open(AR_RESULTS_FILE) as f:
        ar_results = json.load(f)
    with open(SSD_RESULTS_FILE) as f:
        ssd_results = json.load(f)

    assert len(ar_results) == len(ssd_results), "Result count mismatch"

    passed = 0
    failed = 0
    errors = []

    for i, (ar, ssd) in enumerate(zip(ar_results, ssd_results)):
        prompt = ar["prompt"]
        short = prompt[:50] + "..." if len(prompt) > 50 else prompt

        if ar.get("error") or ssd.get("error"):
            failed += 1
            print(f"  [{i+1:3d}] ERROR  {short}")
            errors.append({"prompt": prompt, "ar_error": ar.get("error"), "ssd_error": ssd.get("error")})
            continue

        ar_text = ar["text"]
        ssd_text = ssd["text"]

        if ar_text == ssd_text:
            passed += 1
            print(f"  [{i+1:3d}] PASS  ({ar['tokens']} tokens)")
        else:
            failed += 1
            div_pos = 0
            for j in range(min(len(ar_text), len(ssd_text))):
                if ar_text[j] != ssd_text[j]:
                    div_pos = j
                    break
            else:
                div_pos = min(len(ar_text), len(ssd_text))

            print(f"  [{i+1:3d}] FAIL  {short}")
            print(f"       Diverges at char {div_pos}")
            print(f"       AR:  ...{ar_text[max(0,div_pos-20):div_pos+40]}...")
            print(f"       SSD: ...{ssd_text[max(0,div_pos-20):div_pos+40]}...")
            errors.append({
                "prompt": prompt,
                "ar_text": ar_text,
                "ssd_text": ssd_text,
                "div_pos": div_pos,
            })

    print("\n" + "=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {len(ar_results)}")
    print(f"Accuracy: {passed / len(ar_results) * 100:.1f}%")
    print("=" * 60)

    if errors:
        err_file = "/tmp/ssd_correctness_errors.json"
        with open(err_file, "w") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Failed cases saved to {err_file}")

    return 0 if failed == 0 else 1


def main():
    if len(sys.argv) < 2:
        # Full mode: both servers running
        collect(AR_URL, "AR", AR_RESULTS_FILE)
        collect(SSD_URL, "SSD", SSD_RESULTS_FILE)
        return compare()

    cmd = sys.argv[1]
    if cmd == "collect-ar":
        collect(AR_URL, "AR", AR_RESULTS_FILE)
    elif cmd == "collect-ssd":
        collect(SSD_URL, "SSD", SSD_RESULTS_FILE)
    elif cmd == "compare":
        return compare()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python ssd_correctness_test.py [collect-ar|collect-ssd|compare]")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
