#!/usr/bin/env python3
"""
Simple OpenAI-compatible backend benchmark tool.
Measures tokens/sec, TTFT, and inter-token latency.
"""

import argparse
import time
import statistics
import random
from openai import OpenAI, APIConnectionError

# Diverse prompts to avoid cache duplication issues
BENCHMARK_PROMPTS = [
    "Explain how neural networks learn through backpropagation.",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in plants.",
    "How does a compiler transform source code into machine code?",
    "Explain the concept of entropy in thermodynamics.",
    "What are the main principles of object-oriented programming?",
    "How do vaccines train the immune system?",
    "Describe the architecture of a modern CPU.",
    "What causes the seasons on Earth?",
    "Explain how public key cryptography works.",
    "What are the stages of the software development lifecycle?",
    "How does natural selection drive evolution?",
    "Describe the structure and function of DNA.",
    "What is the difference between machine learning and deep learning?",
    "Explain how HTTP requests and responses work.",
    "What are the fundamental forces in physics?",
    "How do databases maintain ACID properties?",
    "Describe the water cycle and its importance.",
    "What is the role of mitochondria in cells?",
    "Explain the concept of recursion in programming.",
    "How does the human brain process visual information?",
    "What are the key features of functional programming?",
    "Describe how earthquakes occur along fault lines.",
    "What is the difference between REST and GraphQL APIs?",
    "Explain how batteries store and release energy.",
]


def benchmark_request(client, model, prompt, max_tokens, temperature, top_p, top_k=None, rep=None, freq=None, quiet=False, debug=False, stream=True):
    """Run a single request and measure performance."""
    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    tokens = 0
    all_content = []

    # Build extra params for non-standard OpenAI params
    # extra_body contents get merged into request body by the OpenAI client
    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k
    if rep is not None:
        extra_body["repetition_penalty"] = rep
    if freq is not None:
        extra_body["frequency_penalty"] = freq

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        **({"extra_body": extra_body} if extra_body else {})
    )

    if stream:
        usage_tokens = None
        for chunk in response:
            if debug:
                delta_content = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                print(f"  DEBUG chunk: choices={len(chunk.choices) if chunk.choices else 0}, "
                      f"finish={chunk.choices[0].finish_reason if chunk.choices else None}, "
                      f"content={delta_content!r}")
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                all_content.append(content)
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                else:
                    token_times.append(now)
                tokens += 1  # Count chunks for ITL calculation
            # Capture usage from final chunk if available
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_tokens = chunk.usage.completion_tokens
        # Use actual token count from usage if available
        if usage_tokens is not None:
            tokens = usage_tokens
    else:
        # Non-streaming mode - all tokens arrive at once
        if response.choices and response.choices[0].message.content:
            first_token_time = time.perf_counter()  # Response received
            content = response.choices[0].message.content
            all_content.append(content)
            # Get tokens from response usage or estimate
            if hasattr(response, 'usage') and response.usage:
                tokens = response.usage.completion_tokens
            else:
                # Rough estimate: ~4 chars per token
                tokens = len(content) // 4
            if debug:
                print(f"  DEBUG non-stream: tokens={tokens}, content={content[:100]!r}...")

    if tokens == 0 and not quiet:
        print(f"  WARNING: 0 tokens! Content chunks received: {len(all_content)}, content: {''.join(all_content)!r}")
        print(f"    PROMPT: {prompt[:60]}...")

    end_time = time.perf_counter()

    # Calculate metrics
    total_time = end_time - start_time
    if stream:
        # Streaming: TTFT is time to first token, generation is time after first token
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        generation_time = end_time - first_token_time if first_token_time else total_time
    else:
        # Non-streaming: total time IS generation time, TTFT = total time
        ttft = total_time * 1000
        generation_time = total_time

    # Inter-token latency (time between tokens after first)
    if len(token_times) > 1:
        itl_samples = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
        itl_ms = statistics.mean(itl_samples) * 1000
    else:
        itl_ms = 0

    tps = tokens / generation_time if generation_time > 0 else 0

    return {
        "tokens": tokens,
        "ttft_ms": ttft,
        "itl_ms": itl_ms,
        "total_time": total_time,
        "generation_time": generation_time,
        "tokens_per_sec": tps,
    }


def run_benchmark(base_url, api_key, model, prompt, max_tokens, runs, warmup, temperature, top_p, top_k=None, rep=None, freq=None, unique_prompts=False, quiet=False, debug=False, stream=True):
    """Run multiple benchmark iterations."""
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Prepare prompts - either use the single prompt or cycle through unique ones
    if unique_prompts:
        prompts = BENCHMARK_PROMPTS.copy()
        random.shuffle(prompts)
        # Extend if we need more than available
        while len(prompts) < warmup + runs:
            prompts.extend(BENCHMARK_PROMPTS)
    else:
        prompts = [prompt] * (warmup + runs)

    prompt_idx = 0

    # Warmup runs
    if warmup > 0:
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            benchmark_request(client, model, prompts[prompt_idx], max_tokens, temperature, top_p, top_k=top_k, rep=rep, freq=freq, quiet=True, debug=debug, stream=stream)
            prompt_idx += 1

    # Benchmark runs
    if quiet:
        print(f"Running benchmark ({runs} runs)...", end="", flush=True)
    else:
        print(f"Running benchmark ({runs} runs)...")
    results = []
    failed_count = 0
    for i in range(runs):
        result = benchmark_request(client, model, prompts[prompt_idx], max_tokens, temperature, top_p, top_k=top_k, rep=rep, freq=freq, quiet=quiet, debug=debug, stream=stream)
        prompt_idx += 1
        results.append(result)
        if quiet:
            if result['tokens'] == 0:
                failed_count += 1
                print("x", end="", flush=True)
            elif (i + 1) % 10 == 0:
                print(".", end="", flush=True)
        else:
            if stream:
                print(f"  Run {i+1}: {result['tokens_per_sec']:.2f} t/s, "
                      f"TTFT: {result['ttft_ms']:.1f}ms, "
                      f"ITL: {result['itl_ms']:.2f}ms, "
                      f"tokens: {result['tokens']}")
            else:
                print(f"  Run {i+1}: {result['tokens_per_sec']:.2f} t/s, "
                      f"total: {result['total_time']*1000:.1f}ms, "
                      f"tokens: {result['tokens']}")
    if quiet:
        print(f" done ({failed_count} failed)")

    return results


def print_summary(results, stream=True):
    """Print summary statistics."""
    # Separate successful vs failed runs
    successful = [r for r in results if r["tokens"] > 0]
    failed = [r for r in results if r["tokens"] == 0]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total runs: {len(results)}")
    print(f"Successful:  {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    print(f"Failed:      {len(failed)} ({100*len(failed)/len(results):.1f}%)")

    if not successful:
        print("\nNo successful runs to analyze!")
        print("=" * 60)
        return

    # Stats from successful runs only
    tps_values = [r["tokens_per_sec"] for r in successful]
    total_time_values = [r["total_time"] * 1000 for r in successful]
    token_counts = [r["tokens"] for r in successful]

    print(f"\nAvg tokens generated: {statistics.mean(token_counts):.1f}")
    print()
    print(f"Tokens/sec:")
    print(f"  Mean:   {statistics.mean(tps_values):.2f}")
    print(f"  Median: {statistics.median(tps_values):.2f}")
    if len(tps_values) > 1:
        print(f"  StdDev: {statistics.stdev(tps_values):.2f}")
        print(f"  P5:     {sorted(tps_values)[int(len(tps_values)*0.05)]:.2f}")
        print(f"  P95:    {sorted(tps_values)[int(len(tps_values)*0.95)]:.2f}")

    if stream:
        # Streaming-specific metrics
        ttft_values = [r["ttft_ms"] for r in successful]
        itl_values = [r["itl_ms"] for r in successful if r["itl_ms"] > 0]

        print()
        print(f"Time to First Token (ms):")
        print(f"  Mean:   {statistics.mean(ttft_values):.1f}")
        print(f"  Median: {statistics.median(ttft_values):.1f}")
        if len(ttft_values) > 1:
            print(f"  P5:     {sorted(ttft_values)[int(len(ttft_values)*0.05)]:.1f}")
            print(f"  P95:    {sorted(ttft_values)[int(len(ttft_values)*0.95)]:.1f}")

        if itl_values:
            print()
            print(f"Inter-Token Latency (ms):")
            print(f"  Mean:   {statistics.mean(itl_values):.2f}")
            print(f"  Median: {statistics.median(itl_values):.2f}")
    else:
        # Non-streaming: show total time
        print()
        print(f"Total Time (ms):")
        print(f"  Mean:   {statistics.mean(total_time_values):.1f}")
        print(f"  Median: {statistics.median(total_time_values):.1f}")
        if len(total_time_values) > 1:
            print(f"  P5:     {sorted(total_time_values)[int(len(total_time_values)*0.05)]:.1f}")
            print(f"  P95:    {sorted(total_time_values)[int(len(total_time_values)*0.95)]:.1f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI-compatible LLM endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local llama-server
  %(prog)s --base-url http://localhost:8000/v1 --model local-model

  # vLLM server  
  %(prog)s --base-url http://localhost:8000/v1 --model Qwen/Qwen2-7B

  # Custom prompt and token count
  %(prog)s --base-url http://localhost:8000/v1 --model llama --max-tokens 256 --runs 10
        """,
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--api-key", default="not-needed",
                        help="API key (default: not-needed)")
    parser.add_argument("--model", default="default",
                        help="Model name (default: default)")
    parser.add_argument("--prompt", 
                        default="Write a detailed explanation of how transformers work in neural networks.",
                        help="Prompt to use for benchmarking")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of benchmark runs (default: 5, use 100+ for stress test)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature (default: 0.7, use 1.0 for GPT-OSS)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (default: None)")
    parser.add_argument("--rep", type=float, default=None,
                        help="Repetition penalty (default: None)")
    parser.add_argument("--freq", type=float, default=None,
                        help="Frequency penalty (default: None)")
    parser.add_argument("--unique-prompts", action="store_true",
                        help="Use unique prompts for each run to avoid cache issues")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode - only show summary (good for stress tests)")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info for each chunk")
    parser.add_argument("--nostream", action="store_true",
                        help="Disable streaming (use non-streaming API)")

    args = parser.parse_args()
    
    print(f"Benchmarking: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Streaming: {not args.nostream}")
    print()
    
    try:
        stream = not args.nostream
        results = run_benchmark(
            args.base_url,
            args.api_key,
            args.model,
            args.prompt,
            args.max_tokens,
            args.runs,
            args.warmup,
            args.temperature,
            args.top_p,
            top_k=args.top_k,
            rep=args.rep,
            freq=args.freq,
            unique_prompts=args.unique_prompts,
            quiet=args.quiet,
            debug=args.debug,
            stream=stream,
        )
        print_summary(results, stream=stream)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
