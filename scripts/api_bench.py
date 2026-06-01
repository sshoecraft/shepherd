#!/usr/bin/env python3
"""
Unified API benchmark tool supporting OpenAI, Anthropic, and Gemini APIs.
Measures tokens/sec, TTFT, and inter-token latency.
"""

import argparse
import time
import statistics
import random
from abc import ABC, abstractmethod

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


class APIClient(ABC):
    """Abstract base class for API clients."""

    @abstractmethod
    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        """Send a chat request and return metrics."""
        pass

    @abstractmethod
    def list_models(self) -> list:
        """List available models."""
        pass


class OpenAIClient(APIClient):
    """OpenAI-compatible API client."""

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        from openai import OpenAI
        import httpx
        http_client = httpx.Client(verify=not insecure) if insecure else None
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        self.model = model
        self.extra_params = extra_params or {}

    def list_models(self) -> list:
        return [m.id for m in self.client.models.list().data]

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        start_time = time.perf_counter()
        first_token_time = None
        token_times = []
        tokens = 0
        all_content = []

        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        for k, v in self.extra_params.items():
            if v is not None:
                extra_body[k] = v

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            **({"stream_options": {"include_usage": True}} if stream else {}),
            **({"extra_body": extra_body} if extra_body else {})
        )

        if stream:
            usage_tokens = None
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                piece = None
                if delta:
                    extra = getattr(delta, "model_extra", None) or {}
                    piece = (
                        delta.content
                        or getattr(delta, "reasoning_content", None)
                        or extra.get("reasoning_content")
                        or extra.get("reasoning")
                    )
                if debug:
                    print(f"  DEBUG chunk: choices={len(chunk.choices) if chunk.choices else 0}, "
                          f"finish={chunk.choices[0].finish_reason if chunk.choices else None}, "
                          f"content={piece!r}")
                if piece:
                    all_content.append(piece)
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        token_times.append(now)
                    tokens += 1
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_tokens = chunk.usage.completion_tokens
            if usage_tokens is not None:
                tokens = usage_tokens
        else:
            if response.choices and response.choices[0].message.content:
                first_token_time = time.perf_counter()
                content = response.choices[0].message.content
                all_content.append(content)
                if hasattr(response, 'usage') and response.usage:
                    tokens = response.usage.completion_tokens
                else:
                    tokens = len(content) // 4
                if debug:
                    print(f"  DEBUG non-stream: tokens={tokens}, content={content[:100]!r}...")

        end_time = time.perf_counter()
        return self._compute_metrics(start_time, end_time, first_token_time, tokens, all_content, stream)

    def _compute_metrics(self, start_time, end_time, first_token_time, tokens, all_content, stream):
        total_time = end_time - start_time
        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
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


class AnthropicClient(APIClient):
    """Anthropic API client."""

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        import anthropic
        import httpx
        http_client = httpx.Client(verify=not insecure) if insecure else None
        if base_url:
            self.client = anthropic.Anthropic(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            self.client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
        self.model = model
        self.extra_params = extra_params or {}

    def list_models(self) -> list:
        # Anthropic doesn't have a models endpoint, return common models
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        start_time = time.perf_counter()
        first_token_time = None
        tokens = 0
        all_content = []

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if top_p is not None and top_p < 1.0:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k

        if stream:
            last_token_time = None
            chunk_count = 0
            with self.client.messages.stream(**kwargs) as response:
                for text in response.text_stream:
                    if debug:
                        print(f"  DEBUG chunk: {text!r}")
                    if text:
                        all_content.append(text)
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        last_token_time = now
                        chunk_count += 1
                # Get actual token count from final message
                final_message = response.get_final_message()
                if debug:
                    print(f"  DEBUG final_message.usage: {final_message.usage if final_message else None}")
                content_len = len("".join(all_content))
                estimated_tokens = max(1, content_len // 4)
                if final_message and final_message.usage and final_message.usage.output_tokens > 1:
                    tokens = final_message.usage.output_tokens
                else:
                    # Fallback: estimate from content length
                    tokens = estimated_tokens
            # Use last token time as end time for streaming
            if last_token_time:
                end_time = last_token_time
        else:
            response = self.client.messages.create(**kwargs)
            first_token_time = time.perf_counter()
            if response.content:
                for block in response.content:
                    if hasattr(block, 'text'):
                        all_content.append(block.text)
            if debug:
                print(f"  DEBUG response.usage: {response.usage}")
            content_len = len("".join(all_content))
            estimated_tokens = max(1, content_len // 4)
            if response.usage and response.usage.output_tokens and response.usage.output_tokens > 1:
                tokens = response.usage.output_tokens
            else:
                # Fallback: estimate from content length (~4 chars per token)
                # Server returned output_tokens=1 or 0 which is clearly wrong
                tokens = estimated_tokens
            if debug:
                print(f"  DEBUG non-stream: tokens={tokens}, content={''.join(all_content)[:100]!r}...")
            end_time = time.perf_counter()

        total_time = end_time - start_time

        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
            # If generation_time is very small but we have tokens, server is buffering
            # (fake streaming) - use total_time instead
            if generation_time < 0.01 and tokens > 1:
                generation_time = total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
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


class GeminiClient(APIClient):
    """Google Gemini API client."""

    def __init__(self, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)
        self.extra_params = extra_params or {}
        # Note: Gemini SDK doesn't support custom SSL settings easily

    def list_models(self) -> list:
        import google.generativeai as genai
        return [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

    def chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float,
             top_k: int = None, stream: bool = True, debug: bool = False) -> dict:
        import google.generativeai as genai

        start_time = time.perf_counter()
        first_token_time = None
        tokens = 0
        all_content = []

        gen_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if top_k is not None:
            gen_config.top_k = top_k

        if stream:
            response = self.model.generate_content(prompt, generation_config=gen_config, stream=True)
            for chunk in response:
                if debug:
                    text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                    print(f"  DEBUG chunk: {text!r}")
                if hasattr(chunk, 'text') and chunk.text:
                    all_content.append(chunk.text)
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    tokens += 1
            # Try to get actual token count
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.candidates_token_count or tokens
        else:
            response = self.model.generate_content(prompt, generation_config=gen_config, stream=False)
            first_token_time = time.perf_counter()
            if response.text:
                all_content.append(response.text)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = response.usage_metadata.candidates_token_count
            else:
                tokens = len(response.text) // 4 if response.text else 0
            if debug:
                content_preview = response.text[:100] if response.text else ''
                print(f"  DEBUG non-stream: tokens={tokens}, content={content_preview!r}")

        end_time = time.perf_counter()
        total_time = end_time - start_time

        if stream:
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            generation_time = end_time - first_token_time if first_token_time else total_time
        else:
            ttft = total_time * 1000
            generation_time = total_time

        if tokens > 0 and generation_time > 0:
            itl_ms = (generation_time * 1000) / tokens
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


def create_client(api_type: str, base_url: str, api_key: str, model: str, extra_params: dict = None, insecure: bool = False) -> APIClient:
    """Factory function to create the appropriate client."""
    if api_type == "openai":
        return OpenAIClient(base_url, api_key, model, extra_params, insecure=insecure)
    elif api_type == "anthropic":
        return AnthropicClient(base_url, api_key, model, extra_params, insecure=insecure)
    elif api_type == "gemini":
        return GeminiClient(base_url, api_key, model, extra_params, insecure=insecure)
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def run_benchmark(client: APIClient, prompt: str, max_tokens: int, runs: int, warmup: int,
                  temperature: float, top_p: float, top_k: int = None,
                  unique_prompts: bool = False, quiet: bool = False, debug: bool = False,
                  stream: bool = True) -> list:
    """Run multiple benchmark iterations."""

    # Prepare prompts
    if unique_prompts:
        prompts = BENCHMARK_PROMPTS.copy()
        random.shuffle(prompts)
        while len(prompts) < warmup + runs:
            prompts.extend(BENCHMARK_PROMPTS)
    else:
        prompts = [prompt] * (warmup + runs)

    prompt_idx = 0

    # Warmup runs
    if warmup > 0:
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            client.chat(prompts[prompt_idx], max_tokens, temperature, top_p, top_k=top_k, stream=stream, debug=debug)
            prompt_idx += 1

    # Benchmark runs
    if quiet:
        print(f"Running benchmark ({runs} runs)...", end="", flush=True)
    else:
        print(f"Running benchmark ({runs} runs)...")

    results = []
    failed_count = 0
    for i in range(runs):
        result = client.chat(prompts[prompt_idx], max_tokens, temperature, top_p, top_k=top_k, stream=stream, debug=debug)
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


def print_summary(results: list, stream: bool = True):
    """Print summary statistics."""
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
        description="Benchmark LLM API endpoints (OpenAI, Anthropic, Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI-compatible server (vLLM, llama.cpp, etc.)
  %(prog)s --type openai --base-url http://localhost:8000/v1 --model llama

  # Anthropic API (direct or proxy)
  %(prog)s --type anthropic --base-url http://localhost:3456 --model claude-3-opus-20240229

  # Anthropic official API
  %(prog)s --type anthropic --model claude-3-5-sonnet-20241022

  # Google Gemini
  %(prog)s --type gemini --model gemini-1.5-pro

  # Custom settings
  %(prog)s --type openai --base-url http://localhost:8000/v1 --model qwen --max-tokens 512 --runs 20
        """,
    )
    parser.add_argument("--type", "-t", choices=["openai", "anthropic", "gemini"], default="openai",
                        help="API type (default: openai)")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (default: type-specific default)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from environment or 'not-needed')")
    parser.add_argument("--model", default=None,
                        help="Model name (default: auto-detect or type-specific default)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--prompt",
                        default="Write a detailed explanation of how transformers work in neural networks.",
                        help="Prompt to use for benchmarking")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate (default: 2048)")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of benchmark runs (default: 5)")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup runs (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (default: None)")
    parser.add_argument("--unique-prompts", action="store_true",
                        help="Use unique prompts for each run to avoid cache issues")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Quiet mode - only show summary")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info for each chunk")
    parser.add_argument("--nostream", action="store_true",
                        help="Disable streaming")
    parser.add_argument("--insecure", "-k", action="store_true",
                        help="Disable SSL certificate verification (for self-signed certs)")

    args = parser.parse_args()

    # Set defaults based on API type
    if args.base_url is None:
        if args.type == "openai":
            args.base_url = "http://localhost:8000/v1"
        elif args.type == "anthropic":
            args.base_url = None  # Use default Anthropic API
        elif args.type == "gemini":
            args.base_url = None  # Gemini doesn't use base_url
    elif args.type == "openai" and args.base_url and not args.base_url.endswith("/v1"):
        # OpenAI SDK expects /v1 suffix
        args.base_url = args.base_url.rstrip("/") + "/v1"

    if args.api_key is None:
        import os
        if args.type == "openai":
            args.api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        elif args.type == "anthropic":
            args.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not args.api_key:
                print("Error: ANTHROPIC_API_KEY environment variable or --api-key required")
                return
        elif args.type == "gemini":
            args.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not args.api_key:
                print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable or --api-key required")
                return

    # Create client
    try:
        client = create_client(args.type, args.base_url, args.api_key, args.model or "placeholder", insecure=args.insecure)
    except ImportError as e:
        print(f"Error: Missing dependency for {args.type} API: {e}")
        print(f"Install with: pip install {'openai' if args.type == 'openai' else 'anthropic' if args.type == 'anthropic' else 'google-generativeai'}")
        return

    # List models or auto-detect
    if args.list_models or args.model is None:
        try:
            models = client.list_models()
        except Exception as e:
            if args.list_models:
                print(f"Failed to list models: {e}")
                return
            models = []

        if args.list_models:
            for m in models:
                print(m)
            return

        if models and args.model is None:
            args.model = models[0]
            print(f"Auto-selected model: {args.model}")
            # Recreate client with actual model
            client = create_client(args.type, args.base_url, args.api_key, args.model, insecure=args.insecure)
        elif args.model is None:
            print("Error: --model is required (could not auto-detect)")
            return

    print(f"Benchmarking: {args.base_url or f'{args.type} API'}")
    print(f"API Type: {args.type}")
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Streaming: {not args.nostream}")
    print()

    try:
        stream = not args.nostream
        results = run_benchmark(
            client,
            args.prompt,
            args.max_tokens,
            args.runs,
            args.warmup,
            args.temperature,
            args.top_p,
            top_k=args.top_k,
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
