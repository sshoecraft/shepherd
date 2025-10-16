#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/steve/venv/lib/python3.12/site-packages')

from tensorrt_llm.executor import Executor, ExecutorConfig, ModelType, KvCacheConfig

engine_path = "/home/steve/models/Llama-3.1-8B-Instruct-FP16_engine"

try:
    config = ExecutorConfig()
    config.max_beam_width = 1

    kv_config = KvCacheConfig()
    kv_config.enable_block_reuse = True
    config.kv_cache_config = kv_config

    print(f"Loading engine from: {engine_path}")
    executor = Executor(engine_path, ModelType.DECODER_ONLY, config)
    print("✅ Engine loaded successfully!")

except Exception as e:
    print(f"❌ Error loading engine: {e}")
    import traceback
    traceback.print_exc()
