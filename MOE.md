# GLM-4.5-Air MoE Architecture Analysis

## Overview

GLM-4.5-Air is a Mixture of Experts (MoE) model with 106B total parameters and 12B active parameters. Goal: Convert from vLLM/HuggingFace format to TensorRT-LLM for integration with Shepherd's stateful KV-cache and hierarchical memory management.

## vLLM Support Status

- **Officially supported** in vLLM as of 2025
- vLLM worked with GLM team before release to ensure full support
- Documentation: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM-4.5.html
- Implementation: `/home/steve/src/vllm/vllm/model_executor/models/glm4_moe.py`

## Available Model Formats

- **GPTQ INT4 quantized**: `QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix`, `cpatonn/GLM-4.5-Air-GPTQ-4bit`
- **BF16/FP8**: `zai-org/GLM-4.5-Air`

## TensorRT-LLM Support Status

### Official TensorRT-LLM
- **Does NOT support GLM-4.5-Air MoE** in current Docker image (v0.20.0)
- GLM architecture not included in official builds

### Community Fork
- **deepinfra/TensorRT-LLM** has GLM-4.5-Air support
- Requires building from source (1-2 hours compile time)
- Custom Docker build needed

## vLLM Architecture Issues

### KV Cache is NOT Truly Stateful
vLLM's "stateful KV cache" is actually **prefix caching** via content-based hashing:
- Automatically shares KV blocks when token sequences match across requests
- **Deallocates all KV cache when request finishes**
- No persistent session-based cache
- No eviction callbacks for external memory management

This is **incompatible with Shepherd's design** which requires:
- Persistent session-based KV cache across requests
- Eviction callbacks when VRAM fills → move blocks to RAM/storage
- Hierarchical memory manager integration

### vLLM Integration Problems
- vLLM is **Python-first** library, not designed for C++ linking
- C++/CUDA code in `csrc/` is for internal kernels, not external API
- No official C++ API for stateful inference
- Would require running as separate HTTP service (loses stateful benefits)

## GLM-4 MoE Architecture Details

Source: `/home/steve/src/vllm/vllm/model_executor/models/glm4_moe.py`

### 1. Glm4MoeAttention (lines 247-348)

```python
class Glm4MoeAttention:
    - QKV projection (QKVParallelLinear)
    - Optional QK normalization (RMSNorm on Q and K)
    - Rotary position embeddings (RoPE)
        - partial_rotary_factor = 0.5 (only half of head_dim gets rotated)
    - Grouped Query Attention (GQA) support
    - head_dim can be specified or defaults to hidden_size / num_heads
    - max_position_embeddings = 131072 (128K context)
```

### 2. Glm4MoE (lines 118-244)

Unique MoE architecture with both routed and shared experts:

```python
class Glm4MoE:
    Components:
    - Router gate: nn.Linear(hidden_size → n_routed_experts)
        - Uses float32 for routing (not bf16)
        - Has e_score_correction_bias parameter

    - Routed experts: FusedMoE or SharedFusedMoE
        - num_experts = n_routed_experts
        - top_k = num_experts_per_tok
        - intermediate_size = moe_intermediate_size

    - Shared experts (optional): Glm4MoeMLP
        - Always activated (not routed)
        - intermediate_size = moe_intermediate_size * n_shared_experts

    Routing:
    - Uses grouped top-k routing (n_group, topk_group parameters)
    - Scoring function: SIGMOID (not softmax!)
    - renormalize = norm_topk_prob
    - e_score_correction_bias applied during scoring

    Output:
    - final = routed_output * routed_scaling_factor + shared_output
```

### 3. Glm4MoeMLP (lines 79-115)

Standard FFN for dense layers and shared experts:

```python
class Glm4MoeMLP:
    - gate_up_proj: MergedColumnParallelLinear (SwiGLU style)
    - down_proj: RowParallelLinear
    - activation: SiluAndMul (SwiGLU)
```

### 4. Glm4MoeDecoderLayer (lines 351-426)

```python
class Glm4MoeDecoderLayer:
    - input_layernorm: RMSNorm (pre-norm)
    - self_attn: Glm4MoeAttention
    - post_attention_layernorm: RMSNorm
    - mlp: Either Glm4MoeMLP (dense) or Glm4MoE (sparse)
        - Layers < first_k_dense_replace: Dense MLP
        - Layers >= first_k_dense_replace: MoE
```

### 5. Full Model (lines 437-634)

```python
class Glm4MoeModel:
    - embed_tokens: VocabParallelEmbedding
    - layers: List[Glm4MoeDecoderLayer]
    - norm: RMSNorm (final layer norm)

class Glm4MoeForCausalLM:
    - model: Glm4MoeModel
    - lm_head: ParallelLMHead
    - logits_processor: LogitsProcessor
```

## Key Config Parameters

From `transformers.models.glm4_moe.Glm4MoeConfig`:

```python
Required for TensorRT-LLM conversion:
- hidden_size
- num_hidden_layers
- num_attention_heads
- num_key_value_heads (GQA)
- head_dim
- intermediate_size (for dense layers)
- moe_intermediate_size (for expert FFN)
- n_routed_experts
- n_shared_experts
- num_experts_per_tok (top-k)
- n_group, topk_group (grouped routing)
- routed_scaling_factor
- first_k_dense_replace (which layers are MoE vs dense)
- norm_topk_prob (renormalization flag)
- rope_theta (default 10000)
- rope_scaling (optional)
- max_position_embeddings (131072)
- rms_norm_eps (1e-05)
- attention_bias (qkv_bias)
- use_qk_norm
- partial_rotary_factor (0.5)
- hidden_act (activation function, must be "silu")
```

## Critical MoE Features for TensorRT-LLM Conversion

### Must-Have Features
1. **Grouped top-k routing** with `n_group` and `topk_group` parameters
2. **Sigmoid scoring function** (not standard softmax)
3. **e_score_correction_bias** in expert selection
4. **Shared + routed experts** hybrid architecture
5. **routed_scaling_factor** applied to routed outputs
6. **Partial rotary embeddings** (only 50% of head_dim)
7. **Mixed dense/MoE layers** (controlled by first_k_dense_replace)

### Potential Blockers
- TensorRT-LLM may not support grouped top-k routing
- Sigmoid scoring function may not be available
- Shared+routed hybrid architecture is non-standard
- e_score_correction_bias is GLM-specific

## Weight Loading

From `glm4_moe.py` lines 532-634:

```python
Stacked parameters:
- qkv_proj ← [q_proj, k_proj, v_proj]
- gate_up_proj ← [gate_proj, up_proj]

Expert parameters:
- mlp.experts[N].gate_proj
- mlp.experts[N].up_proj
- mlp.experts[N].down_proj

Shared expert parameters:
- mlp.shared_experts.gate_up_proj
- mlp.shared_experts.down_proj

Special handling:
- GPTQ models have extra .bias that should be skipped
- FP8 models need kv-scale remapping
- Expert parallelism (EP) requires careful weight distribution
```

## TensorRT-LLM Conversion Strategy

### Option 1: Use deepinfra/TensorRT-LLM Fork
1. Clone https://github.com/deepinfra/TensorRT-LLM
2. Build from source (requires custom Docker build, 1-2 hours)
3. Hope GLM-4.5-Air support is complete and working
4. Risk: May have bugs, unclear maintenance status

### Option 2: Manual Conversion
1. Study TensorRT-LLM's MoE implementation (if it exists)
2. Create custom model definition for GLM-4 MoE architecture
3. Write conversion script:
   - Load HuggingFace weights
   - Map to TensorRT-LLM format
   - Handle grouped routing, sigmoid scoring, shared experts
4. Build TensorRT engine for RTX 3090 (FP16/INT8)
5. Integrate with Shepherd's backend system

Risk: High complexity, may hit unsupported features

### Option 3: Wait for Official Support
- Monitor TensorRT-LLM releases for GLM-4.5 support
- Use Llama 3.1 70B in meantime

### Option 4: Separate vLLM Service (Not Recommended)
- Run vLLM as HTTP server
- No stateful KV cache integration
- No hierarchical memory management
- Loses main benefit of Shepherd architecture

## Recommended Path Forward

**For now: Use Llama 3.1 70B Instruct with TensorRT-LLM**
- Proven to work
- Full Shepherd integration
- Stateful KV cache with eviction callbacks

**Future: Revisit GLM-4.5-Air when:**
1. Official TensorRT-LLM adds GLM support, OR
2. Time permits investigating deepinfra fork, OR
3. Manual conversion becomes necessary for project requirements

## Related Files

- vLLM GLM-4 MoE implementation: `/home/steve/src/vllm/vllm/model_executor/models/glm4_moe.py`
- vLLM model registry: `/home/steve/src/vllm/vllm/model_executor/models/registry.py`
- Shepherd backend interface: `/home/steve/src/shepherd/backend_interface.h`
- Shepherd TensorRT integration docs: See TENSORRT_*.md files in project root

## Hardware Requirements

**GLM-4.5-Air (106B total, 12B active):**
- GPTQ INT4: ~26GB VRAM (fits on dual RTX 3090 = 48GB)
- BF16: ~106GB (would need offloading or FP8)
- Recommended: GPTQ INT4 for dual RTX 3090 setup

**Current Shepherd Hardware:**
- Dual NVIDIA RTX 3090 (48GB VRAM total)
- 128GB system RAM
- NVMe SSD storage

## References

- vLLM Blog: https://blog.vllm.ai/2025/08/19/glm45-vllm.html
- GLM-4.5 Announcement: https://z.ai/blog/glm-4.5
- HuggingFace: https://huggingface.co/zai-org/GLM-4.5-Air
- GPTQ Models: https://huggingface.co/QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix
- deepinfra TensorRT-LLM: https://github.com/deepinfra/TensorRT-LLM
