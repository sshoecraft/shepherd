# Converting compressed-tensors to AutoAWQ Format for TensorRT-LLM

## Problem

Models quantized with `llmcompressor` or `compressed-tensors` use a different tensor format than AutoAWQ. TensorRT-LLM's checkpoint converter expects AutoAWQ format, so we need to convert between formats.

## Format Differences

### compressed-tensors format
- `weight_packed`: `[out_features, in_features/8]` int32 - sequential packing
- `weight_scale`: `[out_features, groups]` bfloat16
- `weight_zero_point`: `[out_features/8, groups]` int32 - sequential packing
- `weight_shape`: `[out_features, in_features]`

### AutoAWQ format (what TRT-LLM expects)
- `qweight`: `[in_features, out_features/8]` int32 - **interleaved packing**
- `scales`: `[groups, out_features]` float16
- `qzeros`: `[groups, out_features/8]` int32 - **interleaved packing**

### Key Differences
1. **Transpose**: AutoAWQ is `[in, out/8]` vs CT's `[out, in/8]`
2. **Pack order**: AutoAWQ uses interleaved order `[0, 4, 1, 5, 2, 6, 3, 7]` vs CT's sequential `[0, 1, 2, 3, 4, 5, 6, 7]`
3. **Scales/zeros shape**: Transposed (groups first vs last)

## The Conversion Script

`scripts/convert_ct_to_awq.py` handles the conversion:

```bash
./convert_ct_to_awq.py --output /path/to/output /path/to/compressed-tensors-model
```

### Core Algorithm

```python
# 1. Unpack CT weights (sequential order)
unpacked = []
for i in range(8):
    unpacked.append((weight_packed >> (i * 4)) & 0xF)
stacked = torch.stack(unpacked, dim=2)
weight_int4 = stacked.reshape(out_features, in_features)

# 2. Transpose for AWQ
weight_int4_t = weight_int4.t().contiguous()

# 3. Repack with AWQ interleaved order
AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
qweight = torch.zeros(in_features, out_features // 8, dtype=torch.int32)
for i in range(8):
    qweight |= (weight_int4_t[:, i::8].to(torch.int32) << (AWQ_ORDER[i] * 4))

# 4. Transpose scales
scales = weight_scale.t().contiguous().to(torch.float16)

# 5. Unpack and repack zeros with AWQ order
z_unpacked = []
for i in range(8):
    z_unpacked.append((weight_zero_point >> (i * 4)) & 0xF)
z_stacked = torch.stack(z_unpacked, dim=2)
zeros_full = z_stacked.permute(1, 0, 2).reshape(-1, out_features)

qzeros = torch.zeros(zeros_full.shape[0], out_features // 8, dtype=torch.int32)
for i in range(8):
    qzeros |= (zeros_full[:, i::8].to(torch.int32) << (AWQ_ORDER[i] * 4))
```

## TensorRT-LLM Build Configuration

In `build_engine.py`, AWQ models need:

```python
if quant_type == 'awq':
    # AutoAWQ uses W4A16_GPTQ algorithm (asymmetric UINT4 with zero points)
    # but needs use_autoawq=True for correct tensor loading (different format than GPTQ)
    tllm_quant_config = QuantConfig()
    tllm_quant_config.quant_algo = QuantAlgo.W4A16_GPTQ  # NOT W4A16_AWQ!
    tllm_quant_config.group_size = hf_quant_config.get('group_size', 128)
    tllm_quant_config.has_zero_point = hf_quant_config.get('zero_point', True)
    override_fields['use_autoawq'] = True  # Critical for tensor loading!
```

### Why W4A16_GPTQ not W4A16_AWQ?
- `W4A16_AWQ` is for NVIDIA ModelOpt AWQ (symmetric INT4 with pre-quant scales)
- `W4A16_GPTQ` is for AutoAWQ format (asymmetric UINT4 with zero points)
- AutoAWQ and GPTQ use the same GEMM kernel, just different tensor layouts
- `use_autoawq=True` tells TRT-LLM to use AutoAWQ's tensor loading path

## Full Workflow

```bash
# 1. Convert compressed-tensors to AWQ format
cd /home/steve/src/shepherd/scripts
./convert_ct_to_awq.py \
    --output /home/steve/models/MODEL-AWQ \
    /path/to/compressed-tensors-model

# 2. Build TensorRT engine (from venv)
source ~/venv/bin/activate
./build_engine.py /home/steve/models/MODEL-AWQ \
    --output_dir /home/steve/models/MODEL_engine \
    --max_seq_len 32768 \
    --tp 1 --pp 3 \
    --workers 1

# Or use the convenience script:
bash b
```

## Debugging Tips

### Verify shapes after conversion
```python
from safetensors import safe_open
with safe_open("/path/to/model/model-00001-of-00008.safetensors", framework="pt") as f:
    for name in ["qweight", "scales", "qzeros"]:
        key = f"model.layers.0.self_attn.q_proj.{name}"
        if key in f.keys():
            print(f"{name}: {f.get_tensor(key).shape}")
```

Expected for q_proj (8192 out, 8192 in, 128 group_size):
- qweight: [8192, 1024] (in, out/8)
- scales: [64, 8192] (groups, out)
- qzeros: [64, 1024] (groups, out/8)

### Verify dequantization
```python
# Dequant formula: (qweight - qzeros) * scales
# Values should be in range [-1, 1] approximately
```

### Common errors
1. **Shape mismatch in QKV concat**: scales/qzeros need groups as dim 0
2. **65536 vs 8192**: Wrong quant algo (W4A16_AWQ vs W4A16_GPTQ)
3. **Garbage output**: Wrong pack order (must use AWQ interleaved)

## References

- TRT-LLM AWQ issue: https://github.com/NVIDIA/TensorRT-LLM/issues/2803
- AutoAWQ format: https://github.com/casper-hansen/AutoAWQ/issues/566
- AWQ pack order: https://deepwiki.com/intel/auto-round/9.3-awq-format
