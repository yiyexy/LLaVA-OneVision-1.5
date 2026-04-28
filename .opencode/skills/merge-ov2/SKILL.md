---
name: merge-ov2
description: Bilingual guide for merging ViT + LLM into LlavaOnevision2 HF checkpoint and validating weight/inference consistency
compatibility: opencode
metadata:
  domain: model-merge
  framework: llava-onevision2
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when merging a standalone ViT encoder and LLM into a unified LlavaOnevision2 HuggingFace checkpoint, and when validating that the merged weights and inference outputs are consistent with the originals.

当需要将独立的 ViT encoder 和 LLM 合并成统一的 LlavaOnevision2 HuggingFace checkpoint，并验证合并后权重和推理输出与原始模型一致时，使用这个 skill。

## Prerequisites / 前置条件

- Container `llava_megatron_container_ax` running with GPU access
- All paths below assume execution **inside the container** at `/workspace/LLaVA-OneVision-2`
- `PYTHONPATH=transformers_impl:.` must be set for all Python commands
- For large models, use tmpfs (`/train_tmp`) for I/O performance

容器 `llava_megatron_container_ax` 需启动并有 GPU 访问权限。以下所有路径假设在容器内 `/workspace/LLaVA-OneVision-2` 执行。所有 Python 命令需设置 `PYTHONPATH=transformers_impl:.`。大模型建议用内存盘 `/train_tmp`。

## Architecture / 架构

### What merge_ov2 does / merge_ov2 做了什么

```
ViT encoder (e.g. onevision_encoder_patch16_0424)
  + LLM (e.g. Qwen3-4B-Instruct-2507)
  + Processor (e.g. LLaVA-OneVision-1.5-8B-Instruct)
  → Unified LlavaOnevision2ForConditionalGeneration checkpoint
```

Key transformations during merge:

合并时的关键转换：

1. **ViT weights** are prefixed with `visual.` (e.g. `encoder.layers.0.self_attn.q_proj.weight` → `visual.encoder.layers.0.self_attn.qkv.weight`)
2. **QKV fusion**: separate `q_proj / k_proj / v_proj` are concatenated into fused `self_attn.qkv` (introduces ~1e-7 bf16 divergence)
3. **LLM weights** are prefixed with `language_model.` (e.g. `model.layers.0.self_attn.q_proj.weight` → `language_model.model.layers.0.self_attn.q_proj.weight`)
4. **Adapter** (`multi_modal_projector`) is randomly initialized if no adapter checkpoint is provided
5. **`layernorm_post`** from ViT is dropped (not used in LlavaOnevision2)
6. **`class_embedding`** may not exist in some ViT encoders (e.g. patch16 variant)

### Source code layout / 源码结构

```
transformers_impl/merge_ov2/
├── __main__.py          # CLI entry point
├── cli.py               # Argument parsing for merge / validate / dry-run
├── remap.py             # Weight key remapping logic
├── loader.py            # Weight loading from source checkpoints
├── save.py              # Save merged checkpoint
├── io.py                # I/O utilities
├── utils.py             # Shared utilities
├── variants/
│   ├── dense.py         # Dense model variant
│   └── moe.py           # MoE model variant
└── validators/
    ├── vit_layerwise.py   # ViT layer-wise weight validator
    ├── vit_blockorder.py  # ViT block-order validator (patch14+sms=2 only)
    ├── llm_parallel.py    # LLM parallel validator
    ├── llm_sequential.py  # LLM sequential validator
    └── e2e.py             # End-to-end validator
```

## CLI Reference / CLI 参考

### Subcommand: `merge`

Remap + load + (optional validate) + save.

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 merge \
  --variant dense \
  --vit /path/to/vit_encoder \
  --llm /path/to/llm \
  --processor /path/to/processor \
  --out /path/to/output \
  --spatial-merge-size 3 \
  --target-dtype bf16 \
  --vit-validator-strategy layerwise
```

| Argument | Required | Description |
|---|---|---|
| `--variant` | Yes | `dense` or `moe` |
| `--vit` | Yes | Path to standalone ViT encoder checkpoint |
| `--llm` | Yes | Path to LLM checkpoint (e.g. Qwen3-4B) |
| `--processor` | Yes | Path to processor/tokenizer (e.g. LLaVA-OneVision-1.5-8B-Instruct) |
| `--out` | Yes | Output directory for merged checkpoint |
| `--adapter` | No | Path to adapter checkpoint (randomly initialized if omitted) |
| `--spatial-merge-size` | No | 1, 2, or 3 (default depends on ViT) |
| `--target-dtype` | No | `bf16`, `fp16`, or `fp32` |
| `--device` | No | Device for validation (default: auto) |
| `--img` | No | Image path for ViT validation |
| `--sample-text` | No | Text for LLM validation |
| `--validate-skip` | No | Skip specific validation: `vit`, `llm`, or `e2e` |
| `--vit-validator-strategy` | No | `blockorder` (patch14+sms=2 only) or `layerwise` |
| `--llm-validator-strategy` | No | `parallel` or `sequential` |
| `--patch-pos-encoding` / `--no-patch-pos-encoding` | No | Enable/disable patch position encoding |

### Subcommand: `validate`

Validate an already-merged checkpoint against original sources.

验证已合并的 checkpoint 与原始模型的一致性。

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 validate \
  --variant dense \
  --ckpt /path/to/merged_checkpoint \
  --vit /path/to/vit_encoder \
  --llm /path/to/llm \
  --processor /path/to/processor \
  --vit-validator-strategy layerwise
```

### Subcommand: `dry-run`

Remap only; report load coverage; no save.

仅做 remap，报告加载覆盖率，不保存。

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 dry-run \
  --variant dense \
  --vit /path/to/vit_encoder \
  --llm /path/to/llm \
  --processor /path/to/processor \
  --spatial-merge-size 3
```

## Concrete Example / 具体示例

### Merging Qwen3-4B + onevision_encoder_patch16 (sms=3)

```bash
docker exec llava_megatron_container_ax bash -c '
cd /workspace/LLaVA-OneVision-2 && \
PYTHONPATH=transformers_impl:. python -m merge_ov2 merge \
  --variant dense \
  --vit /train_tmp/onevision_encoder_patch16_0424 \
  --llm /train_tmp/Qwen3-4B-Instruct-2507 \
  --processor /train_tmp/LLaVA-OneVision-1.5-8B-Instruct \
  --out /train_tmp/llava_onevision2_4b_p16m33 \
  --spatial-merge-size 3 \
  --target-dtype bf16 \
  --vit-validator-strategy layerwise
'
```

Output checkpoint config will have: `patch_size=16, spatial_merge_size=3, image_size=448, hidden_size=1024, 24 ViT layers, 36 LLM layers`.

输出 checkpoint 的配置: `patch_size=16, spatial_merge_size=3, image_size=448, hidden_size=1024, 24 ViT 层, 36 LLM 层`。

## Variant Cheat Sheet / Variant 参数对照表

OV2 4B has two real-world variants. The `--patch-size` (set by the ViT
checkpoint), `--spatial-merge-size`, and `--vit-validator-strategy` must all
match — using the wrong validator silently crashes inside reshape ops.

OV2 4B 有两套真实使用的 variant。`--patch-size`（由 ViT checkpoint 决定）、
`--spatial-merge-size` 和 `--vit-validator-strategy` 必须配齐 —— 用错
validator 会在 reshape 里直接崩。

| Variant | ViT checkpoint suffix | `--spatial-merge-size` | `--vit-validator-strategy` | Effective image_size |
|---|---|---|---|---|
| `4b` (legacy / 旧) | `onevision-encoder-large` (patch14) | `2` | `blockorder` (default) or `layerwise` | 14 × 2 × N |
| `4b_p16m3` (current / 当前) | `onevision_encoder_patch16_*` | `3` | `layerwise` (**required / 必须**) | 16 × 3 × N = 48 × N |

> **Why `layerwise` is required for `4b_p16m3` / 为什么 4b_p16m3 必须用 layerwise**:
> `vit_blockorder.py`'s reshape hard-codes `patch_size=14, spatial_merge_size=2`.
> With `patch_size=16, spatial_merge_size=3` the reshape dimensions don't
> divide evenly and you get `RuntimeError: shape '[...]' is invalid for input of size N`.
> `vit_blockorder.py` 的 reshape 写死了 `patch_size=14, spatial_merge_size=2`
> 的尺寸假设。换成 `patch_size=16, spatial_merge_size=3` 后维度对不上，
> 会抛 `RuntimeError: shape '[...]' is invalid for input of size N`。

## Post-Merge Validation / 合并后验证

**Always prefer the `validate` subcommand over hand-written scripts.** It runs
the same three validators (`vit`, `llm`, `e2e`) used during `merge`, against
an already-saved checkpoint. Use this when:

- you skipped validation during merge (`--validate-skip`)
- you manually patched a saved checkpoint and want to re-verify
- you downloaded a checkpoint and want to confirm parity

**优先使用 `validate` 子命令而不是手写脚本**。它会对一份已保存的 checkpoint
跑和 merge 时同样的三个 validator（`vit`、`llm`、`e2e`）。适用场景：

- merge 时跳过了验证（`--validate-skip`）
- 手动改过 checkpoint 后重验证
- 下载了 checkpoint 想确认一致性

```bash
docker exec llava_megatron_container_ax bash -c '
cd /workspace/LLaVA-OneVision-2 && \
PYTHONPATH=transformers_impl:. python -m merge_ov2 validate \
  --variant dense \
  --ckpt /train_tmp/llava_onevision2_4b_p16m33 \
  --vit /train_tmp/onevision_encoder_patch16_0424 \
  --llm /train_tmp/Qwen3-4B-Instruct-2507 \
  --processor /train_tmp/LLaVA-OneVision-1.5-8B-Instruct \
  --img /train_tmp/sample.jpg \
  --sample-text "Hello, world!" \
  --vit-validator-strategy layerwise
'
```

### Skipping validators / 跳过 validator

`--validate-skip` is **append**-style; pass it once per validator to skip.
Three validators exist: `vit`, `llm`, `e2e`. Run only the ones you need to
save GPU time.

`--validate-skip` 是 **append** 模式，每跳一个就传一次。一共三个 validator：
`vit`、`llm`、`e2e`。只跑你需要的，省 GPU 时间。

```bash
# Skip e2e only (run vit + llm) / 只跳 e2e
... --validate-skip e2e

# Validate only LLM (skip vit + e2e) / 只验证 LLM
... --validate-skip vit --validate-skip e2e

# Skip everything (no parity check) / 全跳过
... --validate-skip vit --validate-skip llm --validate-skip e2e
```

When you skip `vit`, `--qwen-processor` and `--img` may be omitted; when you
skip `llm`, `--sample-text` may be omitted. The CLI enforces only the flags
required by the validators you actually run.

跳了 `vit` 时可以省略 `--qwen-processor` 和 `--img`；跳了 `llm` 时可以
省略 `--sample-text`。CLI 只对实际要跑的 validator 强制要求对应的参数。

### Next step: Megatron conversion + consistency / 下一步：转 Megatron + 一致性测试

After merge + validate succeeds, the standard next step is HF → mcore
conversion plus the 6-check end-to-end HF↔mcore consistency suite. That
workflow lives in the `llava-onevision2-consistency` skill — load it with
`skill(name="llava-onevision2-consistency")`.

merge + validate 通过后，标准下一步是 HF → mcore 转换 + 6 项 HF↔mcore 端到端
一致性测试。流程在 `llava-onevision2-consistency` skill 里 ——
`skill(name="llava-onevision2-consistency")` 加载。

## Manual validation scripts (OOM fallback) / 手写脚本（OOM 后备方案）

Only use these when the built-in validators OOM (e.g. 30B+ MoE on a single
80 GB GPU). They load only the visual or language sub-component to halve
memory pressure.

只在内置 validator OOM 时用（如 30B+ MoE 单卡 80 GB）。手写脚本只加载
visual 或 language 子组件来缓解显存压力。

### Test 1: ViT Weight Consistency / ViT 权重一致性

Compare every ViT weight tensor between the original encoder and the merged checkpoint.

逐 tensor 比较原始 encoder 和合并 checkpoint 中的 ViT 权重。

```python
# Inside container, PYTHONPATH=transformers_impl:.
import torch
from safetensors.torch import load_file
import torch.nn.functional as F

merged_dir = "/train_tmp/llava_onevision2_4b_p16m33"
vit_dir = "/train_tmp/onevision_encoder_patch16_0424"

# Load weights
merged_w = {}
for f in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]:
    merged_w.update(load_file(f"{merged_dir}/{f}"))
vit_w = load_file(f"{vit_dir}/model.safetensors")

# Compare non-QKV weights (direct mapping with "visual." prefix)
for vit_key, vit_val in vit_w.items():
    if "q_proj" in vit_key or "k_proj" in vit_key or "v_proj" in vit_key:
        continue
    if "layernorm_post" in vit_key:  # dropped in merge
        continue
    merged_key = f"visual.{vit_key}"
    merged_val = merged_w[merged_key]
    cos = F.cosine_similarity(vit_val.flatten().float(), merged_val.flatten().float(), dim=0)
    assert cos > 0.9999, f"FAIL {vit_key}: cos={cos}"

# Compare QKV weights (fused: q+k+v → qkv)
for i in range(24):  # 24 encoder layers
    for suffix in ["weight", "bias"]:
        q = vit_w[f"encoder.layers.{i}.self_attn.q_proj.{suffix}"]
        k = vit_w[f"encoder.layers.{i}.self_attn.k_proj.{suffix}"]
        v = vit_w[f"encoder.layers.{i}.self_attn.v_proj.{suffix}"]
        fused = torch.cat([q, k, v], dim=0)
        merged = merged_w[f"visual.encoder.layers.{i}.self_attn.qkv.{suffix}"]
        cos = F.cosine_similarity(fused.flatten().float(), merged.flatten().float(), dim=0)
        assert cos > 0.9999, f"FAIL QKV layer {i} {suffix}: cos={cos}"

print("ViT weight consistency OK")
```

**Expected**: All cosine similarities > 0.9999. QKV may have ~1e-7 divergence due to bf16 cat.

**预期**: 所有余弦相似度 > 0.9999。QKV 因 bf16 拼接可能有 ~1e-7 的微小差异。

### Test 2: ViT Inference Consistency / ViT 推理一致性

Full forward pass through independent patch_embed → layernorm_pre → 24 encoder layers.

独立的 patch_embed → layernorm_pre → 24 encoder 层的完整前向传播。

```python
import torch, sys
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, AutoModelForCausalLM

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
merged_dir = "/train_tmp/llava_onevision2_4b_p16m33"
vit_dir = "/train_tmp/onevision_encoder_patch16_0424"
sms, patch_size = 3, 16
pixel_unit = patch_size * sms  # 48

# Use a small synthetic image (must be multiple of pixel_unit)
image = Image.new("RGB", (480, 480), color="red")
h, w = 480, 480

# Load merged model's visual component
model = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=DTYPE,
                                              low_cpu_mem_usage=True, trust_remote_code=True)
merged_visual = model.model.visual.to(DEVICE).eval()
del model.model.language_model
import gc; gc.collect()

# Load original ViT
sys.path.insert(0, vit_dir)
from onevision_encoder import OneVisionEncoderModel
orig_vit = OneVisionEncoderModel.from_pretrained(vit_dir, torch_dtype=DTYPE, trust_remote_code=True)
orig_vit = orig_vit.to(DEVICE).eval()

# Prepare pixel values
clip_proc = CLIPImageProcessor.from_pretrained(vit_dir)
clip_px = clip_proc(images=image, return_tensors="pt",
                    do_resize=False, do_center_crop=False)["pixel_values"]
clip_px = clip_px.to(dtype=DTYPE, device=DEVICE)
grid_h, grid_w = h // patch_size, w // patch_size

# Use orig_vit's full forward (row-major output)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
    orig_out = orig_vit(clip_px).last_hidden_state  # (1, N, D)

    # Merged visual: block layout forward
    from llavaonevision2.modeling_llava_onevision2_moe import convert_rope_to_block_layout_by_positions

    def extract_block_patches(img_tensor, ps, s):
        b, c, ph, pw = img_tensor.shape
        h2, w2 = ph // ps, pw // ps
        patches = img_tensor.reshape(b, c, h2, ps, w2, ps).permute(0, 2, 4, 1, 3, 5).reshape(h2, w2, c, ps, ps)
        h_m, w_m = h2 // s, w2 // s
        patches = patches.reshape(h_m, s, w_m, s, c, ps, ps).permute(0, 2, 1, 3, 4, 5, 6).contiguous()
        return patches.reshape(-1, c, ps, ps)

    block_patches = extract_block_patches(clip_px, ps=patch_size, s=sms)
    merged_pre = merged_visual.layernorm_pre(merged_visual.embeddings(block_patches).unsqueeze(0))

    # Build RoPE
    grid_thw = torch.tensor([[1, grid_h, grid_w]], device=DEVICE)
    t_idx = torch.arange(1, device=DEVICE, dtype=torch.float32)
    h_idx = torch.arange(grid_h, device=DEVICE, dtype=torch.float32)
    w_idx = torch.arange(grid_w, device=DEVICE, dtype=torch.float32)
    mt, mh, mw = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
    patch_positions = torch.stack([mt, mh, mw], dim=-1).reshape(-1, 3)
    merged_freqs = merged_visual.video_rope.forward_from_positions(patch_positions)
    merged_freqs = convert_rope_to_block_layout_by_positions(
        merged_freqs, patch_positions, spatial_merge_size=sms, grid_thw=grid_thw)
    block_rope = torch.cat([merged_freqs, merged_freqs], dim=-1).unsqueeze(0)

    # Run encoder layers
    merged_h = merged_pre
    for i in range(len(merged_visual.encoder.layers)):
        merged_h = merged_visual.encoder.layers[i](
            merged_h, attention_mask=None, rotary_pos_emb=block_rope,
            output_attentions=False, cu_seqlens=None, max_seqlen=None)[0]

    # Convert orig row-major output to block layout for comparison
    def rowmajor_to_block(features, t, h2, w2, s):
        d = features.shape[-1]
        return features.view(t, h2 // s, s, w2 // s, s, d) \
                       .permute(0, 1, 3, 2, 4, 5).contiguous().view(t * h2 * w2, d)

    orig_block = rowmajor_to_block(orig_out[0], 1, grid_h, grid_w, sms)
    cos = F.cosine_similarity(merged_h[0].flatten().float(), orig_block.flatten().float(), dim=0)
    diff = (merged_h[0] - orig_block).abs().mean().item()
    print(f"ViT inference: cos={cos:.8f}, diff={diff:.8e}")
    assert cos > 0.999, f"ViT inference mismatch: cos={cos}"
```

**Expected**: cos >= 0.999 (typically 1.0 with identical weights).

**预期**: cos >= 0.999（权重一致时通常为 1.0）。

**Note on image size**: Use small images (e.g. 480x480) to avoid GPU OOM. The image dimensions must be multiples of `patch_size * spatial_merge_size`.

**关于图像大小**: 用小图（如 480x480）避免 GPU OOM。图像尺寸必须是 `patch_size * spatial_merge_size` 的整数倍。

### Test 3: LLM Inference Consistency / LLM 推理一致性

Pure text forward pass comparing logits from the original LLM vs the merged model's language_model.

纯文本前向传播，比较原始 LLM 和合并模型的 language_model 的 logits。

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
merged_dir = "/train_tmp/llava_onevision2_4b_p16m33"
llm_dir = "/train_tmp/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
input_ids = tokenizer("Hello, world!", return_tensors="pt")["input_ids"].to(DEVICE)

# Load original LLM
orig_llm = AutoModelForCausalLM.from_pretrained(llm_dir, torch_dtype=DTYPE,
                                                 trust_remote_code=True).to(DEVICE).eval()
with torch.no_grad():
    orig_logits = orig_llm(input_ids).logits
del orig_llm
import gc; gc.collect(); torch.cuda.empty_cache()

# Load merged model's language_model
merged = AutoModelForCausalLM.from_pretrained(merged_dir, torch_dtype=DTYPE,
                                               low_cpu_mem_usage=True, trust_remote_code=True)
merged_lm = merged.model.language_model.to(DEVICE).eval()
del merged.model.visual
gc.collect()

with torch.no_grad():
    merged_logits = merged_lm(input_ids).logits

cos = F.cosine_similarity(orig_logits.flatten().float(), merged_logits.flatten().float(), dim=0)
diff = (orig_logits - merged_logits).abs().max().item()
print(f"LLM logits: cos={cos:.8f}, max_diff={diff:.8e}")
assert cos > 0.999, f"LLM logits mismatch: cos={cos}"
```

**Expected**: cos = 1.0, diff = 0.0 (LLM weights are copied verbatim, no fusion).

**预期**: cos = 1.0, diff = 0.0（LLM 权重直接复制，无融合）。

## What is NOT tested / 未覆盖的部分

| Not Tested | Reason |
|---|---|
| Vision-language joint inference (image → ViT → projector → LLM → text) | Projector is randomly initialized when no adapter is provided; no reference baseline exists |
| Multi-image / video | Only single static image tested |
| End-to-end generation quality | Requires trained adapter + evaluation benchmarks |

## Known Issues & Workarounds / 已知问题和解决方案

### 1. `blockorder` ViT validator fails with patch16+sms=3

`vit_blockorder.py` does a reshape that assumes patch14+sms=2 dimensions. Use `--vit-validator-strategy layerwise` instead.

`vit_blockorder.py` 的 reshape 假设 patch14+sms=2 的尺寸。改用 `--vit-validator-strategy layerwise`。

### 2. GPU OOM during validation

The built-in validator loads the full merged model + original ViT simultaneously. For 4B+ models on a single 80GB GPU, this may OOM. Workaround: use the standalone scripts above (they load only the visual component, deleting `language_model` first).

内置 validator 同时加载完整合并模型和原始 ViT。4B+ 模型在单张 80GB GPU 上可能 OOM。解决方案：用上面的独立脚本（只加载 visual 部分，先删 `language_model`）。

### 3. Original ViT embeddings output shape mismatch

Some ViT encoders output `(N, 1, D)` from `embeddings()` while merged visual outputs `(N, D)`. The full `model()` forward handles this internally, so use `orig_vit(pixel_values).last_hidden_state` instead of calling `embeddings()` + layers manually for the original ViT.

有些 ViT encoder 的 `embeddings()` 输出 `(N, 1, D)` 而合并后的 visual 输出 `(N, D)`。用 `orig_vit(pixel_values).last_hidden_state` 调用完整 forward 而非手动逐层调用。

### 4. Block layout conversion for comparison

Original ViT outputs features in row-major order; merged visual uses block layout (grouped by `spatial_merge_size`). Use `rowmajor_to_block()` to align before comparison.

原始 ViT 输出 row-major 顺序的特征；合并后的 visual 使用 block layout（按 `spatial_merge_size` 分组）。比较前用 `rowmajor_to_block()` 对齐。

## Quick Reference: Key Weight Mappings / 快速参考：关键权重映射

| Original ViT Key | Merged Key |
|---|---|
| `embeddings.patch_embedding.weight` | `visual.embeddings.patch_embedding.weight` |
| `embeddings.patch_embedding.bias` | `visual.embeddings.patch_embedding.bias` |
| `layernorm_pre.weight/bias` | `visual.layernorm_pre.weight/bias` |
| `encoder.layers.{i}.self_attn.q_proj.*` | (fused into) `visual.encoder.layers.{i}.self_attn.qkv.*` |
| `encoder.layers.{i}.self_attn.k_proj.*` | (fused into) `visual.encoder.layers.{i}.self_attn.qkv.*` |
| `encoder.layers.{i}.self_attn.v_proj.*` | (fused into) `visual.encoder.layers.{i}.self_attn.qkv.*` |
| `encoder.layers.{i}.self_attn.proj.*` | `visual.encoder.layers.{i}.self_attn.proj.*` |
| `encoder.layers.{i}.mlp.fc1/fc2.*` | `visual.encoder.layers.{i}.mlp.fc1/fc2.*` |
| `encoder.layers.{i}.layer_norm1/2.*` | `visual.encoder.layers.{i}.layer_norm1/2.*` |
| `layernorm_post.*` | (dropped) |

| Original LLM Key | Merged Key |
|---|---|
| `model.layers.{i}.*` | `language_model.model.layers.{i}.*` |
| `model.embed_tokens.*` | `language_model.model.embed_tokens.*` |
| `lm_head.*` | `language_model.lm_head.*` |
