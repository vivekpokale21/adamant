# Trellis (APR) experts and the Hadamard fold

[简体中文](zh/trellis.md)

Trellis is a 3-bit family of expert weight formats from [trellis-weight](https://github.com/virtueer/trellis-weight)
(the APR scheme: Adaptive Permutation Reconstruction).  A trellis tensor holds no per-value codes to read one by one:
it holds an index stream, and the value at a position comes from where that index lands on a permutation.  Adamant
supports the six types as expert weights: `GGML_TYPE_TQ2_T` (144), `GGML_TYPE_TQK6` (145), `GGML_TYPE_TQK7` (146),
`GGML_TYPE_TQP5` (147), `GGML_TYPE_TQP6` (148), `GGML_TYPE_TQP7` (149).  They are expert-only formats: a dense tensor
in one of them is refused at load, as every other expert-only type is.

## The blob

One 128-value block is `[fp16 scale][the block's index bytes]`, and a matrix row is `n / 128` of them:

| type | index bytes per 128-value block | bytes per 128 values |
| --- | --- | --- |
| `TQ2_T` | 24 (one byte per value) | 26 |
| `TQK6` | 24 (six bits per value) | 26 |
| `TQK7` | 28 (seven bits per value) | 30 |
| `TQP5` | 80 (five bits, one byte per value) | 82 |
| `TQP6` | 96 (six bits packed) | 98 |
| `TQP7` | 112 (seven bits packed) | 114 |

An expert blob is the engine's layout: `n_ff` gate rows, then `n_ff` up rows, then `n_embd` down rows, the gate and
up rows sharing one type (`iq_row_bytes` gives each row's size).  At the model's geometry a `TQK6` gate/up row is 520
bytes, a `TQK7` down row 150, and an expert 1,049,600 bytes - so the experts of one layer of a 128-expert model are
537,397,200 bytes, and the 25.8 GB of such weights sit in RAM at 3.13 bits per weight.

## The index stream, and the state

`TQP5`, `TQP6`, `TQP7` carry a permutation state with the indices: 15 bits per value, 10 bits of a permutation step
and 5 bits of an increment.  One step permutes the block once and costs one multiply, and the stream reconstructs
itself by walking the permutation forward.  `TQK6` and `TQK7` carry the indices of a fixed Kruskal-Hadamard
construction instead (six and seven bits per value), and `TQ2_T` one byte per value.

The reconstruction is exact in int32: every value becomes an integer point, the dot product over a sub-block sums in
int32, and the block scale multiplies the result.  Both tiers do it the same way, so a CPU dot and a GPU dot over the
same bytes and the same quantized activations are bit for bit the same - which is what
`tests/kernels/tq_expert_parity.cpp` checks, against a double-precision reference.

## The codebook

One fp16 codebook of 2048 points (`tq_lut_f16`) serves all six types.  The CPU reads it directly; the GPU reads an
int8 copy of it (`tq_lut_i8.cuh`, 4 KB) and sums against the Q8_1 activation with `dp4a`.  The int8 copy is the
codebook at 3.984375/127 with the worst value off by 0.0157 - 0.0085 of the codebook's own spread - so the decode on
the card matches the CPU's to within its activation rounding.

## The fold

Some trellis quants fold the matrices with a Hadamard matrix before quantizing, which moves the weight error out of
the few large values into the whole row.  A folded layer can only be read with the same fold applied to its inputs:

- the gate/up input: `h = H_128 (s * x)` for every 128-value block of the activation row, then quantized to Q8_0
  (CPU tier) or Q8_1 (GPU kernels).  On the GPU the rotation and the quantization are one kernel,
  `hadamard_q8_1_rows`.
- the SwiGLU output: the same fold over `n_ff` before the down product.  On the GPU it is fused into the SwiGLU
  kernel, `swiglu_hadamard_q8_1_entries_kernel`, so the activation is written once and never leaves the block.

The fold parameters come from the GGUF metadata, not from a guess: `prism.hadamard.enabled`,
`prism.hadamard.block_size`, `prism.hadamard.fold_layers` (which layers, `-1` for all), and
`prism.hadamard.sign_widths` with a `+1/-1` vector per width (`prism.hadamard.sign_<width>`).  The loader parses them
(`hadamard_spec_from_text`) and the converter writes them into the pack's `experts.hadamard` section, which is part of
expert layout version 8 (`kExpertLayoutVersion`).  A layer's fold is applied on CPU by `hadamard_rotate`; on the GPU
the same signs are uploaded once per host vector by `hadamard_device_signs`, and the layout resolves them in
`native_expert_layout`.

## What refuses what

- **a fold on a non-trellis format**: refused at layout build.  The fold has to match how the weights were written.
- **a folded layer with a SwiGLU limit**: refused at load.  The fused kernel rotates after SwiGLU, so the limit would
  land on the wrong values - the unrotated kernel would have to rotate first, which is a different function.
- **a folded layer whose widths are not the block**: refused (`prism.hadamard.block_size` must divide the row widths).
- **a dense tensor in a trellis type**: the pack refuses it at load (`register_gguf_view`), as every expert-only format does.
- **MMQ**: `mmq::supported` says no for the trellis types, so a prompt chunk reads them through the FP16 path
  (`dequant_f16` on the card, `tq_dequant_row` on CPU).  The FP16 dequantizer is bit-checked against the CPU decode.
- **the peer card**: the peer's persistent kernel quantizes an unrotated activation, so folded layers stay off it.
- **the device-planned MoE tail**: it quantizes the layer's hidden state without the fold, so folded layers stay off
  it (`dec_moe_formats_ok` returns false for the trellis types).  `dm_init` turns the tail off for a pack whose layers
  are folded and says why; the experts then run through the grouped kernels, which rotate.

## Where the kernels live

- CPU: `src/kernels/cpu/tq_cpu.cpp` (scalar), `tq_avx2.cpp`, `tq_avx512.cpp`, dispatched by `tq_rows`; the decoder
  itself is `tq_decode_block` in `include/basalt/ggml/ggml_common_tq.h`, shared by every tier.
- GPU: the trellis `Fmt` and `Split` cases in `src/kernels/cuda/iq_kernels.cu`, the decode-once expert kernels
  (`native_expert_grouped`, which includes the fused SwiGLU + rotation), the rotation kernels, and the FP16
  dequantizer.
- Prompt path: `src/models/qwen4exp/prefill/expert_fp16.cpp` rotates the gathered activation rows and the SwiGLU
  outputs when the layer is folded; `expert_mmq.cpp` never sees a trellis layer (see the refusals above).
