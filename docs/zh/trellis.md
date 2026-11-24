# Trellis(APR)专家权重与 Hadamard 折叠

[English](../trellis.md)

Trellis 是 [trellis-weight](https://github.com/virtueer/trellis-weight) 提供的一组 3-bit 专家权重格式(APR:
Adaptive Permutation Reconstruction,自适应置换重构)。trellis 张量不存放逐个读取的数值编码:它存放一条索引流,
某个位置的值取决于该索引落在置换的哪里。Adamant 支持六种类型作为专家权重:`GGML_TYPE_TQ2_T`(144)、
`GGML_TYPE_TQK6`(145)、`GGML_TYPE_TQK7`(146)、`GGML_TYPE_TQP5`(147)、`GGML_TYPE_TQP6`(148)、
`GGML_TYPE_TQP7`(149)。它们只能用作专家权重:任何稠密张量使用这些类型都会在加载时被拒绝,与其他仅限专家的格式一样。

## blob

一个 128 值的块是 `[fp16 缩放][该块的索引字节]`,矩阵的一行由 `n / 128` 个这样的块组成:

| 类型 | 每 128 值块的索引字节 | 每 128 值字节数 |
| --- | --- | --- |
| `TQ2_T` | 24(每值 1 字节) | 26 |
| `TQK6` | 24(每值 6 bit) | 26 |
| `TQK7` | 28(每值 7 bit) | 30 |
| `TQP5` | 80(5 bit,每值 1 字节) | 82 |
| `TQP6` | 96(6 bit 打包) | 98 |
| `TQP7` | 112(7 bit 打包) | 114 |

专家 blob 沿用引擎的布局:`n_ff` 行 gate,接着 `n_ff` 行 up,gate 与 up 共用同一类型,然后 `n_embd` 行 down
(每行字节数由 `iq_row_bytes` 给出)。在本模型的几何下,`TQK6` 的 gate/up 行是 520 字节,`TQK7` 的 down 行是
150 字节,一个专家是 1,049,600 字节 —— 所以一个 128 专家模型的一层是 537,397,200 字节,这 25.8 GB 权重在内存里
是每权重 3.13 bit。

## 索引流与 state

`TQP5`、`TQP6`、`TQP7` 随索引带一份置换 state:每值 15 bit,其中 10 bit 是置换步长,5 bit 是增量。一次置换对块做
一次置换,代价是一次乘法,索引流通过沿置换向前走来自我重建。`TQK6` 与 `TQK7` 存放固定 Kruskal-Hadamard 构造的
索引(每值 6 与 7 bit),`TQ2_T` 每值 1 字节。

重建是 int32 精确的:每个值变成一个整数点,子块的点积在 int32 中求和,再乘块缩放。两个档位都以同一方式完成,因此
CPU 与 GPU 对同一字节、同一量化激活的点积逐位相同 —— 这正是 `tests/kernels/tq_expert_parity.cpp` 检查的内容,
参照基准是 double 精度的参考实现。

## codebook

一份 2048 个 fp16 点的 codebook(`tq_lut_f16`)服务全部六种类型。CPU 直接读它;GPU 读它的 int8 副本
(`tq_lut_i8.cuh`,4 KB),并用 `dp4a` 与 Q8_1 激活求和。int8 副本是 codebook 按 3.984375/127 缩放的结果,最大偏差
0.0157 —— 是 codebook 自身散布的 0.0085 —— 因此卡上的解码与 CPU 的差别只来自激活的舍入。

## 折叠

部分 trellis 量化在量化前用 Hadamard 矩阵折叠权重矩阵,把权重误差从少数大值摊开到整行。折叠过的层只有在对输入施加
同一折叠时才能读取:

- gate/up 输入:对激活行的每个 128 值块做 `h = H_128 (s * x)`,再量化成 Q8_0(CPU 档位)或 Q8_1(GPU kernel)。
  GPU 上折叠与量化是一个 kernel:`hadamard_q8_1_rows`。
- SwiGLU 输出:在 down 点积之前对 `n_ff` 做同样折叠。GPU 上它融进 SwiGLU kernel
  (`swiglu_hadamard_q8_1_entries_kernel`),激活只写一次,也不离开 block。

折叠参数来自 GGUF 元数据,而不是猜测:`prism.hadamard.enabled`、`prism.hadamard.block_size`、
`prism.hadamard.fold_layers`(哪些层,-1 表示全部)、`prism.hadamard.sign_widths`,以及每个宽度的 `+1/-1` 向量
(`prism.hadamard.sign_<width>`)。加载器解析它们(`hadamard_spec_from_text`),转换器把它们写进包的
`experts.hadamard` 段,这是专家布局版本 8(`kExpertLayoutVersion`)的一部分。CPU 用 `hadamard_rotate` 施加某层的
折叠;GPU 上同一组符号由 `hadamard_device_signs` 每个宿主向量上传一次,布局在 `native_expert_layout` 中解析它们。

## 谁拒绝谁

- **非 trellis 格式上的折叠**:在构建布局时拒绝。折叠必须与权重写入时的方式一致。
- **带 SwiGLU 上限的折叠层**:加载时拒绝。融合的 kernel 在 SwiGLU 之后才折叠,上限会落在错误的数值上 —— 未折叠的
  kernel 必须先折叠再截断,那是另一个函数。
- **宽度不被块整除的折叠层**:拒绝(`prism.hadamard.block_size` 必须整除行宽)。
- **trellis 类型的稠密张量**:拒绝,同其他仅限专家的格式。
- **MMQ**:`mmq::supported` 对 trellis 类型返回否,因此 prompt 分块走 FP16 路径读取它们(卡上用 `dequant_f16`,
  CPU 用 `tq_dequant_row`)。FP16 反量化器与 CPU 解码逐位对照。
- **peer 卡**:peer 的常驻 kernel 量化的是未折叠的激活,所以折叠层不上 peer。
- **设备规划的 MoE 尾段**:它对该层隐状态量化时不做折叠,所以折叠层不上尾段(`dec_moe_formats_ok` 对 trellis 类型
  返回 false)。
- **`--device-plan` 关闭**:折叠的包在 device plan 关闭时加载被拒绝,因为到达 kernel 的专家输入没有被折叠。

## kernel 在哪里

- CPU:`src/kernels/cpu/tq_cpu.cpp`(标量)、`tq_avx2.cpp`、`tq_avx512.cpp`,由 `tq_rows` 分发;解码器本身是
  `include/basalt/ggml/ggml_common_tq.h` 里的 `tq_decode_block`,所有档位共用。
- GPU:`src/kernels/cuda/iq_kernels.cu` 中 trellis 的 `Fmt` 与 `Split` 分支、decode-once 专家 kernel
  (`native_expert_grouped`,含融合的 SwiGLU + 折叠)、折叠 kernel、以及 FP16 反量化器。
- prompt 路径:`src/models/qwen4exp/prefill/expert_fp16.cpp` 在该层折叠时折叠收集到的激活行与 SwiGLU 输出;
  `expert_mmq.cpp` 不会遇到 trellis 层(见上面的拒绝列表)。
