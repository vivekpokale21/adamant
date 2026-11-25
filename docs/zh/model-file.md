# `.basalt` 模型文件

[English](../model-file.md)

一个模型就是一个文件。它装着引擎、服务器与 vision sidecar 要加载的一切：权重、tokenizer 与聊天模板、MTP drafter、8-bit 超连接副本、vision projector，以及带默认采样设置的 GGUF 元数据。它不带任何属于引擎策略的东西：没有 flag（那些来自 `--config` 或命令行），也没有专家 profile（那是引擎自己的）。

- **布局：** 一个 64 KiB 的头部（magic `BASALTMF`、版本 2、至多 511 个带偏移、大小与 SHA-256 的命名 section），其后是按 2 MiB 对齐的各 section。权威描述：[model_format.hpp](../../include/basalt/core/model_format.hpp)。
- **版本：** 引擎只读版本 2。版本 1 的文件可用 [tools/migrations/basalt_v1_to_v2.py](../../tools/migrations/README.md) 转换，无需重新下载。改了什么：[文件格式变更日志](basalt-file-changelog.md)。

## 从 GGUF 构建 pack

```sh
python3 tools/convert/convert.py --gguf MODEL-00001-of-0000N.gguf --mmproj MMPROJ.gguf --out MODEL.basalt \
    [--mtp-source DIR [--mtp-dense FILE]] [--draft-vocab FILE] [--compat-bf16] [--name NAME]
```

`--help` 说明每个输入。`--compat-bf16` 用于转换以其他类型存储 BF16 投影的 GGUF。这样构建、又没有官方调校 drafter 的 pack 会比官方文件慢。

从分片下载中截出的单个文件会保留那次下载的 `split.count` 与 `split.tensors.count`，它们描述的是来源模型而非这个文件。当它声明的每个 tensor 都落在文件之内时，它按完整模型打开；否则以被截断为由拒绝；分片模型的各个分片仍会按这些键校验。

## Packs

- `dense`：F32 与 BF16 tensor。`gguf`：量化的非专家 tensor。`experts`：每层的路由专家。
- `experts.layout`：每层 blob 的起点、大小与格式（专家被 Hadamard 折叠时为 v8）。
- `experts.hadamard`：只存在于专家以 Hadamard 旋转基存储的 GGUF（`prism.hadamard.*`）：旋转块、±1 符号向量，以及每层被折叠的角色。
- `engram`：仅 Engram 表（144 字节的 IQ4_NL 行），按 2 MiB 对齐。
- `vision/mmproj`：只在用 `--mmproj` 构建时才有；引擎与 `basalt-vision --model FILE.basalt` 都就地读取它。

## Trellis（APR）pack

Agention 的 APR GGUF 把路由专家存成 trellis 编码的权重（ggml 类型 `TQ2_T`、`TQK6`、`TQK7`、`TQP5`、`TQP6`、`TQP7`，编号 144-149）：一个 block 是 128 个权重、一个 fp16 scale，加上 trellis 步骤组成的环形位流；每个步骤的状态经哈希映射到 2048 点 fp16 codebook 中的一个或两个点。它们和其他专家格式一样逐字节搬进 pack，CPU 上的行计算走引擎自己的 kernel（[tq_cpu.hpp](../../include/basalt/kernels/cpu/tq_cpu.hpp)），激活为 Q8_0。格式、codebook 与折叠见 [trellis](trellis.md)。

`Gyro` GGUF 还会折叠它的专家：存下的权重是每 `b` 个输入列一组的 `W diag(s) H_b^T`，所以进入被折叠权重的激活必须先经过 `y = H_b (s · x)`。`convert.py` 读取 `prism.hadamard.*` 元数据，写成 `experts.hadamard` section，并把 `experts.layout` 写成 v8；v8 布局没有该 section（或反过来）会在启动时被拒绝，在非 trellis 类型上折叠也会被拒绝，因为那些 kernel 量化的是未旋转的输入。折叠按层生效，所以被折叠的层 gate 与 up 必须同类型。

```sh
python3 tools/convert/convert.py --gguf MODEL-TQ1_0.gguf --sections-from BASE.basalt --compat-bf16 --out MODEL.basalt
```

格式定义在 [ggml_common_tq.h](../../include/basalt/ggml/ggml_common_tq.h)，旋转在 [hadamard.hpp](../../include/basalt/kernels/cpu/hadamard.hpp)；`tests/kernels/cpu/tq_cpu_test.cpp` 从格式的书面描述重新推导一遍，并检查每个指令集给出同样的位；`tests/kernels/cpu/hadamard_test.cpp` 检查旋转与 pack 的拒绝条件。

## Flash-Next NVFP4 pack

官方文件是 NVIDIA 的 checkpoint（[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)，修订 `fc694b54`），稠密投影为 Q6_K。转换器直接读取 safetensors：专家逐位搬移，稠密 tensor 从 BF16 量化一次。

```sh
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54 --local-dir nvidia

# 6-bit dense (the official file)
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia \
    --out Qwen3.8-FN-NVIDIA-NVFP4.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"

# 8-bit dense variant
python3 tools/convert/convert.py --sections-from Qwen3.8-FN-UD-Q4_K_XL.basalt --checkpoint nvidia --dense-q8-0 \
    --out Qwen3.8-FN-NVIDIA-NVFP4-Q8.basalt --name "Qwen3.8-Flash-Next NVIDIA-NVFP4"
```

`--mtp-dense FILE`（官方 NVFP4、UD-IQ1_M、UD-IQ2_XS 或 UD-IQ3_XXS pack 的 `mtp/dense` 字节）会换入一个调校过的 drafter 层；转换器把它记录在 `meta.mtp.dense` 中。输出约 122 GiB（8-bit 变体为 123 GiB）。基础词表必须是 NVIDIA 的。
