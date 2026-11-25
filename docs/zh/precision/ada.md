# Ada 的执行与验证

[English](../../precision/ada.md)

Adamant 的 Ada 实现尚处于开发早期。Linux sm_89 Release 构建已在 CUDA 13.0.88、GCC 14.3.1
和警告视为错误的条件下编译通过。运行时验证尚不完备；能编译并不能确立数值正确性。
[qwen38-flash-next.md](qwen38-flash-next.md) 及与其并列的页面中的上游测量结果不是 Adamant 的结果。

## Ampere（sm_86）

`include/basalt/cuda_target.hpp` 在 8.9 之外接受 compute capability 8.6；`CMAKE_CUDA_ARCHITECTURES`
接受 `86`、`89` 或 `86;89`（默认 `89`），其他值为配置错误。`gpu "auto"` 选择每一块可见的 Ada 或
Ampere 卡；显式指定的其他能力的卡会终止启动。任何 kernel 都不使用 FP8 或 FP4 tensor 指令
（NVFP4 走整数 MMQ），因此按设计没有任何 weight pack 或 KV 格式在 8.6 上被排除。

Ampere 尚未经过验证。sm_86 构建可以编译；Ampere 硬件上仍未完成的有：

- 对照当前与 CPU 参考 kernel 的一致性测试。
- 针对 BF16 参考的 decode 路径 PPL/KLD（`basalt-bench --quality`）。
- `cp.async` copy 路径的 memcheck。
- 依 SM 数量设门槛的快速路径及其在更小卡上的回退。
- 8 GB 级卡的性能与默认调优。

## 执行改动

| 操作 | Adamant 实现 | 需要验证的数值后果 |
|---|---|---|
| MMQ prompt 专家 | 现有整数 MMA tile，不使用 programmatic dependent launch；保留全部已配置格式 | 不打算改动算术或 tile 选择。不假定跨设备相等。 |
| NVFP4 prompt 专家 | 所有 prompt 尺寸都走整数 MMQ，而非原生 FP4 tensor 指令 | 长 prompt 下激活改用 MMQ 的 q8_1 舍入，而不是 E2M1/UE4M3 舍入。这使数值相对上游的原生 FP4 路径发生变化。权重字节保持同一格式。 |
| 稠密 tensor-core 投影与 QSA 中段 | 本地异步 copy 搬运与受支持的整数 MMA 指令 | 必须对照参考实现比较缩放因子的施加与归约顺序，并测试 token 宽度不变性。 |
| 融合的超连接读取 | 用协作式异步 copy 取代 tensor-map 传输；hc8 存储与未融合实现保留 | copy/布局等价与算术不变只是设计意图，而非实测的逐位一致。选择这一读取而非未融合读取可能改变数值。 |
| 融合的 GDN 递推与归一化 | 一个 block 处理四组列，以本地共享内存交换代替分布式共享内存 | 每个 rank 的算术与 norm 部分和的顺序都设计为与参考一致。需要输出/状态一致性与同步性验证。独立的递推/归一化路径保留。 |
| 持久化 GDN 中段 | 不属于 Ada 实现；投影、递推和归一化作为独立 kernel 执行 | 路径选择与归约顺序可能与上游持久化路径不同。请评估质量，不要假定结果相同。 |
| 分阶段搬运的 GPU 专家行 | 用协作式异步 copy 与 block 屏障取代批量 copy 的完成屏障 | 权重字节与行点积算术设计为保持不变。请验证 stage 复用、尾部工作项与对等执行。 |
| 分组专家启动 | 用有序 stream 启动取代依赖启动 | 现有融合的 SwiGLU/q8_1 算术保留；请验证顺序与精确输出。 |
| 标量 CPU 量化器与激活一致性参考 | 主机编译禁用乘加收缩 | Q8_K 在整数舍入前先把乘积舍入到 fp32，与显式舍入的 SIMD 及 GPU kernel 一致。这会使标量量化相对依赖编译器收缩的构建发生变化；请验证精确的平局与极小激活。 |
| 权重预取 | 按缓存行预取取代批量预取 | 不打算改变模型算术。请独立剖析；不作任何速度主张。 |
| 原生 GR RMSNorm 权重预取 | 在支持的行宽下，先于原始输入归约加载 gamma 值，随后在输出遍使用；输入累加循环不变 | `native_gr_norm_parity` 与旧 kernel 逐位一致，并与双精度 CPU 参考比较。450 W 下的开发 A/B：2560 列，基线→预取 2.769→2.493 µs；4096 列，2.615→2.683 µs。结果随宽度变化；不作端到端速度主张。技术做法参考 NInfer Apache-2.0 提交 `9954867a`；未复制任何源码。 |
| 原生 MMQ 专家 gather 与输出 map | 每个原生专家组在其暂存槽位就绪后，由一次分组 gather 处理；identity 行的目标由专家边界推出，而不再依赖每次运行生成的 iota 缓冲 | 权重字节与 MMQ 算术不变。分组/单专家一致性测试通过；IQ4_XS 定向 memcheck 与 racecheck 报告零错误与零危险。一次针对部署包的 memcheck 完成了 511 token 的 prefill 并到达 verify capture，之后超时、未给出 sanitizer 摘要；该项未完成。 |
| 共享专家 gate/up MMVQ 成对启动 | 一次启动用相同的 Q8_1 输入处理两个同格式的 Q8_0 或 Q6_K 投影；每个输出使用现有精确布局的 MMVQ 行与归约代码 | CPU 参考与定向 GPU 一致性测试通过；最终二进制的部署包贪心检查在 3 个请求中共 900 token 通过。成对资格限制在 2–8 列、精确多列模式与共享 stream；其余调用仍走原启动方式。一次匹配的 450 W 开发 A/B 使用部署的 IQ4_XS pack，profile `c31d790a…`，7,461 个常驻槽位，CPU share 0.60，PCIe share 0.55，上下文 262144，int8 KV，spec 4，chunk 32768，seed 1：4096-token prefill 2777.34→2775.68 ms；decode 换算 152.396→153.537 ms，struct 232.277→234.923 ms，prose 152.182→152.530 ms；struct 验证窗口 17.197→17.003 ms，verify 14.096→13.905 ms（58.15→58.81 tokens/s）。这是一次包含若干小幅改动的开发配对；不能据此得出普遍加速。全模型 memcheck 未完成。因没有可用的 BF16 参考运行，BF16 质量对比尚未进行。 |
| decode 的 CPU/PCIe 未命中划分（`ADAMANT_PCIE_BALANCE=1`） | 每层拟合把固定系数 1.85 改为线性池速率系数：从 30 GB/s 时的 1.85 到 12 GB/s 时的 0.8；取整后的 PCIe 份额上限从 0.60 提高到 0.95 | 调度改变哪些专家未命中在 CPU、哪些在 GPU 上运行。专家算术不变，但累加顺序与输出数值可能改变。BF16 质量验证与 Ada 性能测量尚未完成；不作质量或速度主张。 |
| APR trellis 专家行（ggml 编号 144-149） | trellis 格式在卡上对 Q8_1 激活用 `dp4a` 与 codebook 的 int8 副本解码；被 Hadamard 折叠的层在量化前折叠 gate/up 输入，在 down 点积前折叠 SwiGLU 输出，卡上和 FP16 prompt 路径都如此 | 新的权重格式需要自己的门槛：对 BF16 参考基准的 PPL / KLD，以及实测的 pack，而不是假设的 pack。CPU 档位在各指令集上逐位实测 —— TQP7、AVX-512 每专家 596 µs、单核 3.74 GB/s，标量 3.2 ms、0.69 GB/s —— 拒绝检查不需要卡也能运行。`tq_expert_parity` 在 RTX 4060 Laptop GPU 上全部通过：GPU 专家对 double 参考基准 rel 1.7e-2 .. 1.9e-2（门槛 3e-2），decode-once 对逐 entry kernel、单 token 对同一 token 在组内均逐位相同，`hadamard_rows` 与 CPU 折叠逐位相同，`hadamard_q8_1_rows` 与折叠 + 量化逐位相同，FP16 折叠在 fp16 舍入内（rel 2.8e-4），FP16 反量化器是同一批浮点的 fp16。grouped kernel 的行，64 组 x 1 entry，笔记本不上报功率上限：TQK6/TQK7 每次 565 µs、权重 119 GB/s，TQP6/TQP6 612 µs、201 GB/s，TQP5/TQP7 680 µs、171 GB/s，TQP7/TQP5 688 µs、188 GB/s，TQ2_T/TQ2_T 554 µs、151 GB/s；每组 4 entry 时 TQK6/TQK7 是 1049 µs、64 GB/s。pack 转换完成（56.572 GiB，668 s）并能加载：27.5 GiB 专家，卡上常驻 3.5 GiB，内存 21.8 GiB，76 s 就绪。`basalt-probe` 实测冷 decode 15 .. 18 tok/s，卡的拷贝路径 237 GB/s、PCIe 1.7 GB/s、内存 55 GB/s（常驻专家 2241，KV fp16，8k 上下文），其 adapt 测试建议 `--adapt-every 64 --adapt-swaps 256 --adapt-decay 0.70`（+9%）；`basalt-bench` 的 fixture 在 conv / struct / prose 上是 19.3 / 19.0 / 20.0 tok/s，不带起草，因为这个 pack 没有起草器。这些是开发数字，笔记本读不到功率上限，且没有改动任何默认值。PPL / KLD 门槛仍未完成 —— 这台机器上没有 BF16 参考 GGUF。 |

### 0.3.1 路径

- **混合 gate/up 层**（`experts.layout` v7：UD-IQ1_S 第 18 层是 IQ2_S gate 配 IQ2_XXS up）。每一半都由同一行代码按其
  自身格式解码；decode glue 与持久化对等 kernel 有混合构建，MMQ prompt 槽位按两种类型的整块补齐，CPU 层让每一半走
  其类型的行 kernel。设计为与统一 kernel 逐位一致；`mixed_gu_parity` 在 sm_89 上逐位通过。
- **对等 prompt 共享**（`prefill::PeerShare`）。第二张卡用与主卡相同的 MMQ 格式和相同的 q8_1 激活，从暂存的分组
  buffer 计算其专家的 prompt 行，再把行送回。设计为与主卡的行逐位一致；上游 `peer_share_parity` 检查 gather 的字节。
  `peer_share_parity` 与 `peer_experts_parity` 需要两张 GPU，单卡上跳过；未在 Ada 上测量。不携带 FP4 流式 chunk 回退，
  因此达到或超过 `stream_all_min()` 的 chunk 仍留在主卡。
- **glue 路径格式**（Q5_K/Q6_K gate/up 行、IQ4_XS gate/up 行、(IQ3_S|IQ4_XS)/Q8_0 对）。行点积与累加顺序相同，只融合
  启动；`dec_moe_parity --kq` 与 `native_grouped_parity --glue` 在 sm_89 上逐位通过。
- **批量 K/V 流式**（`kv_stream` 的分段 resolve，`--kv-resident` 配 `--batch`）。相同的行经分段游标读取；
  `kv_stream_parity` 在 sm_89 上通过。

在部署的 IQ4_XS pack 上，64 个贪心 token（prose ids、int8 KV、expert cache 700、swaps 0、单张 450 W 的 RTX 4090）在
合并前的构建与本构建之间逐位相同。

## prompt 显存与对等执行

整数 MMQ 会为所有路由行设置中间 buffer 的大小。因此，与原生 FP4 实现相比，大的 NVFP4
prompt chunk 可能需要多得多的暂存内存；自动拟合可能因此选择更小的 chunk。请用真实分配尺寸
和长 prompt 来验证，而不是从短样例外推。只有仅路由的 prompt chunk 会与对等卡共享；达到或超过
`stream_all_min()` 的 chunk 留在主卡上。对等独占流式与常规对等放置都需要在第二张卡上分别覆盖。

prompt 切分保留了它基于权重元数据、独立于已移除的执行 API 的 NVFP4 pack 资格检查。其继承
的成本模型与 chunk 阈值没有针对 Ada 上的整数 MMQ 标定过；切分的正确性与性能收益都尚未
确立。

## 所需证据

[贡献要求](../../../CONTRIBUTING.zh-CN.md)适用，尤其是每个新增或改动的 kernel 都需要的一致性
测试。现有 kernel 与 prompt 路径的一致性测试必须对照当前参考 kernel 和独立 CPU 参考通过，
包括非有限输出拒绝、模型几何、窗口宽度和 token 位置。

改动的 copy 与同步路径必须做内存、竞争、初始化和同步检查。资源与图捕获行为必须在目标 GPU
上验证，而不是由一次通过的主机测试推断。

移植完成要求针对 BF16 参考做 decode 路径的 PPL/KLD 与 top-1 一致率
（[方法](quality.md#bf16-参考基准)），外加 prefill/长上下文以及 MTP 开/关检查。报告结果时
请给出确切的模型与样例标识、固定的专家放置、参考的出处、配置，以及施加的 GPU 功率上限。
不要把短时质量样例当作对所有 prompt 路径的覆盖。

门槛尚未完成。kernel 一致性与内存、竞争、初始化、同步检查已在 sm_89 上通过；端到端质量
结果与性能主张尚未确立。目前已取得的发展测量连同其功率上限与配置列在
[README](../../../README.zh-CN.md#ada-上的实测)。
