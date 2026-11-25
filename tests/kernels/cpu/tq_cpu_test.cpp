// tests/kernels/cpu/tq_cpu_test.cpp - the APR trellis expert rows (TQ2_T, TQK6, TQK7, TQP5, TQP6, TQP7) on the CPU.
//
//     build/tq_cpu_test [--bench] [--print]
//
// The format contract (include/basalt/ggml/ggml_common_tq.h) is re-derived here from its documented text, not from
// the kernels: a block of 128 weights is an fp16 scale plus a circular bit stream of trellis steps, each step's state
// hashed to one or two points of the fp16 codebook, times the scale.  (a) `tq_dequant_row` must reproduce that decode
// bit for bit - the kernels' state walk, hash and codebook are the same thing the header describes.  (b) every ISA
// the CPU runs must give the same bits for a dot product, and the same bits whether a token shares the call with
// others or not (greedy output may not depend on the verify window's size).  (c) the rows against a double reference
// differ only by the kernels' rounding: the int16 state table and the float accumulator order.  (d) the gate/up rows
// are the SwiGLU of the two dot rows, and `native_gu_rows` / `native_down_rows` dispatch to the same kernels as
// `tq_gu_rows` / `tq_rows`.  (e) the geometry: row and blob sizes from the block structs, and the refusals of
// `native_fmt` for a layer the trellis kernels cannot serve.
//
// `--print` prints the frozen row hashes (the decode may change, the bits may not); `--bench` one row range per type
// on each ISA this CPU runs.
#include "basalt/ggml/ggml.h"
#define GGML_COMMON_DECL_CPP   // the block layouts as C++ structs (basalt/ggml/ggml_common.h)
#include "basalt/ggml/ggml_common.h"
#include "basalt/ggml/ggml_common_tq.h"
#include "basalt/kernels/cpu/expert.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/tq_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {
constexpr int H = 2560, FF = 640;          // the expert widths of every shipped model
constexpr int NT = 3;                       // tokens per call in the checks
constexpr int MAXT = cpu::MAXT;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct Stats {
    double sum_abs = 0, sum_abs_ref = 0, max_abs = 0;
    void add(double got, double want) {
        const double a = std::fabs(got - want);
        sum_abs += a;
        sum_abs_ref += std::fabs(want);
        max_abs = std::max(max_abs, a);
    }
    double rel() const { return sum_abs / (sum_abs_ref + 1e-30); }
};

const char* type_name(int t) {
    static const char* names[] = { "TQ2_T", "TQK6", "TQK7", "TQP5", "TQP6", "TQP7" };
    return names[t - 144];
}

// ---------------------------------------------------------------- the format, from the header's text
struct TqSpec { int kbits; int steps; bool pair; };
TqSpec spec(int t) {
    switch (t) {
        case 144: return { 8, 32, false };   // byte-aligned: one byte per 4 weights
        case 145: return { 6, 32, false };
        case 146: return { 7, 32, false };
        case 147: return { 5, 64, true };
        case 148: return { 6, 64, true };
        default:  return { 7, 64, true };
    }
}

size_t block_bytes(int t) {
    switch (t) {
        case 144: return sizeof(block_tq2_t);
        case 145: return sizeof(block_tqk6);
        case 146: return sizeof(block_tqk7);
        case 147: return sizeof(block_tqp5);
        case 148: return sizeof(block_tqp6);
        default:  return sizeof(block_tqp7);
    }
}

// One 128-weight block decoded in float from the header's description: the scale, then each step's state read out of
// the circular LSB-first stream, hashed, and looked up in the fp16 codebook.
void decode_block_ref(const uint8_t* b, int t, float* y) {
    const TqSpec s = spec(t);
    const int total_bits = s.steps * s.kbits;
    const uint8_t* qs = b + 2;
    const float d = ggml_fp16_to_fp32((uint16_t) (b[0] | (b[1] << 8)));
    const auto bit = [&](int p) -> uint32_t { return (qs[p / 8] >> (p % 8)) & 1u; };
    for (int st = 0; st < s.steps; ++st) {
        uint32_t state;
        if (s.kbits == 8) {
            // TQ2_T is byte-aligned: the state is qs[(t + 31) % 32] as the high byte and qs[t] as the low one
            state = (uint32_t) qs[(st + 31) & 31] << 8 | (uint32_t) qs[st];
        } else {
            const int start = (s.pair ? (63 - st) : (31 - st)) * s.kbits;   // wraps over the stream
            state = 0;
            for (int j = 0; j < 16; ++j) state |= bit((start + j) % total_bits) << j;
        }
        const uint32_t x = (state & TQ_STATE_MASK) * 0x9e3779b1u;
        const int p = (int) (x >> 21);
        const int q = (int) ((x >> 10) & 2047u);
        if (s.pair) {
            y[2 * st + 0] = d * ggml_fp16_to_fp32(tq_lut_f16[2 * p + 0]);
            y[2 * st + 1] = d * ggml_fp16_to_fp32(tq_lut_f16[2 * p + 1]);
        } else {
            y[4 * st + 0] = d * ggml_fp16_to_fp32(tq_lut_f16[2 * p + 0]);
            y[4 * st + 1] = d * ggml_fp16_to_fp32(tq_lut_f16[2 * p + 1]);
            y[4 * st + 2] = d * ggml_fp16_to_fp32(tq_lut_f16[2 * q + 0]);
            y[4 * st + 3] = d * ggml_fp16_to_fp32(tq_lut_f16[2 * q + 1]);
        }
    }
}

void decode_row_ref(const uint8_t* w, size_t row_bytes, int n, int t, std::vector<float>& y) {
    y.resize((size_t) n);
    for (int i = 0; i < n / 128; ++i) decode_block_ref(w + (size_t) i * block_bytes(t), t, y.data() + 128 * i);
    (void) row_bytes;
}

// ---------------------------------------------------------------- blobs and activations
// Random bytes with a valid small fp16 scale at the start of each block; mode 1 pins the stream to its extreme codes.
void fill(std::vector<uint8_t>& v, int type, uint64_t seed, int mode) {
    std::mt19937_64 rng(seed);
    const size_t bs = block_bytes(type);
    for (size_t b = 0; b + bs <= v.size(); b += bs) {
        const uint16_t d = (uint16_t) (0x2800 + (rng() & 0xff));   // a scale around 1.0..1.3
        std::memcpy(&v[b], &d, 2);
        for (size_t i = 2; i < bs; i += 8) {
            const uint64_t x = mode == 1 ? ~0ull : rng();
            if (i + 8 <= bs) std::memcpy(v.data() + b + i, &x, 8);
            else for (size_t j = i; j < bs; ++j) v[b + j] = (uint8_t) (x >> (8 * (j - i)));
        }
    }
}

// x (n floats) -> Q8_0, through the engine's own quantizer, and the same bytes decoded in double
struct Act {
    std::vector<uint8_t> bytes;
    std::vector<double> v;
};
Act q8_0_row(const cpu::NativeFmt& f, const std::vector<float>& x) {
    Act a;
    a.bytes.resize(f.act_bytes);
    cpu::native_quant_act(f, x.data(), a.bytes.data());
    const int nblk = (int) x.size() / 32;
    a.v.resize(x.size());
    for (int i = 0; i < nblk; ++i) {
        // this engine's block_q8_0 keeps its scale as fp16 (basalt/ggml/ggml_common.h)
        const block_q8_0* b = (const block_q8_0*) a.bytes.data() + i;
        const double sc = (double) ggml_fp16_to_fp32(b->d);
        for (int j = 0; j < 32; ++j) a.v[32 * i + j] = sc * (double) (int) b->qs[j];
    }
    return a;
}

std::vector<float> random_row(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> x((size_t) n);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (float& v : x) v = u(rng);
    return x;
}

uint64_t hash_row(const float* v, int n) {
    uint64_t h = 0x9e3779b1u;
    for (int i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        h = h * 1099511620805ull + bits;
    }
    return h;
}

// ---------------------------------------------------------------- (a) the decode against the header's text
void decode_matches_format() {
    for (int t = 144; t <= 149; ++t) {
        std::vector<uint8_t> w((size_t) H * block_bytes(t));
        fill(w, t, 0x1234 + (uint64_t) t, 0);
        std::vector<float> got((size_t) H), want;
        cpu::tq_dequant_row(t, w.data(), H, got.data());
        decode_row_ref(w.data(), w.size(), H, t, want);
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < want.size(); ++i) same = got[i] == want[i];
        check(same, std::string(type_name(t)) + ": tq_dequant_row is d * codebook point, bit for bit");
    }
}

// ---------------------------------------------------------------- (b) every ISA, same bits; (c) against double
struct Rows {
    int type;
    size_t row_bytes;
    int n;
    std::vector<uint8_t> w;
    std::vector<Act> acts;
    std::vector<std::vector<double>> ref;   // per token, per row
};

Rows make_rows(int t, int n, uint64_t seed, int nt) {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(t, t, n, n, f, err)) {   // a square layer: enough for the dot checks
        check(false, std::string(type_name(t)) + ": native_fmt: " + err);
        return {};
    }
    Rows r;
    r.type = t;
    r.row_bytes = f.gu_row;
    r.n = n;
    r.w.resize((size_t) n * f.gu_row);
    fill(r.w, t, seed, 0);
    for (int i = 0; i < nt; ++i) r.acts.push_back(q8_0_row(f, random_row(n, seed + 0x9000 + i)));
    r.ref.assign((size_t) nt, std::vector<double>((size_t) n));
    std::vector<float> w;
    decode_row_ref(r.w.data(), r.w.size(), n, t, w);
    for (int i = 0; i < nt; ++i) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += (double) w[(size_t) k] * r.acts[(size_t) i].v[(size_t) k];
        r.ref[(size_t) i][0] = s;   // one row is enough: the reference dot of row 0
    }
    return r;
}

void rows_isas_and_reference() {
    const cpu::TqIsa best = cpu::tq_isa();
    check(best != cpu::TqIsa::scalar || cpu::tq_isa_ok(cpu::TqIsa::scalar), "tq_isa: scalar is always runnable");
    for (int t = 144; t <= 149; ++t) {
        const Rows r = make_rows(t, H, 0x77 + (uint64_t) t, NT);
        if (r.n == 0) continue;
        std::vector<std::vector<float>> per_isa;
        Stats st;
        for (int isa = 0; isa <= 2; ++isa) {
            const cpu::TqIsa k = (cpu::TqIsa) isa;
            if (!cpu::tq_isa_ok(k)) continue;
            std::vector<const void*> act((size_t) NT);
            for (int i = 0; i < NT; ++i) act[(size_t) i] = r.acts[(size_t) i].bytes.data();
            std::vector<float> out((size_t) NT * (size_t) r.n);
            std::vector<float*> op((size_t) NT);
            for (int i = 0; i < NT; ++i) op[(size_t) i] = out.data() + (size_t) i * (size_t) r.n;
            cpu::tq_rows(k, t, r.w.data(), r.row_bytes, r.n, act.data(), NT, op.data(), 0, 1);
            for (int i = 0; i < NT; ++i) st.add((double) out[(size_t) i * (size_t) r.n], r.ref[(size_t) i][0]);
            per_isa.push_back(out);
        }
        bool same = true;
        for (size_t i = 1; same && i < per_isa.size(); ++i) same = per_isa[i] == per_isa[0];
        check(same, std::string(type_name(t)) + ": every ISA this CPU runs gives the same bits (" +
                        std::to_string(per_isa.size()) + " ISAs, best " + cpu::tq_isa_name(best) + ")");
        check(st.rel() < 1e-2, std::string(type_name(t)) + ": against a double dot, relative L1 " +
                                   std::to_string(st.rel()) + " < 1e-2");

        // a token alone gets the same bits as in a group of MAXT
        std::vector<uint8_t> w2((size_t) H * r.row_bytes);
        std::memcpy(w2.data(), r.w.data(), w2.size());
        std::vector<float> alone((size_t) H), in_group((size_t) MAXT * (size_t) H);
        std::vector<const void*> a1{ r.acts[0].bytes.data() };
        std::vector<float*> o1{ alone.data() };
        cpu::tq_rows(best, t, w2.data(), r.row_bytes, H, a1.data(), 1, o1.data(), 0, H);
        std::vector<const void*> aN((size_t) MAXT);
        for (int i = 0; i < MAXT; ++i) aN[(size_t) i] = r.acts[(size_t) i % NT].bytes.data();
        std::vector<float*> oN((size_t) MAXT);
        for (int i = 0; i < MAXT; ++i) oN[(size_t) i] = in_group.data() + (size_t) i * (size_t) H;
        cpu::tq_rows(best, t, w2.data(), r.row_bytes, H, aN.data(), MAXT, oN.data(), 0, H);
        check(std::memcmp(alone.data(), in_group.data(), (size_t) H * sizeof(float)) == 0,
              std::string(type_name(t)) + ": a token alone is the same bits as in a group of " + std::to_string(MAXT));
    }
}

// One row of `n` weights of a trellis type, dotted against a double activation, in double.
// The row's own decode comes from the format's text (decode_block_ref), so this is an independent reference.
double dot_row(const uint8_t* row, int t, int n, const std::vector<double>& av) {
    std::vector<float> w;
    decode_row_ref(row, (size_t) n, n, t, w);
    double s = 0;
    for (int k = 0; k < n; ++k) s += (double) w[(size_t) k] * av[(size_t) k];
    return s;
}

// ---------------------------------------------------------------- (d) gate/up rows, and the dispatch
void gu_rows() {
    const cpu::TqIsa best = cpu::tq_isa();
    for (int t = 144; t <= 149; ++t) {
        cpu::NativeFmt f;
        std::string err;
        if (!cpu::native_fmt(t, 146, H, FF, f, err)) {
            check(false, std::string(type_name(t)) + " gate with TQK7 down: native_fmt: " + err);
            continue;
        }
        std::vector<uint8_t> blob(f.bytes);
        fill(blob, t, 0x4000 + (uint64_t) t, 0);
        // the down slice is TQK7: fill it separately
        std::vector<uint8_t> dn(f.d_row * (size_t) H);
        fill(dn, 146, 0x9000 + (uint64_t) t, 0);
        std::memcpy(blob.data() + f.down_off, dn.data(), dn.size());

        const std::vector<float> x = random_row(H, 0x55 + t);
        Act act = q8_0_row(f, x);
        std::vector<const void*> a1{ act.bytes.data() };

        std::vector<float> ff((size_t) FF), ff2((size_t) FF);
        std::vector<float*> fp{ ff.data() }, fp2{ ff2.data() };
        // `n` is the weights of one row (n_embd), `r0..r1` the rows (n_ff of them)
        cpu::tq_gu_rows(best, t, blob.data(), f.gu_row, f.up_off, H, a1.data(), 1, fp.data(), 0, FF);
        cpu::native_gu_rows(f, blob.data(), a1.data(), 1, fp2.data(), 0, FF);
        check(ff == ff2,
              std::string(type_name(t)) + ": native_gu_rows dispatches to the trellis rows");

        // SwiGLU of the two dot rows, in double: gate row r at blob + r*gu_row, up row r at up_off + r*up_row
        Stats st;
        for (int r = 0; r < FF; ++r) {
            const double gs = dot_row(blob.data() + (size_t) r * f.gu_row, t, H, act.v);
            const double us = dot_row(blob.data() + f.up_off + (size_t) r * f.up_row, t, H, act.v);
            st.add((double) ff[(size_t) r], (gs / (1.0 + std::exp(-gs))) * us);
        }
        check(st.rel() < 3e-2, std::string(type_name(t)) + ": gate/up against a double SwiGLU expert, relative L1 " +
                                   std::to_string(st.rel()) + " < 3e-2");

        // down rows through the dispatch, against a double dot on each TQK7 row
        const std::vector<float> h = random_row(FF, 0x21 + t);
        std::vector<uint8_t> hb(f.h_bytes);
        cpu::native_quant_h(f, h.data(), hb.data());
        std::vector<double> hv((size_t) FF);
        for (int i = 0; i < FF / 32; ++i) {
            const block_q8_0* b = (const block_q8_0*) hb.data() + i;
            const double sc = (double) ggml_fp16_to_fp32(b->d);
            for (int j = 0; j < 32; ++j) hv[(size_t) (32 * i + j)] = sc * (double) (int) b->qs[j];
        }
        std::vector<float> out((size_t) H);
        std::vector<const void*> hq{ hb.data() };
        std::vector<float*> op{ out.data() };
        cpu::native_down_rows(f, blob.data(), hq.data(), 1, op.data(), 0, H);
        Stats ds;
        for (int r = 0; r < H; ++r) {
            ds.add((double) out[(size_t) r], dot_row(blob.data() + f.down_off + (size_t) r * f.d_row, 146, FF, hv));
        }
        check(ds.rel() < 1e-2, std::string(type_name(t)) + ": down rows against a double dot, relative L1 " +
                                   std::to_string(ds.rel()) + " < 1e-2");
    }
}

// ---------------------------------------------------------------- (e) geometry and refusals
void geometry() {
    std::string err;
    cpu::NativeFmt f;
    const bool ok = cpu::native_fmt(145, 146, H, FF, f, err);
    check(ok, "TQK6 gate/up with TQK7 down: native_fmt" + (ok ? "" : ": " + err));
    check(ok && f.gu_act == GGML_TYPE_Q8_0 && f.d_act == GGML_TYPE_Q8_0, "the trellis rows take Q8_0 activations");
    check(ok && f.gu_row == (size_t) (H / 128) * sizeof(block_tqk6) && f.d_row == (size_t) (FF / 128) * sizeof(block_tqk7),
          "row sizes from the block structs (520 B gate row, 150 B down row)");
    check(ok && f.bytes == f.gu_row * (size_t) FF + f.up_row * (size_t) FF + f.d_row * (size_t) H && f.bytes == 1049600,
          "one expert blob is 1,049,600 B");
    check(ok && f.tail == 0 && f.up_off == f.gu_row * (size_t) FF && f.down_off == f.up_off + f.up_row * (size_t) FF,
          "no scale tail; gate | up | down back to back");
    check(ok && f.act_bytes == ggml_row_size(GGML_TYPE_Q8_0, H) && f.h_bytes == ggml_row_size(GGML_TYPE_Q8_0, FF),
          "activation sizes are the Q8_0 rows of n_embd and n_ff");

    cpu::NativeFmt g;
    check(!cpu::native_fmt(145, 148, H, FF, g, err, 0.f, 148), "a trellis gate with a different up type is refused");
    check(!cpu::native_fmt(145, 20, H, FF, g, err), "a trellis gate with an IQ4_NL down is refused");
    check(!cpu::native_fmt(22, 145, H, FF, g, err), "an IQ2_S gate with a trellis down is refused");
    check(!cpu::native_fmt(145, 146, 2592, FF, g, err), "an expert width that is not whole 128 blocks is refused");
    check(!cpu::native_fmt(145, 146, 5120, FF, g, err), "an expert wider than the trellis kernels' row limit is refused");
    check(cpu::tq_kbits(145) == 6 && cpu::tq_kbits(147) == 5 && cpu::tq_is_pair(147) && !cpu::tq_is_pair(145) &&
              !cpu::tq_supported(42), "the type table: K bits, pair trellis, and Q2_0 is not trellis");
}

// ---------------------------------------------------------------- frozen hashes
void print_hashes() {
    const cpu::TqIsa best = cpu::tq_isa();
    uint64_t h = 0;
    for (int t = 144; t <= 149; ++t) {
        const Rows r = make_rows(t, H, 0x77 + (uint64_t) t, NT);
        if (r.n == 0) continue;
        std::vector<const void*> act((size_t) NT);
        for (int i = 0; i < NT; ++i) act[(size_t) i] = r.acts[(size_t) i].bytes.data();
        std::vector<float> out((size_t) NT * (size_t) r.n);
        std::vector<float*> op((size_t) NT);
        for (int i = 0; i < NT; ++i) op[(size_t) i] = out.data() + (size_t) i * (size_t) r.n;
        cpu::tq_rows(best, t, r.w.data(), r.row_bytes, r.n, act.data(), NT, op.data(), 0, 1);
        h = h * 1099511620805ull + hash_row(out.data(), r.n);
    }
    std::printf("  trellis row hashes (best ISA %s): %016llx\n", cpu::tq_isa_name(best),
                (unsigned long long) h);
}

void bench() {
    for (int isa = 0; isa <= 2; ++isa) {
        const cpu::TqIsa k = (cpu::TqIsa) isa;
        if (!cpu::tq_isa_ok(k)) continue;
        for (int t = 144; t <= 149; ++t) {
            const Rows r = make_rows(t, H, 0x77 + (uint64_t) t, 1);
            if (r.n == 0) continue;
            std::vector<uint8_t> w((size_t) FF * r.row_bytes);
            fill(w, t, 0x3000 + (uint64_t) t, 0);
            std::vector<const void*> act{ r.acts[0].bytes.data() };
            std::vector<float> out((size_t) FF);
            std::vector<float*> op{ out.data() };
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 200; ++i) cpu::tq_rows(k, t, w.data(), r.row_bytes, FF, act.data(), 1, op.data(), 0, FF);
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
            std::printf("  %-10s %-7s one FF row: %8.1f us  %6.2f GB/s\n", type_name(t), cpu::tq_isa_name(k), us,
                        (double) w.size() / (us * 1e3));
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::printf("tq_cpu_test\n");
    const bool do_bench = argc > 1 && std::string(argv[1]) == "--bench";
    const bool do_print = argc > 1 && std::string(argv[1]) == "--print";
    decode_matches_format();
    geometry();
    rows_isas_and_reference();
    gu_rows();
    if (do_print) print_hashes();
    if (do_bench) bench();
    std::printf(g_fail ? "tq_cpu_test: %d FAILED\n" : "tq_cpu_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
