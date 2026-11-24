// tests/kernels/cpu/hadamard_test.cpp - the activation-side Hadamard rotation (a folded expert pack's input transform).
//
//     build/hadamard_test
//
// A pack whose experts are stored in a block-Hadamard-rotated basis (a GGUF's `prism.hadamard.*`, the model file's
// `experts.hadamard` section) must rotate every activation that enters a folded weight before quantizing it, on every
// path, or the expert computes a plausible-looking wrong value.  Checks: (a) `hadamard_parse` reads the section and
// refuses every malformed one; (b) `hadamard_rotate` is the documented transform - signs first, then the normalized
// Sylvester-Walsh-Hadamard butterfly per block - against a double reference, and rotating twice returns the input
// (H is orthonormal); (c) the process-wide spec and its sign vectors; (d) `native_quant_act` / `native_quant_h` apply
// the rotation before quantizing, so a folded layer's activation bytes are the quantization of the rotated row.
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/hadamard.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {
constexpr int H = 2560, FF = 640;

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-84s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

double rel_err(const std::vector<float>& got, const std::vector<double>& want) {
    double n = 0, d = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        n += std::fabs((double) got[i] - want[i]);
        d += std::fabs(want[i]);
    }
    return n / (d + 1e-30);
}

// y[c] = sum_r (x[r] * s[r]) * (-1)^popcount(r & c) / sqrt(block), per block, in double
int popcount(int v) {
    int n = 0;
    for (; v; v &= v - 1) ++n;
    return n;
}

std::vector<double> rotate_ref(const std::vector<float>& x, int block, const std::vector<float>* signs) {
    std::vector<double> y(x.size());
    const double scale = 1.0 / std::sqrt((double) block);
    for (size_t b0 = 0; b0 < x.size(); b0 += (size_t) block) {
        for (int c = 0; c < block; ++c) {
            double s = 0;
            for (int r = 0; r < block; ++r) {
                const double v = (double) x[b0 + (size_t) r] * (signs ? (double) (*signs)[b0 + (size_t) r] : 1.0);
                s += v * ((popcount(r & c) & 1) ? -1.0 : 1.0);
            }
            y[b0 + (size_t) c] = s * scale;
        }
    }
    return y;
}

std::vector<float> random_row(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> x((size_t) n);
    for (float& v : x) v = u(rng);
    return x;
}

std::vector<float> signs_row(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> s((size_t) n);
    for (float& v : s) v = (rng() & 1) ? 1.f : -1.f;
    return s;
}

// ---------------------------------------------------------------- (a) the section
std::string section(int block, const std::vector<int>& widths, int n_layers, bool explicit_signs = true) {
    std::string t = "# basalt hadamard v1: prism.hadamard (sign_mode explicit)\nblock " + std::to_string(block) + "\n";
    if (explicit_signs)
        for (int w : widths) {
            std::string v;
            for (int i = 0; i < w; ++i) v += (i ? " " : "") + std::to_string((i % 3) ? 1 : -1);
            t += "signs " + std::to_string(w) + " " + v + "\n";
        }
    for (int l = 0; l < n_layers; ++l) t += "layer " + std::to_string(l) + " 1 1 1\n";
    return t;
}

void parsing() {
    std::string err;
    cpu::HadamardSpec h;
    const bool full = cpu::hadamard_parse(section(128, { H, FF }, 4), 4, h, err);
    check(full && h.block == 128 && h.any() &&
              h.roles.size() == 4 && h.gu(0) && h.down(0) && h.signs.size() == 2 && h.signs.at(H).size() == H,
          "a folded pack's section parses (block, two sign widths, every layer folded)" + (err.empty() ? "" : ": " + err));
    check(cpu::hadamard_signs(H) == nullptr, "the process-wide spec is separate from a parsed one");

    // identity signs: no signs lines, still a fold
    cpu::HadamardSpec id;
    const bool identity = cpu::hadamard_parse(section(32, {}, 3, false), 3, id, err);
    check(identity && id.block == 32 && id.any() && id.signs.empty() && cpu::hadamard_signs(H) == nullptr,
          "identity signs: a block and roles, no signs lines" + (err.empty() ? "" : ": " + err));

    // only some layers folded
    cpu::HadamardSpec part;
    const std::string part_text = "# basalt hadamard v1: x\nblock 128\nlayer 2 1 1 1\nlayer 5 0 0 1\n";
    check(cpu::hadamard_parse(part_text, 8, part, err) && !part.gu(0) && part.gu(2) && part.down(2) &&
              !part.gu(5) && part.down(5) && !part.gu(7),
          "a partly folded model: only the named layers" + (err.empty() ? "" : ": " + err));

    const std::vector<std::string> bad = {
        "# basalt hadamard v2\nblock 128\nlayer 0 1 1 1\n",          // unknown version
        "block 128\nlayer 0 1 1 1\n",                               // no version header
        "# basalt hadamard v1\nlayer 0 1 1 1\n",                    // no block line
        "# basalt hadamard v1\nblock 96\nlayer 0 1 1 1\n",          // block not a power of two
        "# basalt hadamard v1\nblock 8192\nlayer 0 1 1 1\n",        // block too wide
        "# basalt hadamard v1\nblock 128\nblock 64\nlayer 0 1 1 1\n",   // repeated block
        "# basalt hadamard v1\nblock 128\nsigns 100 1\n",           // sign width not whole blocks
        "# basalt hadamard v1\nblock 128\nsigns 128 1 2\nlayer 0 1 1 1\n",   // sign values other than +-1
        "# basalt hadamard v1\nblock 128\nsigns 128 " + std::string(129, '1') + "\nlayer 0 1 1 1\n",  // too long
        "# basalt hadamard v1\nblock 128\nlayer 0 1 0 1\n",         // gate folded but not up
        "# basalt hadamard v1\nblock 128\nlayer 9 1 1 1\n",         // layer beyond the model
        "# basalt hadamard v1\nblock 128\nlayer 0 1 1 1\nlayer 0 0 0 1\n",   // repeated layer
        "# basalt hadamard v1\nblock 128\nroles 0 7\n",             // an unknown line
    };
    for (const std::string& t : bad) {
        cpu::HadamardSpec s;
        err.clear();
        const bool ok = cpu::hadamard_parse(t, 8, s, err);
        check(!ok, "refused: " + t.substr(0, t.find('\n')) + " (" + err + ")");
    }
}

// ---------------------------------------------------------------- (b) the transform
void rotate() {
    for (int block : { 32, 64, 128 }) {
        const std::vector<float> x = random_row(H, 0x1000 + (uint64_t) block);
        std::vector<float> y = x;
        cpu::hadamard_rotate(y.data(), (int) y.size(), block, nullptr);
        check(rel_err(y, rotate_ref(x, block, nullptr)) < 1e-5,
              "block " + std::to_string(block) + ", identity signs: against a double Hadamard reference");

        const std::vector<float> s = signs_row(H, 0x2000 + (uint64_t) block);
        std::vector<float> z = x;
        cpu::hadamard_rotate(z.data(), (int) z.size(), block, s.data());
        check(rel_err(z, rotate_ref(x, block, &s)) < 1e-5,
              "block " + std::to_string(block) + ", explicit signs: signs applied before the butterfly");

        // H is orthonormal and symmetric, so the rotation's inverse is the butterfly again and then the signs:
        // y = H (s x)  ->  H y = s x  ->  s (H y) = x
        std::vector<float> twice = z;
        cpu::hadamard_rotate(twice.data(), (int) twice.size(), block, nullptr);
        for (size_t i = 0; i < twice.size(); ++i) twice[i] *= s[i];
        double n = 0, d = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            n += std::fabs((double) twice[i] - (double) x[i]);
            d += std::fabs((double) x[i]);
        }
        check(n / d < 1e-4, "block " + std::to_string(block) + ": the butterfly and the signs undo the rotation");
    }
    // one row of the down width, the size a folded layer rotates per entry
    const std::vector<float> h = random_row(FF, 7);
    std::vector<float> y = h;
    cpu::hadamard_rotate(y.data(), FF, 128, nullptr);
    check(rel_err(y, rotate_ref(h, 128, nullptr)) < 1e-5, "one n_ff row of block 128 against the reference");
}

// ---------------------------------------------------------------- (c) the process-wide spec
void spec() {
    std::string err;
    cpu::HadamardSpec h;
    if (!cpu::hadamard_parse(section(128, { H, FF }, 4), 4, h, err)) {
        check(false, "parse for the spec test: " + err);
        return;
    }
    cpu::hadamard_set(std::move(h));
    const cpu::HadamardSpec& s = cpu::hadamard_spec();
    check(s.block == 128 && s.any() && s.gu(3) && s.down(3), "hadamard_set makes the spec process-wide");
    const float* p = cpu::hadamard_signs(FF);
    check(p != nullptr && p[0] == -1.f && p[1] == 1.f, "hadamard_signs returns the width's vector");
    check(cpu::hadamard_signs(1024) == nullptr, "a width with no vector is identity signs");
    cpu::hadamard_set(cpu::HadamardSpec{});
    check(!cpu::hadamard_spec().any() && cpu::hadamard_signs(H) == nullptr, "an empty spec means no folded experts");
}

// ---------------------------------------------------------------- (d) quantization applies the rotation
void quantize_applies_rotation() {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(145, 146, H, FF, f, err)) {
        check(false, "trellis native_fmt: " + err);
        return;
    }
    const std::vector<float> x = random_row(H, 0x3333);
    const std::vector<float> sx = signs_row(H, 0x4444);
    const std::vector<float> hs = signs_row(FF, 0x5555);
    const std::vector<float> h = random_row(FF, 0x6666);

    cpu::NativeFmt plain = f;
    std::vector<uint8_t> a_plain(f.act_bytes), h_plain(f.h_bytes);
    cpu::native_quant_act(plain, x.data(), a_plain.data());
    cpu::native_quant_h(plain, h.data(), h_plain.data());

    // the same activations, rotated by hand and quantized without the fold, must be the same bytes
    std::vector<float> rx = x;
    cpu::hadamard_rotate(rx.data(), H, 128, sx.data());
    std::vector<float> rh = h;
    cpu::hadamard_rotate(rh.data(), FF, 128, hs.data());
    std::vector<uint8_t> a_rot(f.act_bytes), h_rot(f.h_bytes);
    cpu::native_quant_act(plain, rx.data(), a_rot.data());
    cpu::native_quant_h(plain, rh.data(), h_rot.data());

    cpu::NativeFmt folded = f;
    folded.had_block = 128;
    folded.had_gu = true;
    folded.had_d = true;
    folded.gu_signs = sx.data();
    folded.d_signs = hs.data();
    std::vector<uint8_t> a_fold(f.act_bytes), h_fold(f.h_bytes);
    cpu::native_quant_act(folded, x.data(), a_fold.data());
    cpu::native_quant_h(folded, h.data(), h_fold.data());
    check(a_fold == a_rot, "native_quant_act rotates x before quantizing (gate/up input)");
    check(h_fold == h_rot, "native_quant_h rotates h before quantizing (down input)");
    check(a_fold != a_plain && h_fold != h_plain, "a folded layer is not the unrotated quantization");
}

// ---------------------------------------------------------------- (e) the v8 layout and its fold
void layout_v8() {
    namespace c = basalt::kernels::cpu;
    std::string err;
    c::NativeFmt f;
    if (!c::native_fmt(145, 146, H, FF, f, err)) { check(false, "trellis fmt for the layout: " + err); return; }
    const std::string head = "# experts v8: layer gu_type d_type offset blob_bytes up_type (n_expert 2, total " +
                             std::to_string(2 * 2 * f.bytes) + ")\n";
    const std::string lines = "0 145 146 0 " + std::to_string(f.bytes) + " 145\n" +
                              "1 145 146 " + std::to_string(f.bytes * 2) + " " + std::to_string(f.bytes) + " 145\n";
    const std::string had = section(128, { H, FF }, 2);
    err.clear();
    const bool folded = c::expert_layout_load_text(head + lines, 2, 2, err, H, FF, 0.f, had);
    check(folded, "a v8 trellis layout with experts.hadamard loads" + (err.empty() ? "" : ": " + err));
    const c::ExpertLayout& L = c::expert_layout();
    check(L.version == 8 && L.fmt[0].had_block == 128 && L.fmt[0].had_gu && L.fmt[0].had_d &&
              L.fmt[0].gu_signs != nullptr && L.fmt[0].d_signs != nullptr && L.fmt[0].gu_signs[0] == -1.f &&
              L.fmt[1].gu_signs == L.fmt[0].gu_signs,
          "the fold reaches every layer: block, roles and sign vectors");
    err.clear();
    check(!c::expert_layout_load_text(head + lines, 2, 2, err, H, FF),
          "a v8 layout without experts.hadamard is refused" + (err.empty() ? "" : ": " + err));
    err.clear();
    const std::string v7 = "# experts v7: layer gu_type d_type offset blob_bytes up_type (n_expert 2, total " +
                           std::to_string(2 * 2 * f.bytes) + ")\n" + lines;
    const bool v7ok = c::expert_layout_load_text(v7, 2, 2, err, H, FF, 0.f, had);
    check(!v7ok, "a fold on a v7 layout is refused" + (err.empty() ? "" : ": " + err));
    // a folded non-trellis layer: the IQ kernels and the MMQ path quantize unrotated inputs
    c::NativeFmt iq;
    if (!c::native_fmt(22, 42, H, FF, iq, err)) { check(false, "IQ2_S fmt: " + err); return; }
    const std::string iq_head = "# experts v8: layer gu_type d_type offset blob_bytes up_type (n_expert 2, total " +
                                std::to_string(2 * 2 * iq.bytes) + ")\n";
    const std::string iq_lines = "0 22 42 0 " + std::to_string(iq.bytes) + " 22\n" +
                                 "1 22 42 " + std::to_string(iq.bytes * 2) + " " + std::to_string(iq.bytes) + " 22\n";
    err.clear();
    const bool iqok = c::expert_layout_load_text(iq_head + iq_lines, 2, 2, err, H, FF, 0.f, had);
    check(!iqok, "a fold on a non-trellis expert is refused" + (err.empty() ? "" : ": " + err));
    // a block that does not divide the expert widths
    err.clear();
    const bool bigblock = c::expert_layout_load_text(head + lines, 2, 2, err, H, FF, 0.f, section(256, { H, FF }, 2));
    check(!bigblock, "a rotation block that does not divide the widths is refused" + (err.empty() ? "" : ": " + err));
    // a folded layer whose width has no sign vector
    err.clear();
    const bool nosigns = c::expert_layout_load_text(head + lines, 2, 2, err, H, FF, 0.f, section(128, { H }, 2));
    check(!nosigns, "a folded width with no sign vector is refused" + (err.empty() ? "" : ": " + err));
}
}  // namespace

int main() {
    std::printf("hadamard_test\n");
    parsing();
    rotate();
    spec();
    quantize_applies_rotation();
    layout_v8();
    std::printf(g_fail ? "hadamard_test: %d FAILED\n" : "hadamard_test: all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
