#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#include "check.hpp"
#include "openv6/base64.hpp"
#include "openv6/fft.hpp"
#include "openv6/json.hpp"
#include "openv6/md5.hpp"
#include "openv6/model_file.hpp"
#include "openv6/phoneme_conversion.hpp"
#include "openv6/speex_resampler.hpp"
#include "openv6/vtmg_model.hpp"

using namespace openv6;

namespace {

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::unique_ptr<json::Document> parse(const std::string& text) { return json::Document::parse(text); }

void test_json() {
    auto doc = parse(R"({"a":1,"b":[true,false,null,"x\n\"\\\/é😀"],"c":{"d":-2.5e3},"a":7,"e":""})");
    auto root = doc->root();
    CHECK(root.kind() == json::Kind::Object);
    CHECK(root.object_length() == 5);
    CHECK(root.get_property("a").get_int32() == 7);
    auto b = root.get_property("b");
    CHECK(b.array_length() == 4);
    CHECK(b[0].get_boolean() && !b[1].get_boolean() && b[2].kind() == json::Kind::Null);
    CHECK(b[3].get_string() == "x\n\"\\/\xc3\xa9\xf0\x9f\x98\x80");
    CHECK(root.get_property("c").get_property("d").get_double() == -2500.0);
    CHECK(root.get_property("c").get_property("d").get_single() == -2500.0f);
    CHECK(root.get_property("e").get_string().empty());
    json::Value missing;
    CHECK(!root.try_get_property("zzz", missing));
    CHECK(throws([&] { root.get_property("zzz"); }));
    CHECK(throws([&] { b.get_property("a"); }));
    CHECK(throws([&] { b[4]; }));

    size_t count = 0;
    std::string names;
    for (auto m : root.enumerate_object()) {
        names += std::string(m.name);
        count++;
    }
    CHECK(count == 5 && names == "abcae");
    int sum = 0;
    auto nums = parse(" [ 1 , 2,3 ,\n4 ] ");
    for (auto v : nums->root().enumerate_array()) sum += v.get_int32();
    CHECK(sum == 10);
    CHECK(parse("[]")->root().array_length() == 0);
    CHECK(parse("{}")->root().object_length() == 0);
    CHECK(parse("3.14159")->root().get_single() == 3.14159f);
    CHECK(parse("9007199254740993")->root().get_int64() == 9007199254740993LL);
    CHECK(parse("true")->root().kind() == json::Kind::True);
    CHECK(throws([] { parse("{\"a\":}"); }));
    CHECK(throws([] { parse("[1,2"); }));
    CHECK(throws([] { parse("\"abc"); }));
    CHECK(throws([] { parse("[1] x"); }));
    CHECK(throws([] { parse("{\"a\":1,}"); }));
    CHECK(throws([] { parse(""); }));
    CHECK(throws([] { parse("[1]")->root().get_property("a"); }));
    CHECK(throws([] { parse("\"s\"")->root().get_int32(); }));
    CHECK(throws([] { parse("1.5")->root().get_int32(); }));
}

void test_base64_md5_half() {
    CHECK(base64_decode("").empty());
    auto v = base64_decode("TWFu");
    CHECK(std::string(v.begin(), v.end()) == "Man");
    v = base64_decode("TWE=");
    CHECK(std::string(v.begin(), v.end()) == "Ma");
    v = base64_decode("TQ==");
    CHECK(std::string(v.begin(), v.end()) == "M");
    CHECK(throws([] { base64_decode("TWF"); }));
    CHECK(throws([] { base64_decode("TW!u"); }));

    auto hex = [](std::string_view s) { return md5_hex({reinterpret_cast<const uint8_t*>(s.data()), s.size()}); };
    CHECK(hex("") == "d41d8cd98f00b204e9800998ecf8427e");
    CHECK(hex("abc") == "900150983cd24fb0d6963f7d28e17f72");
    CHECK(hex("12345678901234567890123456789012345678901234567890123456789012345678901234567890") == "57edf4a22be3c955ac49da2e2107b67a");

    CHECK(half_to_float(0x3c00) == 1.0f);
    CHECK(half_to_float(0xc000) == -2.0f);
    CHECK(half_to_float(0x0001) == std::ldexp(1.0f, -24));
    CHECK(half_to_float(0x0200) == std::ldexp(1.0f, -15));
    CHECK(half_to_float(0x7bff) == 65504.0f);
    CHECK(std::isinf(half_to_float(0x7c00)) && std::isnan(half_to_float(0x7e00)));
    CHECK(half_to_float(0x8000) == 0.0f && std::signbit(half_to_float(0x8000)));
    CHECK(half_to_float(0x3555) == std::ldexp(1365.0f, -12));

    auto doc = parse(R"(["*F32AACAPwAAAEA=","*F16ADwAwA==","*S32AQAAAP///38=","*XXXAAAA"])");
    auto root = doc->root();
    auto f32 = ModelFile::floats(root[0]);
    CHECK(f32.size() == 2 && f32[0] == 1.0f && f32[1] == 2.0f);
    auto f16 = ModelFile::floats(root[1]);
    CHECK(f16.size() == 2 && f16[0] == 1.0f && f16[1] == -2.0f);
    auto ints = ModelFile::ints(root[2]);
    CHECK(ints.size() == 2 && ints[0] == 1 && ints[1] == 0x7fffffff);
    CHECK(throws([&] { ModelFile::floats(root[3]); }));
    CHECK(throws([&] { ModelFile::ints(root[0]); }));
}

void test_fft() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1, 1);
    for (int n : {4, 8, 512, 1024}) {
        RealFourierTransform fft(n);
        std::vector<double> x(n), back(n);
        for (auto& v : x) v = dist(rng);
        std::vector<std::complex<double>> spec(n / 2 + 1);
        fft.forward(x, spec);
        double worst = 0;
        for (int k = 0; k <= n / 2; k++) {
            std::complex<double> sum = 0;
            for (int i = 0; i < n; i++) sum += x[i] * std::polar(1.0, -2.0 * M_PI * ((long long)k * i % n) / n);
            worst = std::max(worst, std::abs(sum - spec[k]));
        }
        CHECK_MSG(worst < 1e-9, "forward n=" + std::to_string(n) + " error " + std::to_string(worst));
        fft.inverse(spec, back);
        double round_trip = 0;
        for (int i = 0; i < n; i++) round_trip = std::max(round_trip, std::abs(back[i] - x[i]));
        CHECK_MSG(round_trip < 1e-12, "inverse n=" + std::to_string(n) + " error " + std::to_string(round_trip));
    }
}

void test_resampler() {
    const int in_rate = 32000, out_rate = 44100;
    const int n = 32000;
    const double freq = 1000;
    std::vector<float> in(n);
    for (int i = 0; i < n; i++) in[i] = static_cast<float>(0.5 * std::sin(2 * M_PI * freq * i / in_rate));
    int out_length = static_cast<int>(std::ceil(n * (double)out_rate / in_rate));
    auto out = speex_resampler::resample(in, in_rate, out_rate, out_length);
    CHECK(static_cast<int>(out.size()) == out_length);
    double energy = 0;
    int crossings = 0, count = 0;
    for (int i = 2000; i < out_length - 2000; i++, count++) {
        energy += out[i] * out[i];
        if (out[i - 1] < 0 && out[i] >= 0) crossings++;
    }
    double rms = std::sqrt(energy / count);
    CHECK_MSG(std::abs(rms - 0.5 / std::sqrt(2.0)) < 0.01, "rms " + std::to_string(rms));
    double measured = crossings * (double)out_rate / count;
    CHECK_MSG(std::abs(measured - freq) < 5, "frequency " + std::to_string(measured));

    std::vector<float> dc(4000, 1.0f);
    auto flat = speex_resampler::resample(dc, in_rate, out_rate, 5000);
    CHECK_MSG(std::abs(flat[2500] - 1.0f) < 1e-3, "dc " + std::to_string(flat[2500]));
}

void test_vtmg() {
    std::string dir = std::string(OPENV6_TEST_DATA) + "/vtmg/";
    auto model = VtmgModel::load(dir + "timing-synthetic-scratch.vtmg");
    auto read_floats = [](const std::string& path) {
        auto raw = read_file(path);
        std::vector<float> v(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), v.size() * 4);
        return v;
    };
    auto features = read_floats(dir + "vtmg-input.bin");
    auto native = read_floats(dir + "vtmg-scratch-output.bin");
    int frames = static_cast<int>(features.size() / 82);
    auto predicted = model.forward(features, frames, 0);
    CHECK(predicted.size() == static_cast<size_t>(frames - 6) * 3);
    float max_error = 0;
    for (int f = 0; f < frames - 6; f++)
        for (int c = 0; c < 3; c++) max_error = std::max(max_error, std::abs(predicted[f * 3 + c] - native[(f + 3) * 3 + c]));
    CHECK_MSG(max_error < 2e-6f, "VTMG forward differs from native by " + std::to_string(max_error));
    CHECK(throws([&] { model.forward(features, frames, model.conditions()); }));
    CHECK(throws([&] { model.forward(std::span<const float>(features).first(6 * 82), 6, 0); }));
}

void test_phoneme_conversion() {
    using namespace phoneme_conversion;
    auto join = [](const std::vector<ModelPhoneme>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); i++) s += (i ? " " : "") + v[i].name;
        return s;
    };
    struct Case { int language; const char* entered; const char* expected; };
    const Case cases[] = {
        {0, "C i", "h i"}, {0, "4' i", "r i"}, {0, "4 M", "r u"}, {0, "C a", "hy a"},
        {1, "s t r O: N", "s t r ao ng"}, {1, "w @r l d", "w er l d"}, {1, "D @", "dh ah"}, {1, "f I@", "f ih r"},
        {4, "n i", "n i|CH"}, {4, "k uo", "k uo+0 uo+1"}, {4, "ts\\_h iE_n", "ts\\_h iE_n+0 iE_n+1"}, {1, "k XX", "k sil"},
    };
    for (const auto& c : cases) {
        auto got = join(to_model(c.entered, c.language));
        CHECK_MSG(got == c.expected, std::string(c.entered) + " -> " + got);
    }

    auto sources = to_model("k uo", 4);
    CHECK(sources.size() == 3 && sources[0].source == 0 && sources[1].source == 1 && sources[2].source == 1);
    CHECK(throws([] { to_model("a", 2); }));
}

int run_reference() {
    const char* root = reference_dir();
    if (!root) {
        std::puts("OPENV6_REFERENCE unset; skipping model decode");
        return 77;
    }
    std::string dir = std::string(root) + "/reference/";
    namespace chr = std::chrono;
    auto time = [](auto&& f) {
        auto t0 = chr::steady_clock::now();
        f();
        return chr::duration<double>(chr::steady_clock::now() - t0).count();
    };

    std::unique_ptr<ModelFile> vtb2, vpit;
    double load_vtb2 = time([&] { vtb2 = std::make_unique<ModelFile>(ModelFile::load(dir + "asa.vtb2")); });
    double load_vpit = time([&] { vpit = std::make_unique<ModelFile>(ModelFile::load(dir + "asa.vpit")); });
    std::printf("asa.vtb2 loaded in %.3f s, asa.vpit in %.3f s\n", load_vtb2, load_vpit);
    CHECK(load_vtb2 < 1.0 && load_vpit < 1.0);

    auto timbre = vtb2->root();
    CHECK(timbre.kind() == json::Kind::Object && timbre.object_length() > 0);
    auto encoder = timbre.get_property("c688ac0ef614fc2004874245946c9580");
    auto params = encoder.get_property("3c77d931295c40d4d75d971398fee8a1");
    auto dilations = ModelFile::ints(params.get_property("b028f94998d330bc643746f7a453feed"));
    CHECK(!dilations.empty());
    auto layers = encoder.get_property("60934ffa2a466aa4dea617defa78f36b");
    CHECK(layers.array_length() == dilations.size());
    auto weight = ModelFile::floats(layers[0].get_property("3d8a9ac90f7ac150415ce60a7b3aea38").get_property("30c16498ff30815c43d58f610a91f938"));
    CHECK(weight.size() == 64 * 2 * 64 * 3);
    bool finite = true;
    for (float w : weight) finite = finite && std::isfinite(w) && std::abs(w) < 100;
    CHECK(finite);
    auto table = ModelFile::phoneme_table(timbre.get_property("edc2ce94290a3b18bc7f745276643253"));
    CHECK(!table.empty());
    std::printf("encoder layers %zu, phoneme table %zu\n", layers.array_length(), table.size());

    auto pitch = vpit->root();
    CHECK(pitch.get_property("a4ba655a061e6151f80574e6088492a0").kind() == json::Kind::Object);

    auto timing = VtmgModel::load(dir + "VOCALOID6.vtmg");
    CHECK(timing.conditions() == 26);
    CHECK(!timing.consonant_weights().empty());
    std::printf("VOCALOID6.vtmg: %d conditions, %zu consonant weights\n", timing.conditions(), timing.consonant_weights().size());
    return 0;
}

}

int main() {
    test_json();
    test_base64_md5_half();
    test_fft();
    test_resampler();
    test_vtmg();
    test_phoneme_conversion();
    int reference = run_reference();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    return reference;
}
