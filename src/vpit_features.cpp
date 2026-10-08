#include "openv6/vpit_features.hpp"

#include <algorithm>
#include <cmath>

#include "openv6/networks.hpp"
#include "openv6/score_features.hpp"

namespace openv6 {

namespace {

constexpr std::string_view EncoderKey = "6dbe9f75292d82b3cc6c57bd65089b20";
constexpr std::string_view NoteKey = "1dce9c8b2329f1cb9c73fd879bdeb54d";
constexpr std::string_view NoteConfig = "5f6f2b54407c50ebf5343fcddbdb7c10";
constexpr std::string_view NoteHidden = "b8efaa0567206ae04ab50308481882b2";
constexpr std::string_view NoteBottleneck = "c1f0216d696655c00da1ce68cbc92033";
constexpr std::string_view W1 = "220316df91912792cb42bc663a8df636", B1 = "bf1fb08e8ffbab205dbb90120465e856";
constexpr std::string_view W2 = "7352c2e9486a73ce344f167dfad31876", B2 = "c968ce1719cfab911f40b224b5bb28df";
constexpr std::string_view Stack = "9680b2b3ca5c84fcfbf088a3ead0ae22";
constexpr std::string_view Config = "a4ba655a061e6151f80574e6088492a0";
constexpr std::string_view Bins = "903afc65900371c7521ac167421196a6";
constexpr std::string_view Low = "205192c4ec6717232a8e5e2515841a42", High = "f98f2b4f4eab060dad241d3e4fb1c4dc";

float onset(std::span<const V6Phoneme> phonemes, size_t i) {
    const auto& p = phonemes[i];
    if (p.name == "SIL" || p.name == "Sil" || p.name == "BR" || p.name == "Br") return 1;
    int prev_class = i > 0 ? score_features::phoneme_class(phonemes[i - 1].name) : 0;
    int cls = score_features::phoneme_class(p.name);
    bool voiced = cls == 2 || cls == 3 || cls == 7 || cls == 8;
    if (i > 0 && !(phonemes[i - 1].pitch < p.pitch + 50.0f) && voiced && prev_class >= 2 && prev_class <= 8) return 1;
    return p.nucleus && p.name != "sil" ? 1 : 0;
}

std::vector<V6Phoneme> encoder_phonemes(std::span<const V6Phoneme> phonemes) {
    std::vector<V6Phoneme> list;
    for (auto& p : score_features::breath_to_silence(phonemes))
        if (p.name != "-") list.push_back(std::move(p));
    for (size_t i = 0; i + 1 < list.size(); i++) list[i].end = list[i + 1].start;
    list.back().end = phonemes.back().end;
    return list;
}

void gaussian(float pitch, int bins, float lo, float hi, std::span<float> output) {
    for (int i = 0; i < bins; i++) {
        float d = i * (hi - lo) / bins + lo - pitch;
        output[i] = std::exp(-d * d / 20000.0f);
    }
}

}

struct VpitFeatures::Impl {
    json::Value encoder_doc;
    PhonemeTable table;
    GatedStack encoder;
    int note_inputs, note_hidden, note_bottleneck, bins;
    float lo, hi;
    std::vector<float> w1, b1, w2, b2;
    GatedStack note;

    explicit Impl(json::Value model)
        : encoder_doc(model.get_property(EncoderKey)),
          table(score_features::table(encoder_doc)),
          encoder(encoder_doc.get_property(keys::Encoder), 415, 64, 40),
          note_inputs(note_config(model, keys::Inputs)),
          note_hidden(note_config(model, NoteHidden)),
          note_bottleneck(note_config(model, NoteBottleneck)),
          bins(model.get_property(Config).get_property(Bins).get_int32()),
          lo(model.get_property(Config).get_property(Low).get_single()),
          hi(model.get_property(Config).get_property(High).get_single()),
          w1(ModelFile::floats(model.get_property(NoteKey).get_property(W1))),
          b1(ModelFile::floats(model.get_property(NoteKey).get_property(B1))),
          w2(ModelFile::floats(model.get_property(NoteKey).get_property(W2))),
          b2(ModelFile::floats(model.get_property(NoteKey).get_property(B2))),
          note(model.get_property(NoteKey).get_property(Stack), note_bottleneck + bins, 64, note_config(model, keys::Outputs)) {}

    static int note_config(json::Value model, std::string_view key) {
        return model.get_property(NoteKey).get_property(NoteConfig).get_property(key).get_int32();
    }
};

VpitFeatures::VpitFeatures(json::Value model) : impl_(std::make_unique<Impl>(model)) {}

VpitFeatures::~VpitFeatures() = default;

std::vector<float> VpitFeatures::encoder_rows(std::span<const V6Phoneme> phonemes, int& frames) const {
    int width;
    return score_features::frame_rows(impl_->encoder_doc, encoder_phonemes(phonemes), frames, width);
}

std::vector<float> VpitFeatures::rows(std::span<const V6Phoneme> phonemes, int& frames) const {
    const Impl& m = *impl_;
    auto merged = score_features::breath_to_silence(phonemes);
    frames = static_cast<int>(merged.back().end / score_features::Period);
    auto records = score_features::frame_records(m.table, merged, true);
    std::vector<float> pitch(frames, merged.back().pitch), onsets(frames, 0.0f);
    int first = 0;
    for (size_t i = 0; i < merged.size(); i++) {
        int last = std::min(static_cast<int>(std::ceil(merged[i].end / score_features::Period)), frames);
        float value = onset(merged, i);
        for (int f = first; f < last; f++) {
            pitch[f] = merged[i].pitch;
            onsets[f] = value;
        }
        first = std::max(first, last);
    }
    int encoder_frames;
    auto encoder_in = encoder_rows(phonemes, encoder_frames);
    auto encoded = m.encoder.run_aligned(encoder_in, encoder_frames);
    int note_width = m.note_bottleneck + m.bins;
    std::vector<float> note_in(static_cast<size_t>(frames) * note_width);
    std::vector<float> h(m.note_hidden);
    for (int f = 0; f < frames; f++) {
        std::copy(m.b1.begin(), m.b1.end(), h.begin());
        native_math::mat_vec(m.w1, m.note_hidden, m.note_inputs, std::span<const float>(encoded).subspan(static_cast<size_t>(f) * 40, m.note_inputs), h);
        for (float& v : h) v = std::tanh(v);
        std::span<float> g(note_in.data() + static_cast<size_t>(f) * note_width, m.note_bottleneck);
        std::copy(m.b2.begin(), m.b2.end(), g.begin());
        native_math::mat_vec(m.w2, m.note_bottleneck, m.note_hidden, h, g);
        for (float& v : g) v = std::tanh(v);
        std::span<float> gauss(note_in.data() + static_cast<size_t>(f) * note_width + m.note_bottleneck, m.bins);
        gaussian(pitch[f], m.bins, m.lo, m.hi, gauss);
        for (float& v : gauss) v *= onsets[f];
    }
    auto note_out = m.note.run_aligned(note_in, frames);
    constexpr int outputs = Width - 29;
    std::vector<float> result(static_cast<size_t>(frames) * Width);
    constexpr int ctx0[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9};
    constexpr int record = score_features::RecordWidth;
    for (int f = 0; f < frames; f++) {
        size_t o = static_cast<size_t>(f) * Width, r = static_cast<size_t>(f) * 3 * record;
        result[o] = pitch[f];
        result[o + 1] = onsets[f];
        for (int k = 0; k < 11; k++) result[o + 2 + k] = records[r + ctx0[k]];
        for (int k = 0; k < 8; k++) result[o + 13 + k] = records[r + record + k];
        for (int k = 0; k < 8; k++) result[o + 21 + k] = records[r + 2 * record + k];
        std::copy_n(note_out.begin() + static_cast<size_t>(f) * outputs, outputs, result.begin() + o + 29);
    }
    return result;
}

}
