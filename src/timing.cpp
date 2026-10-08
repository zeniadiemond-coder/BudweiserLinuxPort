#include "openv6/timing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "openv6/score_features.hpp"

namespace openv6::timing {

namespace {

constexpr int None = 14;

struct Record {
    double start, end;
    int midi;
    bool rest;
    const std::vector<std::string>* phonemes;
    int meter;
    float position;
    bool flag;
};

const std::vector<std::string> PadPhonemes = {"sil"};
const Record Pad = {0, 2, -1, true, &PadPhonemes, 0, 0, false};

int timing_class(const std::string& name) {
    if (name == "junk") return 13;
    int c = score_features::phoneme_class(name);
    return c == 13 ? None : c;
}

int nucleus_index(const std::vector<std::string>& phonemes) {
    for (int i = static_cast<int>(phonemes.size()) - 1; i >= 0; i--) {
        if (score_features::nuclei().contains(phonemes[i])) return i;
    }
    return 0;
}

Record to_record(const V6Note& note) {
    return {note.start, note.end, note.rest() ? -1 : static_cast<int>(round_even(*note.pitch * 0.01f)) + 69,
            note.rest() || (note.phonemes.size() == 1 && note.phonemes[0] == "sil"), &note.phonemes, note.meter, note.position, note.flag};
}

void pair(float x, float lo, float hi, float* output) {
    float t = std::clamp((x - lo) / (hi - lo), 0.f, 1.f);
    output[0] = 1.f - t;
    output[1] = t;
}

void row(const Record& cur, const Record& prev, const Record& next, float* output) {
    const auto& ph = *cur.phonemes;
    int n = static_cast<int>(ph.size()), onset = nucleus_index(ph), coda = n - onset - 1;
    output[onset] = 1;
    output[n - onset + 2] = 1;
    output[6 + (onset > 0 ? timing_class(ph[0]) : None)] = 1;
    output[21 + (onset > 0 ? timing_class(ph[onset - 1]) : None)] = 1;
    output[36 + (coda > 0 ? timing_class(ph[n - coda]) : None)] = 1;
    output[51 + (coda > 0 ? timing_class(ph[n - 1]) : None)] = 1;
    output[66] = cur.flag ? 1 : 0;
    output[67] = prev.flag ? 1 : 0;
    output[68] = next.flag ? 1 : 0;
    output[69] = cur.rest ? 1 : 0;
    output[70] = prev.rest ? 1 : 0;
    output[71] = next.rest ? 1 : 0;
    pair(static_cast<float>(cur.rest || prev.rest ? 0 : cur.midi - prev.midi), -12.f, 12.f, output + 72);
    pair(static_cast<float>(cur.rest || next.rest ? 0 : next.midi - cur.midi), -12.f, 12.f, output + 74);
    pair(std::log(static_cast<float>(cur.end - cur.start)), std::log(0.05f), std::log(2.f), output + 76);
    float beat = cur.meter != 0 ? 1.f / static_cast<float>(cur.meter) : std::numeric_limits<float>::infinity();
    double pos = cur.position;
    output[78] = std::abs(pos) < 0.0010000000474974513 ? 1 : 0;
    output[79] = std::abs(std::fmod(pos, static_cast<double>(beat))) < 0.0010000000474974513 ? 1 : 0;
    output[80] = std::abs(std::fmod(pos, static_cast<double>(beat * 0.5f))) < 0.0010000000474974513 ? 1 : 0;
    output[81] = std::abs(std::fmod(pos, static_cast<double>(beat * 0.25f))) < 0.0010000000474974513 ? 1 : 0;
}

struct Entry {
    double start;
    std::vector<std::string> names;
    int cls;
    bool note_start;
    bool nucleus;
    float pitch;
    int note;
};

struct Split {
    double start;
    std::string name;
    int cls;
    bool note_start;
    bool nucleus;
    float pitch;
    int note;
};

double t0(float x) { return (x + 1.0) * 0.5 * 0.32 - 0.16; }
double t1(float x) { return (x + 1.0) * 0.5 * 0.31 + 0.01; }
float note_pitch(const V6Note& n) { return n.rest() ? -10000.f : round_even(*n.pitch * 0.01f) * 100.f; }

}

std::vector<float> rows(std::span<const V6Note> notes, int& rows) {
    std::vector<Record> recs(3, Pad);
    for (const auto& note : notes) recs.push_back(to_record(note));
    recs.insert(recs.end(), 3, Pad);
    rows = static_cast<int>(recs.size());
    std::vector<float> output(static_cast<size_t>(rows) * Width);
    for (int i = 0; i < rows; i++) {
        row(recs[i], i > 0 ? recs[i - 1] : Pad, i + 1 < rows ? recs[i + 1] : Pad, output.data() + static_cast<size_t>(i) * Width);
    }
    return output;
}

std::vector<V6Phoneme> phonemes(std::span<const V6Note> notes, std::span<const float> outputs, const StringMap<float>& weights) {
    std::vector<Entry> entries;
    entries.push_back({0, {"sil"}, -1, true, true, -100000.f, 0});
    float last_pitch = -100000.f;
    for (size_t i = 1; i < notes.size(); i++) {
        const auto& prev = notes[i - 1];
        const auto& cur = notes[i];
        const auto& ph = cur.phonemes;
        const auto& pp = prev.phonemes;
        int onset = nucleus_index(ph), pk = nucleus_index(pp), coda = static_cast<int>(pp.size()) - pk - 1;
        double v = cur.start + t0(outputs[i * 3]);
        double p = prev.start + t0(outputs[(i - 1) * 3]);
        double onset_duration = onset > 0 ? t1(outputs[i * 3 + 1]) : 0;
        double a = std::min((p + v) * 0.5, p + 0.06);
        double b = std::min((a + p) * 0.5, p + 0.05);
        double o = std::max(a, v - onset_duration);
        double c = std::max(b, o - t1(outputs[(i - 1) * 3 + 2]));
        float pitch = note_pitch(cur);
        int index = static_cast<int>(i);
        if (coda > 0) entries.push_back({c, {pp.begin() + pk + 1, pp.end()}, prev.language, false, false, last_pitch, index - 1});
        if (onset > 0) entries.push_back({o, {ph.begin(), ph.begin() + onset}, cur.language, true, false, pitch, index});
        if (cur.flag) entries.push_back({v, {"-"}, cur.language, true, true, pitch, index});
        else entries.push_back({v, {ph[onset]}, cur.language, onset == 0, true, pitch, index});
        last_pitch = pitch;
    }
    double end = notes.back().end;
    std::vector<Split> split;
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];
        int n = static_cast<int>(e.names.size());
        if (n == 1) {
            split.push_back({e.start, e.names[0], e.cls, e.note_start, e.nucleus, e.pitch, e.note});
            continue;
        }
        double span = (i + 1 < entries.size() ? entries[i + 1].start : end) - e.start;
        std::vector<double> w;
        for (const auto& name : e.names) {
            auto it = weights.find(name);
            w.push_back(it != weights.end() ? static_cast<double>(it->second) : span / n);
        }
        double total = 0;
        for (double x : w) total += x;
        double t = e.start;
        for (int k = 0; k < n; k++) {
            split.push_back({t, e.names[k], e.cls, k == 0 && e.note_start, k == n - 1 && e.nucleus, e.pitch, e.note});
            t += span / total * w[k];
        }
    }
    split[0].pitch = split[1].pitch;
    for (size_t i = 1; i < split.size(); i++) {
        if (split[i].pitch < -9999.f) split[i].pitch = split[i - 1].pitch;
    }
    split.back().cls = -1;
    std::vector<V6Phoneme> result;
    for (size_t i = 0; i < split.size(); i++) {
        const auto& s = split[i];
        V6Phoneme phoneme{s.start, i + 1 < split.size() ? split[i + 1].start : end, s.name, s.pitch, s.note_start, s.nucleus, s.cls, s.note};
        if (s.name == "sil" && i < split.size() - 1) {
            if (phoneme.duration() >= 1.0) {
                V6Phoneme sil = phoneme, br = phoneme;
                sil.end = phoneme.end - 0.6;
                br.start = phoneme.end - 0.6;
                br.name = "br";
                result.push_back(std::move(sil));
                result.push_back(std::move(br));
                continue;
            }
            phoneme.name = "br";
        }
        result.push_back(std::move(phoneme));
    }
    return result;
}

std::vector<V6Phoneme> run(const VtmgModel& model, std::span<const V6Note> notes, int condition) {
    int count;
    auto input = rows(notes, count);
    return phonemes(notes, model.forward(input, count, condition), model.consonant_weights());
}

std::vector<V6Phoneme> edit(std::span<const V6Phoneme> timed, std::span<const V6Note> notes, double tick) {
    constexpr double VsmRate = 44100.0 / 256;
    std::vector<V6Phoneme> result(timed.begin(), timed.end());
    std::vector<double> position(timed.size());
    for (size_t k = 0; k < timed.size(); k++) {
        double note_start = notes[timed[k].note].start;
        position[k] = note_start + tick * round_away((round_away(timed[k].start * VsmRate) / VsmRate - note_start) / tick);
    }
    for (size_t n = 0; n < notes.size(); n++) {
        const auto& ticks = notes[n].phoneme_ticks;
        if (!ticks) continue;
        std::vector<size_t> owned;
        for (size_t k = 0; k < timed.size(); k++) {
            if (timed[k].note == static_cast<int>(n)) owned.push_back(k);
        }
        double note_start = notes[n].start;
        auto tick_of = [&](double time) { return round_away((time - note_start) / tick); };
        auto floor_of = [&](double time) { return std::floor((time - note_start) / tick + 1e-9); };
        for (size_t j = 0; j < ticks->size(); j++) {
            if (!(*ticks)[j]) continue;
            int t = *(*ticks)[j];
            size_t k = owned[j];
            double lo = std::max(tick_of(position[k]) - 120, tick_of(0)), hi = tick_of(position[k]) + 120;
            if (k > 0) lo = std::max(lo, std::max(tick_of(position[k - 1]), floor_of(result[k - 1].start)) + 10);
            if (k + 1 < timed.size()) hi = std::min(hi, std::min(tick_of(position[k + 1]), floor_of(result[k + 1].start)) - 10);
            double start = note_start + std::clamp(static_cast<double>(t), std::min(lo, hi), hi) * tick;
            result[k].start = start;
            if (k > 0) result[k - 1].end = start;
        }
    }
    return result;
}

std::vector<Section> sections(std::span<const V6Phoneme> timed) {
    std::vector<Section> sections;
    std::vector<V6Phoneme> current;
    double origin = 0;
    for (size_t i = 0; i < timed.size(); i++) {
        V6Phoneme p = timed[i];
        if (p.name != "sil" || p.duration() <= 0.12) {
            double start = std::max(p.start - origin, 0.06);
            p.start = start;
            p.end -= origin;
            current.push_back(std::move(p));
            continue;
        }
        if (i != 0) {
            V6Phoneme cut = p;
            cut.start = p.start - origin;
            cut.end = p.start - origin + 0.06;
            current.push_back(std::move(cut));
            sections.push_back({origin, std::move(current)});
            current.clear();
        }
        if (i < timed.size() - 1) {
            origin = p.end - 0.06;
            p.start = 0;
            p.end = 0.06;
            current.push_back(std::move(p));
        }
    }
    if (!current.empty()) sections.push_back({origin, std::move(current)});
    return sections;
}

}
