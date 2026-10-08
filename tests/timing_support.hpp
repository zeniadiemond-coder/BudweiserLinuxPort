#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "openv6/json.hpp"
#include "openv6/types.hpp"
#include "openv6/util.hpp"

namespace timing_support {

using namespace openv6;

inline std::vector<float> read_floats(const std::filesystem::path& path) {
    std::string bytes = read_text(path);
    std::vector<float> result(bytes.size() / 4);
    std::memcpy(result.data(), bytes.data(), result.size() * 4);
    return result;
}

inline float max_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float max = 0;
    for (size_t i = 0; i < a.size(); i++) max = std::max(max, std::fabs(a[i] - b[i]));
    return max;
}

inline bool same_attributes(const V6Phoneme& a, const V6Phoneme& b) {
    return a.name == b.name && a.note_start == b.note_start && a.nucleus == b.nucleus && a.cls == b.cls && a.pitch == b.pitch;
}

inline double compare(const std::vector<V6Phoneme>& expected, const std::vector<V6Phoneme>& actual) {
    if (expected.size() != actual.size()) return INFINITY;
    double max = 0;
    for (size_t i = 0; i < expected.size(); i++) {
        if (!same_attributes(expected[i], actual[i])) return INFINITY;
        max = std::max({max, std::fabs(expected[i].start - actual[i].start), std::fabs(expected[i].end - actual[i].end)});
    }
    return max;
}

inline V6Phoneme oracle_phoneme(json::Value p) {
    return {p.get_property("Start").get_double(), p.get_property("End").get_double(), std::string(p.get_property("Name").get_string()),
            p.get_property("Pitch").get_single(), p.get_property("NoteStart").get_boolean(), p.get_property("Nucleus").get_boolean(),
            p.get_property("Class").get_int32(), p.get_property("Note").get_int32()};
}

inline std::vector<V6Phoneme> oracle_phonemes(json::Value list) {
    std::vector<V6Phoneme> result;
    for (json::Value p : list.enumerate_array()) result.push_back(oracle_phoneme(p));
    return result;
}

inline std::vector<V6Note> oracle_notes(json::Value list) {
    std::vector<V6Note> notes;
    for (json::Value n : list.enumerate_array()) {
        V6Note note;
        note.start = n.get_property("Start").get_double();
        note.end = n.get_property("End").get_double();
        if (n.get_property("Pitch").kind() == json::Kind::Number) note.pitch = n.get_property("Pitch").get_single();
        for (json::Value p : n.get_property("Phonemes").enumerate_array()) note.phonemes.emplace_back(p.get_string());
        note.meter = n.get_property("Meter").get_int32();
        note.position = n.get_property("Position").get_single();
        note.flag = n.get_property("Flag").get_boolean();
        note.language = n.get_property("Language").get_int32();
        json::Value ticks = n.get_property("PhonemeTicks");
        if (ticks.kind() == json::Kind::Array) {
            std::vector<std::optional<int>> values;
            for (json::Value t : ticks.enumerate_array()) {
                if (t.kind() == json::Kind::Number) values.push_back(t.get_int32());
                else values.push_back(std::nullopt);
            }
            note.phoneme_ticks = std::move(values);
        }
        notes.push_back(std::move(note));
    }
    return notes;
}

inline std::vector<V6Phoneme> capture_phonemes(json::Value list) {
    std::vector<V6Phoneme> result;
    for (json::Value p : list.enumerate_array()) {
        result.push_back({p.get_property("start").get_double(), p.get_property("end").get_double(), std::string(p.get_property("name").get_string()),
                          p.get_property("f10").get_single(), p.get_property("b14").get_int32() != 0, p.get_property("b15").get_int32() != 0,
                          p.get_property("i20").get_int32()});
    }
    return result;
}

inline std::vector<V6Note> capture_notes(json::Value score) {
    std::vector<V6Note> notes;
    for (json::Value n : score.get_property("notes").enumerate_array()) {
        V6Note note;
        note.start = n.get_property("start").get_double();
        note.end = n.get_property("end").get_double();
        if (n.get_property("pitch").kind() == json::Kind::Number) note.pitch = n.get_property("pitch").get_single();
        for (json::Value p : n.get_property("phonemes").enumerate_array()) note.phonemes.emplace_back(p.get_string());
        note.meter = n.get_property("i18").get_int32();
        note.position = n.get_property("f14").get_single();
        note.flag = n.get_property("b21").get_int32() != 0;
        json::Value language;
        if (n.try_get_property("language", language)) note.language = language.get_int32();
        notes.push_back(std::move(note));
    }
    json::Value edits;
    if (score.try_get_property("edits", edits)) {
        for (json::Value e : edits.enumerate_array()) {
            auto& note = notes[e[0].get_int32()];
            if (!note.phoneme_ticks) note.phoneme_ticks = std::vector<std::optional<int>>(note.phonemes.size());
            (*note.phoneme_ticks)[e[1].get_int32()] = e[2].get_int32();
        }
    }
    return notes;
}

inline double capture_tick(json::Value score) {
    json::Value tick;
    return score.try_get_property("tick", tick) ? tick.get_double() : 60.0 / (120 * 480);
}

}
