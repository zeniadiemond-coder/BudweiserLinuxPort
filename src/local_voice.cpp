#include "openv6/local_voice.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

#include "openv6/timing.hpp"
#include "openv6/vsm_pitch.hpp"
#include "openv6/phoneme_conversion.hpp"

namespace openv6 {

namespace fs = std::filesystem;

namespace {

constexpr double PrerollSeconds = 0.5;
constexpr int64_t TailTicks = 1440, BarTicks = 1920;

int s5_language(int vsm_language) {
    switch (vsm_language) {
        case 0: return 0;
        case 1: return 1;
        case 4: return 2;
        default: throw std::out_of_range("vsm_language");
    }
}

std::string lower(std::string_view s) {
    std::string result(s);
    for (char& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return result;
}

bool find_property(json::Value object, std::string_view name, json::Value& out) {
    if (object.kind() != json::Kind::Object) throw InvalidDataError("Score: expected a JSON object.");
    std::string wanted = lower(name);
    bool found = false;
    for (auto member : object.enumerate_object()) {
        if (lower(member.name) == wanted) {
            out = member.value;
            found = true;
        }
    }
    return found && out.kind() != json::Kind::Null;
}

json::Value required(json::Value object, std::string_view name) {
    json::Value value;
    if (!find_property(object, name, value)) throw InvalidDataError("Score: missing property '" + std::string(name) + "'.");
    return value;
}

json::Value array_of(json::Value value, std::string_view name) {
    if (value.kind() != json::Kind::Array) throw InvalidDataError("Score: '" + std::string(name) + "' must be an array.");
    return value;
}

template <class T>
std::vector<T> numbers(json::Value value, std::string_view name, T (json::Value::*read)() const) {
    std::vector<T> result;
    for (auto item : array_of(value, name).enumerate_array()) result.push_back((item.*read)());
    return result;
}

}

std::span<const std::pair<std::string_view, float AiNoteExpression::*>> ai_param_names() {
    static const std::pair<std::string_view, float AiNoteExpression::*> names[] = {
        {"pitchFine", &AiNoteExpression::pitch_fine},
        {"pitchDriftStart", &AiNoteExpression::pitch_drift_start},
        {"pitchDriftEnd", &AiNoteExpression::pitch_drift_end},
        {"pitchScalingCenter", &AiNoteExpression::pitch_scaling_center},
        {"pitchScalingOrigin", &AiNoteExpression::pitch_scaling_origin},
        {"pitchTransitionStart", &AiNoteExpression::pitch_transition_start},
        {"pitchTransitionEnd", &AiNoteExpression::pitch_transition_end},
        {"amplitudeWhole", &AiNoteExpression::amplitude_whole},
        {"amplitudeStart", &AiNoteExpression::amplitude_start},
        {"amplitudeEnd", &AiNoteExpression::amplitude_end},
        {"vibratoLeadingDepth", &AiNoteExpression::vibrato_leading_depth},
        {"vibratoFollowingDepth", &AiNoteExpression::vibrato_following_depth},
    };
    return names;
}

namespace {

V6ScoreNote parse_note(json::Value value) {
    V6ScoreNote note{};
    json::Value v;
    note.startTick = required(value, "startTick").get_int64();
    note.durationTick = required(value, "durationTick").get_int64();
    note.pitch = required(value, "pitch").get_int32();
    if (find_property(value, "lyric", v)) note.lyric = v.get_string();
    if (find_property(value, "phonemes", v)) note.phonemes = v.get_string();
    if (find_property(value, "languageId", v)) note.languageId = v.get_int32();
    if (find_property(value, "phonemeTicks", v)) {
        std::vector<std::optional<int>> ticks;
        for (auto item : array_of(v, "phonemeTicks").enumerate_array())
            ticks.push_back(item.kind() == json::Kind::Null ? std::nullopt : std::optional<int>(item.get_int32()));
        note.phonemeTicks = std::move(ticks);
    }
    if (find_property(value, "phoneticHint", v)) note.phoneticHint = std::string(v.get_string());
    if (find_property(value, "aiParams", v)) {
        json::Value parameter;
        for (auto [name, member] : ai_param_names())
            if (find_property(v, name, parameter)) note.aiParams.*member = parameter.get_single();
    }
    if (find_property(value, "vibrato", v)) {
        V6ScoreVibrato vibrato;
        json::Value field;
        if (find_property(v, "depth", field)) vibrato.depth = field.get_int32();
        if (find_property(v, "rate", field)) vibrato.rate = field.get_int32();
        if (find_property(v, "duration", field)) vibrato.durationTick = field.get_int64();
        if (find_property(v, "ai", field)) vibrato.ai = field.get_boolean();
        note.vibrato = vibrato;
    }
    return note;
}

}

V6Score parse_score(json::Value root) {
    V6Score score{};
    json::Value v;
    score.tempoBpm = required(root, "tempoBpm").get_double();
    score.languageId = required(root, "languageId").get_int32();
    for (auto note : array_of(required(root, "notes"), "notes").enumerate_array()) score.notes.push_back(parse_note(note));
    if (find_property(root, "pitch", v)) score.pitch = numbers<float>(v, "pitch", &json::Value::get_single);
    if (find_property(root, "pitchTicks", v)) score.pitchTicks = numbers<int64_t>(v, "pitchTicks", &json::Value::get_int64);
    if (find_property(root, "controllers", v)) {
        std::vector<V6Controller> controllers;
        for (auto item : array_of(v, "controllers").enumerate_array()) {
            V6Controller controller;
            controller.name = required(item, "name").get_string();
            for (auto event : array_of(required(item, "events"), "events").enumerate_array())
                controller.events.push_back({required(event, "tick").get_int64(), required(event, "value").get_int32()});
            controllers.push_back(std::move(controller));
        }
        score.controllers = std::move(controllers);
    }
    if (find_property(root, "breaths", v)) {
        std::vector<V6Breath> breaths;
        for (auto item : array_of(v, "breaths").enumerate_array())
            breaths.push_back({required(item, "startTick").get_int64(), required(item, "durationTick").get_int64()});
        score.breaths = std::move(breaths);
    }
    return score;
}

V6Score parse_score(std::string_view json_text) {
    auto document = json::Document::parse(std::string(json_text));
    return parse_score(document->root());
}

void validate_score(const V6Score& score, bool phonemized) {
    constexpr int64_t MaxTick = std::numeric_limits<int>::max() - TailTicks;
    auto fail = [](const std::string& message) { throw InvalidDataError("Invalid score: " + message); };
    auto exceeds = [&](int64_t start, int64_t duration) { return start > MaxTick || duration > MaxTick - start; };
    if (score.languageId != 0 && score.languageId != 1 && score.languageId != 4) fail("languageId must be 0 (JP), 1 (EN) or 4 (ZH).");
    if (!(score.tempoBpm > 0) || !std::isfinite(score.tempoBpm)) fail("tempoBpm must be positive.");
    if (score.notes.empty()) fail("no notes to render.");
    for (size_t i = 0; i < score.notes.size(); i++) {
        const auto& note = score.notes[i];
        std::string where = "note " + std::to_string(i + 1) + " ('" + note.lyric + "') ";
        if (note.startTick < 0) fail(where + "starts before tick 0.");
        if (note.durationTick <= 0) fail(where + "has no duration.");
        if (exceeds(note.startTick, note.durationTick)) fail(where + "exceeds the native tick range.");
        if (note.pitch < 0 || note.pitch > 127) fail(where + "pitch must be 0-127.");
        if (note.languageId && *note.languageId != 0 && *note.languageId != 1 && *note.languageId != 4) fail(where + "languageId must be 0, 1 or 4.");
        for (auto [name, member] : ai_param_names())
            if (!(note.aiParams.*member >= 0 && note.aiParams.*member <= 1)) fail(where + "aiParams." + std::string(name) + " must be within 0..1.");
        if (note.vibrato) {
            if (note.vibrato->depth < 0 || note.vibrato->depth > 127) fail(where + "vibrato.depth must be 0-127.");
            if (note.vibrato->rate < 0 || note.vibrato->rate > 127) fail(where + "vibrato.rate must be 0-127.");
            if (note.vibrato->durationTick && (*note.vibrato->durationTick < 0 || *note.vibrato->durationTick > note.durationTick))
                fail(where + "vibrato.duration must be within the note.");
        }
        if (phonemized && note.phonemes.find_first_not_of(' ') == std::string::npos) fail(where + "has no phonemes and no usable lyric.");
        if (i > 0 && note.startTick < score.notes[i - 1].startTick + score.notes[i - 1].durationTick) fail(where + "overlaps the previous note; scores must be monophonic.");
    }
    if (score.pitch) {
        if (score.pitch->empty()) fail("pitch is empty.");
        for (float v : *score.pitch)
            if (!std::isfinite(v)) fail("pitch contains a non-finite value.");
        if (!score.pitchTicks || score.pitchTicks->size() != score.pitch->size()) fail("pitch needs pitchTicks of the same length.");
        if ((*score.pitchTicks)[0] < 0 || !std::is_sorted(score.pitchTicks->begin(), score.pitchTicks->end())) fail("pitchTicks must be non-negative and non-decreasing.");
        if (score.pitchTicks->back() > MaxTick) fail("pitchTicks exceed the native tick range.");
    }
    if (score.controllers) {
        for (const auto& controller : *score.controllers) {
            if (controller.name.empty() || controller.events.empty()) fail("a controller needs a name and events.");
            for (size_t i = 0; i < controller.events.size(); i++) {
                if (controller.events[i].tick < 0 || (i > 0 && controller.events[i].tick <= controller.events[i - 1].tick))
                    fail("controller " + controller.name + " events must use strictly increasing non-negative ticks.");
            }
            if (controller.events.back().tick > MaxTick) fail("controller " + controller.name + " events exceed the native tick range.");
        }
    }
    if (score.breaths) {
        for (const auto& breath : *score.breaths) {
            if (breath.startTick < -MaxTick || exceeds(breath.startTick, breath.durationTick)) fail("a breath exceeds the native tick range.");
        }
    }
}

namespace {

std::string existing_or(const fs::path& preferred, const fs::path& fallback) { return (fs::exists(preferred) ? preferred : fallback).string(); }

}

ScoreNotes build_notes(const V6Score& score, double tick) {
    auto time = [&](int64_t t) { return PrerollSeconds + t * tick; };
    auto rest = [&](int64_t from, int64_t to) {
        V6Note note;
        note.start = from < 0 ? 0 : time(from);
        note.end = time(to);
        note.phonemes = {"sil"};
        note.position = (from < 0 ? to : from) % BarTicks / static_cast<float>(BarTicks);
        note.language = s5_language(score.languageId);
        return note;
    };
    ScoreNotes result;
    result.index.resize(score.notes.size());
    result.sources.resize(score.notes.size());
    int64_t end = -static_cast<int64_t>(round_even(PrerollSeconds / tick));
    for (size_t i = 0; i < score.notes.size(); i++) {
        const auto& note = score.notes[i];
        if (note.startTick > end) result.notes.push_back(rest(end, note.startTick));
        int language = note.languageId.value_or(score.languageId);
        auto phonemes = phoneme_conversion::to_model(note.phonemes, language);
        std::optional<std::vector<std::optional<int>>> ticks;
        if (note.phonemeTicks) {
            ticks.emplace();
            for (size_t k = 0; k < phonemes.size(); k++) {
                bool repeat = k > 0 && phonemes[k - 1].source == phonemes[k].source;
                ticks->push_back(repeat ? std::nullopt : note.phonemeTicks->at(phonemes[k].source));
            }
        }
        std::vector<std::string> names;
        for (const auto& p : phonemes) names.push_back(p.name);
        bool extension = names.size() == 1 && names[0] == "-";
        if (extension) {
            names = {"sil"};
            for (auto it = result.notes.rbegin(); it != result.notes.rend(); ++it) {
                if (!it->rest()) {
                    names = {it->phonemes.back()};
                    break;
                }
            }
        }
        result.index[i] = static_cast<int>(result.notes.size());
        for (const auto& p : phonemes) result.sources[i].push_back(p.source);
        V6Note sung;
        sung.start = time(note.startTick);
        sung.end = time(note.startTick + note.durationTick);
        float cents = (note.pitch - 69) * 100.f;
        sung.pitch = std::clamp(cents, -4500.f, 2400.f);
        sung.pitch_offset = *sung.pitch - cents;
        sung.expression = note.aiParams;
        if (note.vibrato) {
            sung.vibrato_depth = note.vibrato->depth;
            sung.vibrato_rate = note.vibrato->rate;
            sung.ai_vibrato = note.vibrato->ai;
            sung.vibrato_start = time(note.startTick + note.durationTick - note.vibrato->durationTick.value_or(note.durationTick * 2 / 3));
        }
        sung.phonemes = std::move(names);
        sung.position = note.startTick % BarTicks / static_cast<float>(BarTicks);
        sung.flag = extension;
        sung.language = s5_language(language);
        sung.phoneme_ticks = std::move(ticks);
        result.notes.push_back(std::move(sung));
        end = note.startTick + note.durationTick;
    }
    result.notes.push_back(rest(end, end + TailTicks));
    return result;
}

namespace {

std::optional<std::vector<int>> curve(const V6Score& score, std::string_view name, int frames, const std::function<double(int64_t)>& time) {
    if (!score.controllers) return std::nullopt;
    for (const auto& controller : *score.controllers) {
        if (controller.name != name) continue;
        std::vector<int> result(frames);
        size_t count = controller.events.size();
        int64_t e = -1;
        for (int f = 0; f < frames; f++) {
            while (e + 1 < static_cast<int64_t>(count) && time(controller.events[e + 1].tick) <= f / 200.0) e++;
            result[f] = controller.events.at(e < 0 ? 0 : e).value;
        }
        return result;
    }
    return std::nullopt;
}

}

V6LocalVoice::V6LocalVoice(const std::string& models, const V6Registration& registration, int language, std::optional<int> np_idx) {
    fs::path directory(models);
    std::string timing_path = existing_or(directory / (registration.timbre_model + ".vtmg"), directory / "VOCALOID6.vtmg");
    std::string pitch_path = existing_or(directory / (registration.pitch_model + ".vpit"), directory / "VOCALOID6.vpit");
    condition_ = registration.timing_condition.value_or(timing::Conditions[s5_language(language)][0]);
    synthesizer_ = std::make_unique<V6Synthesizer>(
        std::make_shared<ModelFile>(ModelFile::load((directory / (registration.timbre_model + ".vtb2")).string())),
        std::make_shared<ModelFile>(ModelFile::load(pitch_path)), std::make_shared<VtmgModel>(VtmgModel::load(timing_path)),
        np_idx.value_or(registration.np_idx), registration.param_exp, registration.param_dyn);
}

LocalRenderResult V6LocalVoice::render(const V6Score& score) const {
    validate_score(score);
    double tick = 60.0 / (score.tempoBpm * 480);
    auto time = [&](int64_t t) { return PrerollSeconds + t * tick; };
    auto notes = build_notes(score, tick).notes;
    auto timed = synthesizer_->time(notes, condition_, tick);
    int frames = static_cast<int>(round_away(notes.back().end * 200));
    std::optional<V6PitchEdit> pitch_edit;
    if (score.pitch) pitch_edit = V6PitchEdit{score.pitchTicks.value(), *score.pitch, tick, PrerollSeconds};
    V6Controls controls;
    controls.expression = curve(score, "Expression", frames, time);
    controls.dynamics = curve(score, "Dynamics", frames, time);
    controls.air = curve(score, "S5Air", frames, time);
    controls.character = curve(score, "S5Character", frames, time);
    std::vector<TimeSpan> breaths;
    if (score.breaths)
        for (const auto& b : *score.breaths)
            if (b.durationTick > 0) breaths.push_back({time(b.startTick), time(b.startTick + b.durationTick)});
    auto result = synthesizer_->render(notes, timed, controls, pitch_edit ? &*pitch_edit : nullptr, breaths);
    V6Pitch pitch{5, {}};
    pitch.pitches.reserve(result.f0.size());
    for (float c : result.f0) pitch.pitches.push_back(c == vsm_pitch::Gap ? -1.f : (c + 6900.f) / 100.f);
    return {std::move(result.samples), std::move(pitch)};
}

std::vector<std::vector<int>> V6LocalVoice::phoneme_positions(const V6Score& score) const {
    validate_score(score);
    constexpr double VsmRate = 44100.0 / 256;
    double tick = 60.0 / (score.tempoBpm * 480);
    auto built = build_notes(score, tick);
    auto timed = synthesizer_->time(built.notes, condition_, tick);
    std::vector<std::vector<int>> positions;
    for (size_t i = 0; i < score.notes.size(); i++) {
        std::vector<const V6Phoneme*> own;
        for (const auto& p : timed)
            if (p.note == built.index[i]) own.push_back(&p);
        double start = built.notes[built.index[i]].start;
        auto to_tick = [&](double t) { return static_cast<int>(round_away((round_away(t * VsmRate) / VsmRate - start) / tick)); };
        std::vector<const V6Phoneme*> entered;
        for (size_t k = 0; k < own.size(); k++)
            if (!(k > 0 && own[k]->name == "br" && own[k - 1]->name == "sil")) entered.push_back(own[k]);
        const auto& sources = built.sources[i];
        std::vector<int> row;
        for (size_t k = 0; k < entered.size(); k++)
            if (k == 0 || sources.at(k) != sources.at(k - 1)) row.push_back(to_tick(entered[k]->start));
        row.push_back(to_tick(own.back()->end));
        positions.push_back(std::move(row));
    }
    return positions;
}

}
