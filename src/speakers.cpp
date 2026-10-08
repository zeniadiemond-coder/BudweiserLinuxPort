#include "openv6/speakers.hpp"

#include <filesystem>
#include <memory>

#include "openv6/json.hpp"
#include "openv6/util.hpp"

extern const char* const openv6_speakers_json;

namespace openv6 {

V6Registration V6Registration::read(const std::string& location, const std::string& bank_id) {
    auto bank = json::Document::parse(read_text(std::filesystem::path(location) / "bank.json"));
    json::Value component = bank->root().get_property("Components\\" + bank_id);
    json::Value value;
    auto integer = [&](std::string_view name, int fallback) { return component.try_get_property(name, value) ? value.get_int32() : fallback; };
    V6Registration registration;
    registration.pitch_model = std::string(component.get_property("PModelName").get_string());
    registration.timbre_model = component.try_get_property("TModelName", value) ? std::string(value.get_string()) : registration.pitch_model;
    registration.np_idx = integer("NPIdx", 0);
    if (component.try_get_property("TIdxs1", value)) registration.timing_condition = value.get_int32() & 0xff;
    registration.param_exp = integer("ParamExp", 3000);
    registration.param_dyn = integer("ParamDyn", 3000);
    return registration;
}

namespace speakers {

namespace {

const json::Value& tables() {
    static const auto document = json::Document::parse(openv6_speakers_json);
    static const json::Value root = document->root().get_property("tables");
    return root;
}

bool lists(json::Value table, std::string_view model) {
    for (json::Value name : table.get_property("models").enumerate_array()) {
        if (name.get_string() == model) return true;
    }
    return false;
}

bool find(const V6Registration& registration, json::Value& table) {
    for (json::Value candidate : tables().enumerate_array()) {
        if (lists(candidate, registration.timbre_model)) {
            table = candidate;
            return true;
        }
    }
    return false;
}

std::string describe(size_t index, json::Value row) {
    json::Value value;
    std::string label;
    if (row.try_get_property("label", value) && value.kind() == json::Kind::String) label = value.get_string();
    std::string name = label.empty() ? "Row " + std::to_string(index) : std::to_string(index) + ": " + label;
    if (row.try_get_property("voices", value) && value.kind() == json::Kind::Array) {
        name += " (";
        bool first = true;
        for (json::Value voice : value.enumerate_array()) {
            if (!first) name += ", ";
            name += voice.get_string();
            first = false;
        }
        name += ")";
    }
    return name;
}

}

std::vector<V6Speaker> available(const V6Registration& registration) {
    json::Value table;
    if (!find(registration, table)) return {};
    std::vector<V6Speaker> result;
    for (json::Value row : table.get_property("rows").enumerate_array()) {
        int index = static_cast<int>(result.size());
        result.push_back({index, describe(index, row)});
    }
    return result;
}

int np_idx(const V6Registration& registration, int row) {
    if (row == registration.timbre_row()) return registration.np_idx;
    json::Value table;
    if (!find(registration, table)) throw InvalidDataError(registration.timbre_model + " has no known speakers.");
    json::Value rows = table.get_property("rows");
    if (row < 0 || row >= static_cast<int>(rows.array_length())) {
        throw InvalidDataError(registration.timbre_model + " has no speaker row " + std::to_string(row) + ".");
    }
    json::Value pitch;
    int pitch_row = rows[row].try_get_property("pitch", pitch) && pitch.kind() == json::Kind::Number ? pitch.get_int32() : registration.pitch_row();
    return row | pitch_row << 8;
}

}

}
