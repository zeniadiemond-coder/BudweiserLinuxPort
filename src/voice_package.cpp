#include "openv6/voice_package.hpp"

#include <cstdlib>
#include <filesystem>
#include <sstream>

#include "openv6/json.hpp"

namespace openv6 {

namespace {

namespace fs = std::filesystem;

std::string scalar(std::string_view raw) {
    std::string value(trim(raw));
    if (!value.empty() && (value[0] == '"' || value[0] == '\'')) {
        size_t close = value.find(value[0], 1);
        return value.substr(1, close == std::string::npos ? std::string::npos : close - 1);
    }
    size_t comment = value.find(" #");
    return std::string(trim(comment == std::string::npos ? value : value.substr(0, comment)));
}

int integer(const std::string& key, const std::string& value) {
    try {
        size_t used;
        int result = std::stoi(value, &used);
        if (used == value.size()) return result;
    } catch (const std::exception&) {
    }
    throw InvalidDataError("v6.yaml " + key + " must be an integer.");
}

V6Config parse_config(const std::string& text) {
    V6Config config;
    std::istringstream lines(text);
    std::string line;
    bool any = false;
    while (std::getline(lines, line)) {
        if (!line.empty() && (line[0] == ' ' || line[0] == '\t' || line[0] == '#' || line[0] == '-')) continue;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        any = true;
        std::string key(trim(std::string_view(line).substr(0, colon)));
        std::string value = scalar(line.substr(colon + 1));
        if (key == "format_version") config.format_version = integer(key, value);
        else if (key == "bank_id") config.bank_id = value;
        else if (key == "language_id") config.language_id = integer(key, value);
        else if (key == "model_directory") config.model_directory = value;
    }
    if (!any) throw InvalidDataError("v6.yaml is empty.");
    return config;
}

}

void V6Config::validate() const {
    if (format_version != 1) throw InvalidDataError("Unsupported V6 package version " + std::to_string(format_version) + ".");
    if (trim(bank_id).empty()) throw InvalidDataError("v6.yaml requires bank_id.");
    if (language_id != 0 && language_id != 1 && language_id != 4) throw InvalidDataError("V6 language_id must be 0 (JP), 1 (EN), or 4 (ZH).");
}

std::string expand_home(const std::string& path) {
    if (path == "~" || path.starts_with("~/") || path.starts_with("~\\")) {
        const char* home = std::getenv("HOME");
        fs::path expanded = fs::path(home ? home : "");
        size_t rest = path.find_first_not_of("/\\", 1);
        return rest == std::string::npos ? expanded.string() : (expanded / path.substr(rest)).string();
    }
    return path;
}

std::string VoicePackage::model_directory() const {
    if (trim(config.model_directory).empty()) return location;
    fs::path full = (fs::absolute(location) / expand_home(config.model_directory)).lexically_normal();
    if (!full.has_filename() && full.has_parent_path()) full = full.parent_path();
    return full.string();
}

VoicePackage load_voice_package(const std::string& directory) {
    VoicePackage package;
    package.location = directory;
    package.config = parse_config(read_text(fs::path(directory) / "v6.yaml"));
    package.config.validate();
    package.registration = V6Registration::read(directory, package.config.bank_id);
    return package;
}

}
