#pragma once

#include <string>

#include "openv6/speakers.hpp"

namespace openv6 {

struct V6Config {
    int format_version = 1;
    std::string bank_id;
    int language_id = 0;
    std::string model_directory;

    void validate() const;
};

struct VoicePackage {
    std::string location;
    V6Config config;
    V6Registration registration;

    std::string model_directory() const;
};

std::string expand_home(const std::string& path);

VoicePackage load_voice_package(const std::string& directory);

}
