#pragma once

#include <optional>
#include <string>
#include <vector>

namespace openv6 {

struct V6Registration {
    std::string pitch_model, timbre_model;
    int np_idx = 0;
    std::optional<int> timing_condition;
    int param_exp = 3000, param_dyn = 3000;

    int timbre_row() const { return np_idx & 0xff; }
    int pitch_row() const { return (np_idx >> 8) & 0xff; }

    static V6Registration read(const std::string& location, const std::string& bank_id);
};

struct V6Speaker {
    int row;
    std::string name;
};

namespace speakers {

std::vector<V6Speaker> available(const V6Registration& registration);

int np_idx(const V6Registration& registration, int row);

}

}
