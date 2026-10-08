#include "openv6/phonemizer.hpp"

#include <algorithm>

#include "openv6/g2p.hpp"

namespace openv6::phonemizer {

namespace {

bool is_blank(std::string_view s) { return trim(s).empty(); }

bool equals_ignore_case(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        unsigned char x = a[i], y = b[i];
        if (x < 0x80 ? std::tolower(x) != std::tolower(y) : x != y) return false;
    }
    return true;
}

bool is_rest_lyric(std::string_view word) {
    for (std::string_view rest : {"r", "rest", "br", "息", "sil", "pau", "sp", "ap", "cl", "っ", "ッ"})
        if (equals_ignore_case(word, rest)) return true;
    return false;
}

size_t kana_prefix(std::string_view s) {
    size_t i = 0;
    while (i + 2 < s.size() && static_cast<unsigned char>(s[i]) == 0xE3) {
        unsigned char b1 = s[i + 1];
        bool in_range = (b1 >= 0x81 && b1 <= 0x83) && (static_cast<unsigned char>(s[i + 2]) & 0xC0) == 0x80;
        if (!in_range) break;
        i += 3;
    }
    return i;
}

size_t pitch_suffix_start(std::string_view word) {
    for (size_t start = 0; start < word.size(); start++) {
        size_t s = start;
        if (word[s] == '_') s++;
        if (s >= word.size() || word[s] < 'A' || word[s] > 'G') continue;
        s++;
        if (s < word.size() && (word[s] == '#' || word[s] == 'b')) s++;
        if (s + 1 == word.size() && word[s] >= '0' && word[s] <= '9') return start;
    }
    return std::string_view::npos;
}

std::string trim_arrows(std::string word) {
    for (;;) {
        if (word.ends_with("\xE2\x86\x91") || word.ends_with("\xE2\x86\x93")) word.resize(word.size() - 3);
        else return word;
    }
}

std::vector<std::string> tokens(std::string_view phonemes) {
    auto parts = split(phonemes, ' ', false);
    return {parts.begin(), parts.end()};
}

constexpr std::string_view Breath = "br";

struct Note {
    int64_t position;
    int64_t duration;
    std::string lyric;
    std::optional<std::string> phonetic_hint;
};

using NoteGroup = std::vector<Note>;

struct GroupResult {
    bool breath = false;
    std::vector<std::vector<std::string>> tokens;
    std::string error;
};

}

std::optional<std::string> clean_lyric(int language_id, std::string_view lyric) {
    auto parts = tokens(lyric);
    if (parts.empty()) return std::nullopt;
    std::string word = parts.back();
    size_t kana = language_id == 0 ? kana_prefix(word) : 0;
    if (kana > 0) {
        word.resize(kana);
    } else {
        size_t start = pitch_suffix_start(word);
        if (start != std::string_view::npos) word.resize(start);
        word = trim_arrows(std::move(word));
    }
    if (word.empty() || is_rest_lyric(word)) return std::nullopt;
    return word;
}

bool is_rest(int language_id, std::optional<std::string_view> phonetic_hint, std::string_view lyric) {
    return (!phonetic_hint || is_blank(*phonetic_hint)) && !clean_lyric(language_id, lyric);
}

bool is_breath(std::string_view lyric) { return equals_ignore_case(trim(lyric), Breath) || equals_ignore_case(trim(lyric), "息"); }

bool is_vowel_extension(std::string_view lyric) { return lyric.starts_with("+~") || lyric.starts_with("+*"); }

namespace {

std::vector<GroupResult> convert(const std::vector<NoteGroup>& groups, int language_id, const std::string& native_directory) {
    std::vector<GroupResult> results(groups.size());
    for (size_t g = 0; g < groups.size(); g++) {
        const auto& group = groups[g];
        const auto& lead = group[0];
        auto& result = results[g];
        if (is_rest(language_id, lead.phonetic_hint, lead.lyric)) {
            result.breath = is_breath(lead.lyric);
            continue;
        }
        std::string lyric = clean_lyric(language_id, lead.lyric).value_or(lead.lyric);
        std::vector<bool> extend;
        for (const auto& note : group) extend.push_back(is_vowel_extension(note.lyric));
        try {
            std::optional<std::string_view> hint;
            if (lead.phonetic_hint) hint = *lead.phonetic_hint;
            for (const auto& phonemes : g2p::assign(lyric, hint, language_id, extend, native_directory)) result.tokens.push_back(tokens(phonemes));
        } catch (const InvalidDataError& e) {
            result.error = e.what();
        }
    }
    return results;
}

}

void phonemize_score(V6Score& score, const std::string& native_directory) {
    auto& notes = score.notes;
    std::vector<NoteGroup> groups;
    std::vector<std::vector<size_t>> members;
    std::vector<int> languages;
    for (size_t i = 0; i < notes.size(); i++) {
        const auto& note = notes[i];
        if (!note.phonemes.empty()) continue;
        bool extends = !groups.empty() && note.lyric.starts_with("+") && members.back().back() == i - 1 &&
                       groups.back().back().position + groups.back().back().duration == note.startTick;
        Note converted{note.startTick, note.durationTick, note.lyric, note.phoneticHint};
        if (extends) {
            groups.back().push_back(std::move(converted));
            members.back().push_back(i);
        } else {
            groups.push_back({std::move(converted)});
            members.push_back({i});
            languages.push_back(note.languageId.value_or(score.languageId));
        }
    }
    std::vector<bool> dropped(notes.size());
    std::vector<V6Breath> breaths = score.breaths.value_or(std::vector<V6Breath>{});
    for (int language : {0, 1, 4}) {
        std::vector<NoteGroup> subset;
        std::vector<size_t> origin;
        for (size_t g = 0; g < groups.size(); g++) {
            if (languages[g] != language) continue;
            subset.push_back(groups[g]);
            origin.push_back(g);
        }
        if (subset.empty()) continue;
        auto results = convert(subset, language, native_directory);
        for (size_t k = 0; k < subset.size(); k++) {
            const auto& result = results[k];
            const auto& indices = members[origin[k]];
            if (!result.error.empty())
                throw InvalidDataError("Note " + std::to_string(indices[0] + 1) + " ('" + notes[indices[0]].lyric + "'): " + result.error);
            for (size_t n = 0; n < indices.size(); n++) {
                auto& note = notes[indices[n]];
                if (result.tokens.empty()) {
                    dropped[indices[n]] = true;
                    if (result.breath) breaths.push_back({note.startTick, note.durationTick});
                    continue;
                }
                for (const auto& token : result.tokens[n]) note.phonemes += (note.phonemes.empty() ? "" : " ") + token;
                note.languageId = language;
            }
        }
    }
    if (!breaths.empty()) score.breaths = std::move(breaths);
    size_t kept = 0;
    for (size_t i = 0; i < notes.size(); i++)
        if (!dropped[i]) {
            if (kept != i) notes[kept] = std::move(notes[i]);
            kept++;
        }
    notes.resize(kept);
}

}
