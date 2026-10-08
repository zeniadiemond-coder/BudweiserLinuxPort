#include "openv6/json.hpp"

#include <charconv>
#include <cstring>

namespace openv6::json {

class Document::Parser {
public:
    explicit Parser(Document& doc) : doc_(doc), p_(doc.text_.data()), end_(p_ + doc.text_.size()) {}

    void run() {
        doc_.root_ = parse_value(0);
        skip_space();
        if (p_ != end_) fail("trailing characters");
    }

private:
    [[noreturn]] void fail(const char* what) const {
        throw InvalidDataError(std::string("JSON: ") + what + " at offset " + std::to_string(p_ - doc_.text_.data()));
    }

    void skip_space() {
        while (p_ != end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
    }

    uint32_t add(Kind kind, uint32_t first = 0, uint32_t count = 0, std::string_view text = {}) {
        doc_.nodes_.push_back({kind, first, count, text});
        return static_cast<uint32_t>(doc_.nodes_.size() - 1);
    }

    bool literal(std::string_view word) {
        if (static_cast<size_t>(end_ - p_) < word.size() || std::memcmp(p_, word.data(), word.size()) != 0) return false;
        p_ += word.size();
        return true;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    uint32_t hex4() {
        if (end_ - p_ < 4) fail("bad \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad \\u escape");
        }
        return v;
    }

    std::string_view parse_string() {
        const char* start = p_;
        const char* quote = static_cast<const char*>(std::memchr(p_, '"', end_ - p_));
        if (!quote) fail("unterminated string");
        if (!std::memchr(start, '\\', quote - start)) {
            p_ = quote + 1;
            return {start, static_cast<size_t>(quote - start)};
        }
        std::string out;
        while (true) {
            if (p_ == end_) fail("unterminated string");
            char c = *p_++;
            if (c == '"') break;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (p_ == end_) fail("unterminated string");
            switch (char e = *p_++) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                        p_ += 2;
                        uint32_t low = hex4();
                        if (low < 0xDC00 || low > 0xDFFF) fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
                    (void)e;
            }
        }
        doc_.unescaped_.push_back(std::move(out));
        return doc_.unescaped_.back();
    }

    uint32_t parse_value(int depth) {
        if (depth > 256) fail("nesting too deep");
        skip_space();
        if (p_ == end_) fail("unexpected end");
        switch (*p_) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': {
                ++p_;
                return add(Kind::String, 0, 0, parse_string());
            }
            case 't':
                if (literal("true")) return add(Kind::True);
                fail("bad literal");
            case 'f':
                if (literal("false")) return add(Kind::False);
                fail("bad literal");
            case 'n':
                if (literal("null")) return add(Kind::Null);
                fail("bad literal");
            default: {
                const char* start = p_;
                if (*p_ == '-') ++p_;
                const char* digits = p_;
                while (p_ != end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '.' || *p_ == 'e' || *p_ == 'E' || *p_ == '+' || *p_ == '-')) ++p_;
                if (p_ == digits) fail("unexpected character");
                return add(Kind::Number, 0, 0, {start, static_cast<size_t>(p_ - start)});
            }
        }
    }

    uint32_t parse_array(int depth) {
        ++p_;
        size_t base = scratch_items_.size();
        skip_space();
        if (p_ != end_ && *p_ == ']') {
            ++p_;
        } else {
            while (true) {
                uint32_t v = parse_value(depth + 1);
                scratch_items_.push_back(v);
                skip_space();
                if (p_ == end_) fail("unterminated array");
                if (*p_ == ',') { ++p_; continue; }
                if (*p_ == ']') { ++p_; break; }
                fail("expected , or ]");
            }
        }
        uint32_t first = static_cast<uint32_t>(doc_.items_.size());
        doc_.items_.insert(doc_.items_.end(), scratch_items_.begin() + base, scratch_items_.end());
        uint32_t count = static_cast<uint32_t>(scratch_items_.size() - base);
        scratch_items_.resize(base);
        return add(Kind::Array, first, count);
    }

    uint32_t parse_object(int depth) {
        ++p_;
        size_t base = scratch_members_.size();
        skip_space();
        if (p_ != end_ && *p_ == '}') {
            ++p_;
        } else {
            while (true) {
                skip_space();
                if (p_ == end_ || *p_ != '"') fail("expected property name");
                ++p_;
                std::string_view name = parse_string();
                skip_space();
                if (p_ == end_ || *p_ != ':') fail("expected :");
                ++p_;
                uint32_t v = parse_value(depth + 1);
                scratch_members_.push_back({name, v});
                skip_space();
                if (p_ == end_) fail("unterminated object");
                if (*p_ == ',') { ++p_; continue; }
                if (*p_ == '}') { ++p_; break; }
                fail("expected , or }");
            }
        }
        uint32_t first = static_cast<uint32_t>(doc_.members_.size());
        doc_.members_.insert(doc_.members_.end(), scratch_members_.begin() + base, scratch_members_.end());
        uint32_t count = static_cast<uint32_t>(scratch_members_.size() - base);
        scratch_members_.resize(base);
        return add(Kind::Object, first, count);
    }

    Document& doc_;
    const char* p_;
    const char* end_;
    std::vector<uint32_t> scratch_items_;
    std::vector<MemberEntry> scratch_members_;
};

std::unique_ptr<Document> Document::parse(std::string text) {
    std::unique_ptr<Document> doc(new Document());
    doc->text_ = std::move(text);
    Parser(*doc).run();
    return doc;
}

namespace {
[[noreturn]] void wrong_kind(const char* wanted) {
    throw InvalidDataError(std::string("JSON value is not ") + wanted);
}
}

Kind Value::kind() const { return doc_ ? doc_->nodes_[index_].kind : Kind::Undefined; }

bool Value::try_get_property(std::string_view name, Value& out) const {
    if (kind() != Kind::Object) wrong_kind("an object");
    const auto& n = doc_->nodes_[index_];
    for (uint32_t i = n.first + n.count; i-- > n.first;) {
        const auto& m = doc_->members_[i];
        if (m.name == name) {
            out = Value(doc_, m.value);
            return true;
        }
    }
    return false;
}

Value Value::get_property(std::string_view name) const {
    Value v;
    if (!try_get_property(name, v)) throw InvalidDataError("JSON property not found: " + std::string(name));
    return v;
}

Value Value::operator[](size_t index) const {
    if (kind() != Kind::Array) wrong_kind("an array");
    const auto& n = doc_->nodes_[index_];
    if (index >= n.count) throw InvalidDataError("JSON array index out of range");
    return Value(doc_, doc_->items_[n.first + index]);
}

size_t Value::array_length() const {
    if (kind() != Kind::Array) wrong_kind("an array");
    return doc_->nodes_[index_].count;
}

size_t Value::object_length() const {
    if (kind() != Kind::Object) wrong_kind("an object");
    return doc_->nodes_[index_].count;
}

Value::ArrayRange Value::enumerate_array() const {
    if (kind() != Kind::Array) wrong_kind("an array");
    const auto& n = doc_->nodes_[index_];
    const uint32_t* b = doc_->items_.data() + n.first;
    return {doc_, b, b + n.count};
}

Value::ObjectRange Value::enumerate_object() const {
    if (kind() != Kind::Object) wrong_kind("an object");
    const auto& n = doc_->nodes_[index_];
    return {doc_, n.first, n.count};
}

Member Value::ObjectRange::iterator::operator*() const {
    const auto& m = doc_->members_[i_];
    return {m.name, Value(doc_, m.value)};
}

std::string_view Value::get_string() const {
    if (kind() != Kind::String) wrong_kind("a string");
    return doc_->nodes_[index_].text;
}

bool Value::get_boolean() const {
    switch (kind()) {
        case Kind::True: return true;
        case Kind::False: return false;
        default: wrong_kind("a boolean");
    }
}

namespace {
template <class T>
T parse_number(std::string_view text) {
    T v{};
    auto r = std::from_chars(text.data(), text.data() + text.size(), v);
    if (r.ec != std::errc() || r.ptr != text.data() + text.size()) wrong_kind("a representable number");
    return v;
}
}

int32_t Value::get_int32() const {
    if (kind() != Kind::Number) wrong_kind("a number");
    return parse_number<int32_t>(doc_->nodes_[index_].text);
}

int64_t Value::get_int64() const {
    if (kind() != Kind::Number) wrong_kind("a number");
    return parse_number<int64_t>(doc_->nodes_[index_].text);
}

float Value::get_single() const {
    if (kind() != Kind::Number) wrong_kind("a number");
    return parse_number<float>(doc_->nodes_[index_].text);
}

double Value::get_double() const {
    if (kind() != Kind::Number) wrong_kind("a number");
    return parse_number<double>(doc_->nodes_[index_].text);
}

}
