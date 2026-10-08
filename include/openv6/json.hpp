#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "openv6/util.hpp"

namespace openv6::json {

enum class Kind { Undefined, Object, Array, String, Number, True, False, Null };

class Document;

struct Member;

class Value {
public:
    class ArrayRange;
    class ObjectRange;

    Value() = default;

    Kind kind() const;

    Value get_property(std::string_view name) const;
    bool try_get_property(std::string_view name, Value& out) const;

    Value operator[](size_t index) const;
    size_t array_length() const;
    size_t object_length() const;
    ArrayRange enumerate_array() const;
    ObjectRange enumerate_object() const;

    std::string_view get_string() const;
    bool get_boolean() const;
    int32_t get_int32() const;
    int64_t get_int64() const;
    float get_single() const;
    double get_double() const;

private:
    friend class Document;
    friend class ArrayRange;
    friend class ObjectRange;
    Value(const Document* doc, uint32_t index) : doc_(doc), index_(index) {}
    const Document* doc_ = nullptr;
    uint32_t index_ = 0;
};

struct Member {
    std::string_view name;
    Value value;
};

class Value::ArrayRange {
public:
    class iterator {
    public:
        Value operator*() const { return Value(doc_, *p_); }
        iterator& operator++() { ++p_; return *this; }
        bool operator!=(const iterator& o) const { return p_ != o.p_; }
    private:
        friend class ArrayRange;
        iterator(const Document* d, const uint32_t* p) : doc_(d), p_(p) {}
        const Document* doc_;
        const uint32_t* p_;
    };
    iterator begin() const { return {doc_, begin_}; }
    iterator end() const { return {doc_, end_}; }
    size_t size() const { return end_ - begin_; }
private:
    friend class Value;
    ArrayRange(const Document* d, const uint32_t* b, const uint32_t* e) : doc_(d), begin_(b), end_(e) {}
    const Document* doc_;
    const uint32_t *begin_, *end_;
};

class Value::ObjectRange {
public:
    class iterator {
    public:
        Member operator*() const;
        iterator& operator++() { ++i_; return *this; }
        bool operator!=(const iterator& o) const { return i_ != o.i_; }
    private:
        friend class ObjectRange;
        iterator(const Document* d, uint32_t i) : doc_(d), i_(i) {}
        const Document* doc_;
        uint32_t i_;
    };
    iterator begin() const { return {doc_, first_}; }
    iterator end() const { return {doc_, first_ + count_}; }
    size_t size() const { return count_; }
private:
    friend class Value;
    ObjectRange(const Document* d, uint32_t f, uint32_t c) : doc_(d), first_(f), count_(c) {}
    const Document* doc_;
    uint32_t first_, count_;
};

class Document {
public:
    static std::unique_ptr<Document> parse(std::string text);

    Value root() const { return Value(this, root_); }

private:
    friend class Value;
    friend class Value::ObjectRange::iterator;
    struct Node {
        Kind kind;
        uint32_t first;
        uint32_t count;
        std::string_view text;
    };
    struct MemberEntry {
        std::string_view name;
        uint32_t value;
    };
    class Parser;

    Document() = default;
    Document(const Document&) = delete;
    Document& operator=(const Document&) = delete;

    std::string text_;
    std::vector<Node> nodes_;
    std::vector<MemberEntry> members_;
    std::vector<uint32_t> items_;
    std::deque<std::string> unescaped_;
    uint32_t root_ = 0;
};

}
