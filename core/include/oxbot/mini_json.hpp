#pragma once

// A deliberately small JSON value/parser used by the offline build and by the
// BotZone amalgamated source.  It accepts the JSON types used by the BotZone
// protocol and does not depend on libstdc++ extensions or a third-party
// library.  It is not intended to be a general-purpose JSON implementation.

#include <cctype>
#include <cmath>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace oxbot {
namespace json {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    Value() = default;
    explicit Value(std::nullptr_t) : type(Type::Null) {}
    explicit Value(bool v) : type(Type::Bool), boolean(v) {}
    explicit Value(int v) : type(Type::Number), number(static_cast<double>(v)) {}
    explicit Value(double v) : type(Type::Number), number(v) {}
    explicit Value(const char* v) : type(Type::String), string(v ? v : "") {}
    explicit Value(std::string v) : type(Type::String), string(std::move(v)) {}

    static Value make_array() {
        Value v;
        v.type = Type::Array;
        return v;
    }
    static Value make_object() {
        Value v;
        v.type = Type::Object;
        return v;
    }

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool as_bool(bool fallback = false) const {
        return is_bool() ? boolean : fallback;
    }
    int as_int(int fallback = 0) const {
        return is_number() && std::isfinite(number) && std::floor(number) == number &&
               number >= std::numeric_limits<int>::min() && number <= std::numeric_limits<int>::max()
            ? static_cast<int>(number) : fallback;
    }
    std::string as_string(const std::string& fallback = {}) const {
        return is_string() ? string : fallback;
    }

    bool contains(const std::string& key) const {
        return is_object() && object.find(key) != object.end();
    }
    const Value* find(const std::string& key) const {
        if (!is_object()) return nullptr;
        const auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }
    Value* find(const std::string& key) {
        if (!is_object()) return nullptr;
        const auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }
    const Value& operator[](const std::string& key) const {
        static const Value null_value;
        const Value* v = find(key);
        return v ? *v : null_value;
    }
    Value& operator[](const std::string& key) {
        if (!is_object()) {
            type = Type::Object;
            array.clear();
            string.clear();
        }
        return object[key];
    }
    void push_back(Value v) {
        if (!is_array()) {
            type = Type::Array;
            object.clear();
            string.clear();
        }
        array.push_back(std::move(v));
    }
};

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

class Parser {
public:
    explicit Parser(const std::string& input) : input_(input) {}

    Value parse_document() {
        if (input_.size() > 8 * 1024 * 1024) fail("input too large");
        skip_space();
        Value v = parse_value();
        skip_space();
        if (pos_ != input_.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string& input_;
    std::size_t pos_ = 0;
    unsigned depth_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw ParseError(what + " at offset " + std::to_string(pos_));
    }
    void skip_space() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) ++pos_;
    }
    char peek() const { return pos_ < input_.size() ? input_[pos_] : '\0'; }
    char take() {
        if (pos_ >= input_.size()) fail("unexpected end");
        return input_[pos_++];
    }
    void expect(char c) {
        if (take() != c) fail(std::string("expected '") + c + "'");
    }

    Value parse_value() {
        struct DepthGuard {
            unsigned& depth;
            explicit DepthGuard(unsigned& d) : depth(d) { ++depth; }
            ~DepthGuard() { --depth; }
        } guard(depth_);
        if (depth_ > 128) fail("nesting too deep");
        skip_space();
        switch (peek()) {
            case 'n': parse_literal("null"); return Value(nullptr);
            case 't': parse_literal("true"); return Value(true);
            case 'f': parse_literal("false"); return Value(false);
            case '"': return Value(parse_string());
            case '[': return parse_array();
            case '{': return parse_object();
            default:
                if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek()))) {
                    return Value(parse_number());
                }
                fail("invalid value");
        }
    }
    void parse_literal(const char* literal) {
        while (*literal) {
            if (take() != *literal++) fail("invalid literal");
        }
    }
    unsigned hex4() {
        unsigned code = 0;
        for (int i = 0; i < 4; ++i) {
            const char h = take();
            code <<= 4;
            if (h >= '0' && h <= '9') code += static_cast<unsigned>(h - '0');
            else if (h >= 'a' && h <= 'f') code += static_cast<unsigned>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') code += static_cast<unsigned>(h - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return code;
    }
    static void append_utf8(std::string& out, unsigned code) {
        if (code <= 0x7f) out.push_back(static_cast<char>(code));
        else if (code <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 63)));
        } else if (code <= 0xffff) {
            out.push_back(static_cast<char>(0xe0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 63)));
            out.push_back(static_cast<char>(0x80 | (code & 63)));
        } else {
            out.push_back(static_cast<char>(0xf0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 63)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 63)));
            out.push_back(static_cast<char>(0x80 | (code & 63)));
        }
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < input_.size()) {
            const char c = take();
            if (c == '"') return out;
            if (c == '\\') {
                const char e = take();
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        unsigned code = hex4();
                        if (code >= 0xd800 && code <= 0xdbff) {
                            if (take() != '\\' || take() != 'u') fail("missing low surrogate");
                            const unsigned low = hex4();
                            if (low < 0xdc00 || low > 0xdfff) fail("invalid low surrogate");
                            code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                        } else if (code >= 0xdc00 && code <= 0xdfff) fail("lone low surrogate");
                        append_utf8(out, code);
                        break;
                    }
                    default: fail("invalid string escape");
                }
            } else {
                if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }
    double parse_number() {
        const std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (peek() == '0') ++pos_;
        else {
            if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid number");
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (peek() == '.') {
            ++pos_;
            if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid fraction");
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!std::isdigit(static_cast<unsigned char>(peek()))) fail("invalid exponent");
            while (std::isdigit(static_cast<unsigned char>(peek()))) ++pos_;
        }
        try {
            const double result = std::stod(input_.substr(start, pos_ - start));
            if (!std::isfinite(result)) fail("nonfinite number");
            return result;
        } catch (...) {
            fail("invalid number");
        }
    }
    Value parse_array() {
        expect('[');
        Value out = Value::make_array();
        skip_space();
        if (peek() == ']') { ++pos_; return out; }
        while (true) {
            out.array.push_back(parse_value());
            skip_space();
            const char c = take();
            if (c == ']') return out;
            if (c != ',') fail("expected ',' or ']'");
            skip_space();
        }
    }
    Value parse_object() {
        expect('{');
        Value out = Value::make_object();
        skip_space();
        if (peek() == '}') { ++pos_; return out; }
        while (true) {
            skip_space();
            if (peek() != '"') fail("object key must be a string");
            const std::string key = parse_string();
            skip_space();
            expect(':');
            if (out.object.count(key)) fail("duplicate key");
            out.object[key] = parse_value();
            skip_space();
            const char c = take();
            if (c == '}') return out;
            if (c != ',') fail("expected ',' or '}'");
            skip_space();
        }
    }
};

inline Value parse(const std::string& input) { return Parser(input).parse_document(); }

inline std::string escape(const std::string& input) {
    std::ostringstream out;
    for (const unsigned char c : input) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char* hex = "0123456789abcdef";
                    out << "\\u00" << hex[c >> 4] << hex[c & 15];
                } else out << static_cast<char>(c);
        }
    }
    return out.str();
}

inline std::string dump(const Value& v) {
    switch (v.type) {
        case Value::Type::Null: return "null";
        case Value::Type::Bool: return v.boolean ? "true" : "false";
        case Value::Type::Number: {
            if (!std::isfinite(v.number)) throw std::runtime_error("cannot encode nonfinite number");
            if (std::floor(v.number) == v.number && v.number >= -9007199254740991.0 && v.number <= 9007199254740991.0) {
                return std::to_string(static_cast<long long>(v.number));
            }
            std::ostringstream out;
            out.precision(17);
            out << v.number;
            return out.str();
        }
        case Value::Type::String: return "\"" + escape(v.string) + "\"";
        case Value::Type::Array: {
            std::string out = "[";
            for (std::size_t i = 0; i < v.array.size(); ++i) {
                if (i) out += ',';
                out += dump(v.array[i]);
            }
            return out + ']';
        }
        case Value::Type::Object: {
            std::string out = "{";
            bool first = true;
            for (const auto& item : v.object) {
                if (!first) out += ',';
                first = false;
                out += "\"" + escape(item.first) + "\":" + dump(item.second);
            }
            return out + '}';
        }
    }
    return "null";
}

}  // namespace json
}  // namespace oxbot
