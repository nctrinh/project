#include "json.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace antiuav {

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    Json parse_document() {
        Json v = parse_value();
        skip_ws();
        if (pos_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + msg);
    }

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) ++pos_;
    }

    bool consume(const char* lit) {
        size_t n = std::char_traits<char>::length(lit);
        if (s_.compare(pos_, n, lit) == 0) { pos_ += n; return true; }
        return false;
    }

    Json parse_value() {
        skip_ws();
        if (pos_ >= s_.size()) fail("unexpected end");
        char c = s_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Json(parse_string());
        if (consume("true")) return Json(true);
        if (consume("false")) return Json(false);
        if (consume("null")) return Json();
        if (consume("NaN")) return Json(std::nan(""));
        if (consume("Infinity")) return Json(INFINITY);
        if (consume("-Infinity")) return Json(-INFINITY);
        return parse_number();
    }

    Json parse_number() {
        size_t start = pos_;
        while (pos_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '-' ||
                                    s_[pos_] == '+' || s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
        }
        if (start == pos_) fail("invalid value");
        return Json(std::stod(s_.substr(start, pos_ - start)));
    }

    std::string parse_string() {
        ++pos_;  // opening quote
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (pos_ >= s_.size()) fail("bad escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) fail("bad \\u escape");
                    unsigned cp = std::stoul(s_.substr(pos_, 4), nullptr, 16);
                    pos_ += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && s_.compare(pos_, 2, "\\u") == 0) {
                        unsigned lo = std::stoul(s_.substr(pos_ + 2, 4), nullptr, 16);
                        pos_ += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    // UTF-8 encode
                    if (cp < 0x80) out += static_cast<char>(cp);
                    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
                    else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: fail("bad escape");
            }
        }
        fail("unterminated string");
    }

    Json parse_array() {
        ++pos_;
        Json arr = Json::array();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return arr; }
        while (true) {
            arr.arr_.push_back(parse_value());
            skip_ws();
            if (pos_ >= s_.size()) fail("unterminated array");
            if (s_[pos_] == ',') { ++pos_; continue; }
            if (s_[pos_] == ']') { ++pos_; return arr; }
            fail("expected , or ]");
        }
    }

    Json parse_object() {
        ++pos_;
        Json obj = Json::object();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return obj; }
        while (true) {
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected key");
            std::string key = parse_string();
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') fail("expected :");
            ++pos_;
            obj.obj_.emplace_back(std::move(key), parse_value());
            skip_ws();
            if (pos_ >= s_.size()) fail("unterminated object");
            if (s_[pos_] == ',') { ++pos_; continue; }
            if (s_[pos_] == '}') { ++pos_; return obj; }
            fail("expected , or }");
        }
    }

    const std::string& s_;
    size_t pos_ = 0;
};

Json Json::parse(const std::string& text) { return JsonParser(text).parse_document(); }

Json Json::parse_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open JSON file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    try {
        return parse(ss.str());
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
}

double Json::as_number() const {
    if (type_ == Type::Number) return num_;
    if (type_ == Type::Bool) return bool_ ? 1.0 : 0.0;
    throw std::runtime_error("JSON value is not a number");
}

bool Json::as_bool() const {
    if (type_ == Type::Bool) return bool_;
    if (type_ == Type::Number) return num_ != 0.0;
    throw std::runtime_error("JSON value is not a bool");
}

const std::string& Json::as_string() const {
    if (type_ != Type::String) throw std::runtime_error("JSON value is not a string");
    return str_;
}

size_t Json::size() const {
    if (type_ == Type::Array) return arr_.size();
    if (type_ == Type::Object) return obj_.size();
    return 0;
}

const Json& Json::operator[](size_t i) const {
    if (type_ != Type::Array || i >= arr_.size()) throw std::runtime_error("JSON array index out of range");
    return arr_[i];
}

void Json::push_back(Json v) {
    if (type_ == Type::Null) type_ = Type::Array;
    if (type_ != Type::Array) throw std::runtime_error("JSON push_back on non-array");
    arr_.push_back(std::move(v));
}

bool Json::contains(const std::string& key) const {
    if (type_ != Type::Object) return false;
    for (const auto& kv : obj_)
        if (kv.first == key) return true;
    return false;
}

const Json& Json::at(const std::string& key) const {
    if (type_ == Type::Object)
        for (const auto& kv : obj_)
            if (kv.first == key) return kv.second;
    throw std::runtime_error("JSON key not found: " + key);
}

Json& Json::operator[](const std::string& key) {
    if (type_ == Type::Null) type_ = Type::Object;
    if (type_ != Type::Object) throw std::runtime_error("JSON operator[] on non-object");
    for (auto& kv : obj_)
        if (kv.first == key) return kv.second;
    obj_.emplace_back(key, Json());
    return obj_.back().second;
}

static void write_escaped(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);  // UTF-8 kept as is (ensure_ascii=False)
                }
        }
    }
    out += '"';
}

void Json::dump_impl(std::string& out, int indent, int level) const {
    auto newline = [&](int lvl) {
        if (indent < 0) return;
        out += '\n';
        out.append(static_cast<size_t>(indent * lvl), ' ');
    };
    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += bool_ ? "true" : "false"; break;
        case Type::Number: {
            if (!std::isfinite(num_)) { out += "null"; break; }
            if (num_ == std::floor(num_) && std::fabs(num_) < 1e15) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(num_));
                out += buf;
            } else {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.17g", num_);
                out += buf;
            }
            break;
        }
        case Type::String: write_escaped(out, str_); break;
        case Type::Array: {
            out += '[';
            for (size_t i = 0; i < arr_.size(); ++i) {
                if (i) out += indent < 0 ? ", " : ",";
                newline(level + 1);
                arr_[i].dump_impl(out, indent, level + 1);
            }
            if (!arr_.empty()) newline(level);
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            for (size_t i = 0; i < obj_.size(); ++i) {
                if (i) out += indent < 0 ? ", " : ",";
                newline(level + 1);
                write_escaped(out, obj_[i].first);
                out += ": ";
                obj_[i].second.dump_impl(out, indent, level + 1);
            }
            if (!obj_.empty()) newline(level);
            out += '}';
            break;
        }
    }
}

std::string Json::dump(int indent) const {
    std::string out;
    dump_impl(out, indent, 0);
    return out;
}

void Json::dump_file(const std::string& path, int indent) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write JSON file: " + path);
    f << dump(indent) << '\n';
}

}  // namespace antiuav
