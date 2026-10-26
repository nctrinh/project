// Minimal JSON value + parser + writer (no third-party dependency on the board).
// Used for data_manifest.json, IR_label.json and the output files.
#ifndef ANTIUAV_JSON_H
#define ANTIUAV_JSON_H

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace antiuav {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Type::Bool), bool_(b) {}
    Json(int v) : type_(Type::Number), num_(v) {}
    Json(long v) : type_(Type::Number), num_(static_cast<double>(v)) {}
    Json(long long v) : type_(Type::Number), num_(static_cast<double>(v)) {}
    Json(size_t v) : type_(Type::Number), num_(static_cast<double>(v)) {}
    Json(float v) : type_(Type::Number), num_(v) {}
    Json(double v) : type_(Type::Number), num_(v) {}
    Json(const char* s) : type_(Type::String), str_(s) {}
    Json(std::string s) : type_(Type::String), str_(std::move(s)) {}

    static Json array() { Json j; j.type_ = Type::Array; return j; }
    static Json object() { Json j; j.type_ = Type::Object; return j; }

    static Json parse(const std::string& text);
    static Json parse_file(const std::string& path);

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }

    double as_number() const;
    int as_int() const { return static_cast<int>(as_number()); }
    bool as_bool() const;
    const std::string& as_string() const;

    // Array access
    size_t size() const;
    const Json& operator[](size_t i) const;
    void push_back(Json v);

    // Object access (insertion order is kept for writing)
    bool contains(const std::string& key) const;
    const Json& at(const std::string& key) const;
    Json& operator[](const std::string& key);  // inserts if missing
    const std::vector<std::pair<std::string, Json>>& items() const { return obj_; }

    // Non-finite numbers are written as null (like the Python -inf -> None snapshot).
    std::string dump(int indent = -1) const;
    void dump_file(const std::string& path, int indent = 2) const;

private:
    void dump_impl(std::string& out, int indent, int level) const;

    Type type_ = Type::Null;
    bool bool_ = false;
    double num_ = 0.0;
    std::string str_;
    std::vector<Json> arr_;
    std::vector<std::pair<std::string, Json>> obj_;

    friend class JsonParser;
};

}  // namespace antiuav

#endif  // ANTIUAV_JSON_H
