#include "shepherd.h"
#include "json_tools.h"
#include "tools.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <regex>

// Simple JSON value structure for basic parsing
struct JsonValue {
    enum Type { STRING, NUMBER, BOOLEAN, OBJECT, ARRAY, NULL_VALUE };
    Type type;
    std::string string_value;
    double number_value;
    bool bool_value;
    std::map<std::string, JsonValue> object_value;
    std::vector<JsonValue> array_value;

    JsonValue() : type(NULL_VALUE) {}
    explicit JsonValue(const std::string& s) : type(STRING), string_value(s) {}
    explicit JsonValue(double d) : type(NUMBER), number_value(d) {}
    explicit JsonValue(bool b) : type(BOOLEAN), bool_value(b) {}
};

// Simple JSON parser (basic implementation)
class SimpleJsonParser {
private:
    std::string json;
    size_t pos;

    void skip_whitespace() {
        while (pos < json.size() && std::isspace(json[pos])) {
            pos++;
        }
    }

    std::string parse_string() {
        if (json[pos] != '"') throw std::runtime_error("Expected '\"'");
        pos++; // skip opening quote

        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++; // skip backslash
                switch (json[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case '\\': result += '\\'; break;
                    case '"': result += '"'; break;
                    default: result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }

        if (pos >= json.size()) throw std::runtime_error("Unterminated string");
        pos++; // skip closing quote
        return result;
    }

    double parse_number() {
        size_t start = pos;
        if (json[pos] == '-') pos++;

        while (pos < json.size() && std::isdigit(json[pos])) pos++;

        if (pos < json.size() && json[pos] == '.') {
            pos++;
            while (pos < json.size() && std::isdigit(json[pos])) pos++;
        }

        return std::stod(json.substr(start, pos - start));
    }

    JsonValue parse_value() {
        skip_whitespace();

        if (pos >= json.size()) throw std::runtime_error("Unexpected end of input");

        switch (json[pos]) {
            case '"':
                return JsonValue(parse_string());

            case '{': {
                pos++; // skip '{'
                JsonValue obj;
                obj.type = JsonValue::OBJECT;

                skip_whitespace();
                if (pos < json.size() && json[pos] == '}') {
                    pos++;
                    return obj;
                }

                while (true) {
                    skip_whitespace();
                    std::string key = parse_string();
                    skip_whitespace();

                    if (pos >= json.size() || json[pos] != ':') {
                        throw std::runtime_error("Expected ':'");
                    }
                    pos++; // skip ':'

                    obj.object_value[key] = parse_value();

                    skip_whitespace();
                    if (pos >= json.size()) throw std::runtime_error("Expected '}' or ','");

                    if (json[pos] == '}') {
                        pos++;
                        break;
                    } else if (json[pos] == ',') {
                        pos++;
                    } else {
                        throw std::runtime_error("Expected '}' or ','");
                    }
                }
                return obj;
            }

            case '[': {
                pos++; // skip '['
                JsonValue arr;
                arr.type = JsonValue::ARRAY;

                skip_whitespace();
                if (pos < json.size() && json[pos] == ']') {
                    pos++;
                    return arr;
                }

                while (true) {
                    arr.array_value.push_back(parse_value());

                    skip_whitespace();
                    if (pos >= json.size()) throw std::runtime_error("Expected ']' or ','");

                    if (json[pos] == ']') {
                        pos++;
                        break;
                    } else if (json[pos] == ',') {
                        pos++;
                    } else {
                        throw std::runtime_error("Expected ']' or ','");
                    }
                }
                return arr;
            }

            case 't':
                if (json.substr(pos, 4) == "true") {
                    pos += 4;
                    return JsonValue(true);
                }
                throw std::runtime_error("Invalid token");

            case 'f':
                if (json.substr(pos, 5) == "false") {
                    pos += 5;
                    return JsonValue(false);
                }
                throw std::runtime_error("Invalid token");

            case 'n':
                if (json.substr(pos, 4) == "null") {
                    pos += 4;
                    return JsonValue();
                }
                throw std::runtime_error("Invalid token");

            default:
                if (std::isdigit(json[pos]) || json[pos] == '-') {
                    return JsonValue(parse_number());
                }
                throw std::runtime_error("Unexpected character");
        }
    }

public:
    JsonValue parse(const std::string& json_str) {
        json = json_str;
        pos = 0;
        return parse_value();
    }
};

// JSON serializer
std::string serialize_json_value(const JsonValue& value) {
    switch (value.type) {
        case JsonValue::STRING:
            return "\"" + value.string_value + "\"";

        case JsonValue::NUMBER:
            return std::to_string(value.number_value);

        case JsonValue::BOOLEAN:
            return value.bool_value ? "true" : "false";

        case JsonValue::NULL_VALUE:
            return "null";

        case JsonValue::OBJECT: {
            std::string result = "{";
            bool first = true;
            for (const auto& pair : value.object_value) {
                if (!first) result += ",";
                result += "\"" + pair.first + "\":" + serialize_json_value(pair.second);
                first = false;
            }
            result += "}";
            return result;
        }

        case JsonValue::ARRAY: {
            std::string result = "[";
            bool first = true;
            for (const auto& item : value.array_value) {
                if (!first) result += ",";
                result += serialize_json_value(item);
                first = false;
            }
            result += "]";
            return result;
        }
    }
    return "null";
}

std::vector<ParameterDef> ParseJSONTool::get_parameters_schema() const {
    return {
        {"json", "string", "The JSON string to parse", true, "", "", {}}
    };
}

std::map<std::string, std::any> ParseJSONTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string json_str = tool_utils::get_string(args, "json");
    if (json_str.empty()) {
        result["error"] = std::string("json string is required");
        result["success"] = false;
        return result;
    }

    try {
        SimpleJsonParser parser;
        JsonValue parsed = parser.parse(json_str);

        // Convert JsonValue to std::any representation
        result["parsed"] = std::string("JSON parsed successfully");
        result["type"] = std::string("object"); // simplified
        result["success"] = true;

        // Build summary based on type
        std::string summary;
        if (parsed.type == JsonValue::ARRAY) {
            summary = "Parsed JSON array (" + std::to_string(parsed.array_value.size()) + " items)";
        } else if (parsed.type == JsonValue::OBJECT) {
            summary = "Parsed JSON object (" + std::to_string(parsed.object_value.size()) + " keys)";
        } else {
            summary = "Parsed JSON value";
        }
        result["summary"] = summary;

        dout(1) << "ParseJSON: Successfully parsed JSON string" << std::endl;

    } catch (const std::exception& e) {
        result["error"] = std::string("error parsing JSON: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> SerializeJSONTool::get_parameters_schema() const {
    return {
        {"data", "string", "The data to serialize to JSON", true, "", "", {}}
    };
}

std::map<std::string, std::any> SerializeJSONTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    try {
        // For now, serialize the input arguments as JSON
        std::string json_result = "{";
        bool first = true;

        for (const auto& pair : args) {
            if (!first) json_result += ",";
            json_result += "\"" + pair.first + "\":";

            // Try to convert std::any to string representation
            try {
                auto str_val = std::any_cast<std::string>(pair.second);
                json_result += "\"" + str_val + "\"";
            } catch (const std::bad_any_cast&) {
                try {
                    auto int_val = std::any_cast<int>(pair.second);
                    json_result += std::to_string(int_val);
                } catch (const std::bad_any_cast&) {
                    try {
                        auto bool_val = std::any_cast<bool>(pair.second);
                        json_result += bool_val ? "true" : "false";
                    } catch (const std::bad_any_cast&) {
                        json_result += "null";
                    }
                }
            }
            first = false;
        }
        json_result += "}";

        result["json"] = json_result;
        result["content"] = json_result;
        result["summary"] = std::string("Serialized to ") + std::to_string(json_result.size()) + " bytes";
        result["success"] = true;

        dout(1) << "SerializeJSON: Successfully serialized data to JSON" << std::endl;

    } catch (const std::exception& e) {
        result["error"] = std::string("error serializing JSON: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> QueryJSONTool::get_parameters_schema() const {
    return {
        {"json", "string", "The JSON string to query", true, "", "", {}},
        {"path", "string", "JSONPath expression to query the data", true, "", "", {}}
    };
}

// Parse a filter expression like @.symbol == 'CTCLE' or @.price > 10
// Returns true if the element matches the filter
static bool evaluate_filter(const nlohmann::json& element, const std::string& filter_expr) {
    // Parse: @.field op value
    size_t pos = 0;

    // Skip @
    if (pos < filter_expr.size() && filter_expr[pos] == '@') pos++;
    // Skip .
    if (pos < filter_expr.size() && filter_expr[pos] == '.') pos++;

    // Read field name
    size_t field_start = pos;
    while (pos < filter_expr.size() && filter_expr[pos] != ' ' &&
           filter_expr[pos] != '=' && filter_expr[pos] != '!' &&
           filter_expr[pos] != '<' && filter_expr[pos] != '>') {
        pos++;
    }
    std::string field = filter_expr.substr(field_start, pos - field_start);

    // Skip whitespace
    while (pos < filter_expr.size() && filter_expr[pos] == ' ') pos++;

    // Read operator
    std::string op;
    while (pos < filter_expr.size() && filter_expr[pos] != ' ' &&
           filter_expr[pos] != '\'' && filter_expr[pos] != '"' &&
           !(filter_expr[pos] >= '0' && filter_expr[pos] <= '9') &&
           filter_expr[pos] != '-') {
        op += filter_expr[pos++];
    }

    // Skip whitespace
    while (pos < filter_expr.size() && filter_expr[pos] == ' ') pos++;

    // Read value — string (quoted) or number
    std::string value_str;
    bool is_string_value = false;
    if (pos < filter_expr.size() && (filter_expr[pos] == '\'' || filter_expr[pos] == '"')) {
        char quote = filter_expr[pos++];
        is_string_value = true;
        while (pos < filter_expr.size() && filter_expr[pos] != quote) {
            value_str += filter_expr[pos++];
        }
    } else {
        while (pos < filter_expr.size() && filter_expr[pos] != ' ' && filter_expr[pos] != ')') {
            value_str += filter_expr[pos++];
        }
    }

    // Check if element has the field
    if (!element.is_object() || !element.contains(field)) return false;

    const auto& field_val = element[field];

    if (is_string_value) {
        if (!field_val.is_string()) return false;
        std::string actual = field_val.get<std::string>();
        if (op == "==" || op == "=") return actual == value_str;
        if (op == "!=" || op == "!==") return actual != value_str;
        return false;
    }

    // Numeric comparison
    double cmp_val = 0;
    try { cmp_val = std::stod(value_str); } catch (...) { return false; }

    double actual_num = 0;
    if (field_val.is_number()) {
        actual_num = field_val.get<double>();
    } else {
        return false;
    }

    if (op == "==" || op == "=") return actual_num == cmp_val;
    if (op == "!=" || op == "!==") return actual_num != cmp_val;
    if (op == ">") return actual_num > cmp_val;
    if (op == ">=") return actual_num >= cmp_val;
    if (op == "<") return actual_num < cmp_val;
    if (op == "<=") return actual_num <= cmp_val;
    return false;
}

// Traverse JSON with a JSONPath expression
// Supports: $, .key, [n], [*], [?(@.field op value)]
static nlohmann::json jsonpath_query(const nlohmann::json& root, const std::string& path) {
    // Collect results as an array (JSONPath always returns a node set)
    std::vector<nlohmann::json> current = {root};

    size_t pos = 0;
    // Skip leading $
    if (pos < path.size() && path[pos] == '$') pos++;

    while (pos < path.size()) {
        // Skip dot separator
        if (path[pos] == '.') {
            pos++;
            if (pos >= path.size()) break;
        }

        if (path[pos] == '[') {
            pos++; // skip [
            // Filter expression?
            if (pos < path.size() && path[pos] == '?') {
                pos++; // skip ?
                // Skip (
                if (pos < path.size() && path[pos] == '(') pos++;

                // Read until matching )
                int depth = 1;
                std::string filter_expr;
                while (pos < path.size() && depth > 0) {
                    if (path[pos] == '(') depth++;
                    else if (path[pos] == ')') { depth--; if (depth == 0) { pos++; break; } }
                    filter_expr += path[pos++];
                }
                // Skip ]
                if (pos < path.size() && path[pos] == ']') pos++;

                // Apply filter to all current array elements
                std::vector<nlohmann::json> filtered;
                for (const auto& node : current) {
                    if (node.is_array()) {
                        for (const auto& elem : node) {
                            if (evaluate_filter(elem, filter_expr)) {
                                filtered.push_back(elem);
                            }
                        }
                    }
                }
                current = filtered;

            } else if (pos < path.size() && path[pos] == '*') {
                // [*] — all elements
                pos++; // skip *
                if (pos < path.size() && path[pos] == ']') pos++;

                std::vector<nlohmann::json> expanded;
                for (const auto& node : current) {
                    if (node.is_array()) {
                        for (const auto& elem : node) {
                            expanded.push_back(elem);
                        }
                    } else if (node.is_object()) {
                        for (auto& [k, v] : node.items()) {
                            expanded.push_back(v);
                        }
                    }
                }
                current = expanded;

            } else {
                // Numeric index or quoted key
                std::string index_str;
                bool is_quoted = false;
                if (pos < path.size() && (path[pos] == '\'' || path[pos] == '"')) {
                    char quote = path[pos++];
                    is_quoted = true;
                    while (pos < path.size() && path[pos] != quote) {
                        index_str += path[pos++];
                    }
                    if (pos < path.size()) pos++; // skip closing quote
                } else {
                    while (pos < path.size() && path[pos] != ']') {
                        index_str += path[pos++];
                    }
                }
                if (pos < path.size() && path[pos] == ']') pos++;

                std::vector<nlohmann::json> next;
                if (is_quoted) {
                    // Object key access
                    for (const auto& node : current) {
                        if (node.is_object() && node.contains(index_str)) {
                            next.push_back(node[index_str]);
                        }
                    }
                } else {
                    // Numeric index
                    try {
                        int idx = std::stoi(index_str);
                        for (const auto& node : current) {
                            if (node.is_array()) {
                                if (idx < 0) idx = (int)node.size() + idx;
                                if (idx >= 0 && idx < (int)node.size()) {
                                    next.push_back(node[idx]);
                                }
                            }
                        }
                    } catch (...) {
                        // Try as object key
                        for (const auto& node : current) {
                            if (node.is_object() && node.contains(index_str)) {
                                next.push_back(node[index_str]);
                            }
                        }
                    }
                }
                current = next;
            }

        } else {
            // Dot notation: read key name
            size_t key_start = pos;
            while (pos < path.size() && path[pos] != '.' && path[pos] != '[') {
                pos++;
            }
            std::string key = path.substr(key_start, pos - key_start);
            if (key.empty()) continue;

            std::vector<nlohmann::json> next;
            for (const auto& node : current) {
                if (node.is_object() && node.contains(key)) {
                    next.push_back(node[key]);
                }
            }
            current = next;
        }
    }

    // Return single value or array of results
    if (current.empty()) {
        return nlohmann::json(nullptr);
    } else if (current.size() == 1) {
        return current[0];
    } else {
        return nlohmann::json(current);
    }
}

std::map<std::string, std::any> QueryJSONTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string json_str = tool_utils::get_string(args, "json");
    std::string path = tool_utils::get_string(args, "path");

    if (json_str.empty()) {
        result["error"] = std::string("json string is required");
        result["success"] = false;
        return result;
    }

    if (path.empty()) {
        result["error"] = std::string("path is required");
        result["success"] = false;
        return result;
    }

    try {
        auto parsed = nlohmann::json::parse(json_str);
        auto query_result = jsonpath_query(parsed, path);

        std::string output = query_result.dump(2);
        result["content"] = output;
        result["success"] = true;

        // Build summary
        if (query_result.is_null()) {
            result["summary"] = std::string("No matches found");
        } else if (query_result.is_array()) {
            result["summary"] = std::string("Found ") + std::to_string(query_result.size()) + " result(s)";
        } else {
            result["summary"] = std::string("Found value at path");
        }

        dout(1) << "QueryJSON: Processed query for path: " << path << std::endl;

    } catch (const nlohmann::json::exception& e) {
        result["error"] = std::string("error querying JSON: ") + e.what();
        result["success"] = false;
    } catch (const std::exception& e) {
        result["error"] = std::string("error querying JSON: ") + e.what();
        result["success"] = false;
    }

    return result;
}

void register_json_tools(Tools& tools) {
    tools.register_tool(std::make_unique<ParseJSONTool>());
    tools.register_tool(std::make_unique<SerializeJSONTool>());
    tools.register_tool(std::make_unique<QueryJSONTool>());

    dout(1) << "Registered JSON tools: parse_json, serialize_json, query_json" << std::endl;
}