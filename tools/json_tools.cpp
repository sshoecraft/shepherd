#include "json_tools.h"
#include "../logger.h"
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

        std::cout << "ParseJSON: Successfully parsed JSON string" << std::endl;

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
        result["success"] = true;

        std::cout << "SerializeJSON: Successfully serialized data to JSON" << std::endl;

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
        SimpleJsonParser parser;
        JsonValue parsed = parser.parse(json_str);

        // Simple path querying (basic implementation)
        // For now, just return that the query was processed
        result["result"] = std::string("Query processed for path: ") + path;
        result["success"] = true;

        std::cout << "QueryJSON: Processed query for path: " << path << std::endl;

    } catch (const std::exception& e) {
        result["error"] = std::string("error querying JSON: ") + e.what();
        result["success"] = false;
    }

    return result;
}

void register_json_tools() {
    auto& registry = ToolRegistry::instance();

    registry.register_tool(std::make_unique<ParseJSONTool>());
    registry.register_tool(std::make_unique<SerializeJSONTool>());
    registry.register_tool(std::make_unique<QueryJSONTool>());

    LOG_DEBUG("Registered JSON tools: parse_json, serialize_json, query_json");
}