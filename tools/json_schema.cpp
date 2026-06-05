#include "json_schema.h"

namespace tool_schema {

nlohmann::json param_to_schema(const ParameterDef& p) {
    nlohmann::json s;
    s["type"] = p.type.empty() ? "string" : p.type;
    if (!p.description.empty()) {
        s["description"] = p.description;
    }

    if (p.type == "array") {
        nlohmann::json items;
        items["type"] = p.array_item_type.empty() ? "string" : p.array_item_type;
        s["items"] = items;
    } else if (p.type == "object" && !p.object_properties.empty()) {
        s["properties"] = nlohmann::json::object();
        nlohmann::json required_arr = nlohmann::json::array();
        for (const auto& child : p.object_properties) {
            s["properties"][child.name] = param_to_schema(child);
            if (child.required) {
                required_arr.push_back(child.name);
            }
        }
        if (!required_arr.empty()) {
            s["required"] = required_arr;
        }
    }

    if (!p.default_value.empty()) {
        s["default"] = p.default_value;
    }

    return s;
}

nlohmann::json params_to_object_schema(const std::vector<ParameterDef>& params) {
    nlohmann::json schema;
    schema["type"] = "object";
    schema["properties"] = nlohmann::json::object();
    nlohmann::json required_arr = nlohmann::json::array();

    for (const auto& p : params) {
        schema["properties"][p.name] = param_to_schema(p);
        if (p.required) {
            required_arr.push_back(p.name);
        }
    }
    if (!required_arr.empty()) {
        schema["required"] = required_arr;
    }
    return schema;
}

}
