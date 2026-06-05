#pragma once

#include "tool.h"
#include "nlohmann/json.hpp"
#include <vector>

namespace tool_schema {

// Convert a single ParameterDef into a JSON Schema fragment (no name key).
// Handles primitive types, arrays (via array_item_type), and nested objects
// (via object_properties).
nlohmann::json param_to_schema(const ParameterDef& p);

// Convert a parameter list into a JSON Schema object:
//   {"type":"object", "properties":{...}, "required":[...]}
// "required" is omitted when no parameter is required.
nlohmann::json params_to_object_schema(const std::vector<ParameterDef>& params);

}
