
#pragma once

#include <string>

#if 0
#include <string>
#include <vector>
#include <memory>
#include <deque>
#include <stdexcept>
#include "model_config.h"
#endif

/// @brief Represents a single message in the conversation
struct Message {
	enum Type {
		SYSTEM,
		USER,
		ASSISTANT,
		TOOL,		 // Tool response message
		FUNCTION	 // Function call message
	};

	Type type;
	std::string content;
	int tokens;

	// Optional fields for tool/function messages
	std::string tool_name;
	std::string tool_call_id;

	Message(Type t, const std::string& c, int tokens = 0) : type(t), content(c), tokens(tokens) {}

	// Convert role string to Type enum
	static Type stringToType(const std::string& role) {
		if (role == "system") return SYSTEM;
		if (role == "user") return USER;
		if (role == "assistant") return ASSISTANT;
		if (role == "tool") return TOOL;
		if (role == "function") return FUNCTION;
		return USER;  // Default fallback
	}

	// Get standardized role string for RAG storage
	// Backends will translate these to their specific format
	// (e.g., Gemini translates "assistant" -> "model")
	std::string get_role() const {
		switch (type) {
			case SYSTEM: return "system";
			case USER: return "user";
			case ASSISTANT: return "assistant";  // Standardized, not backend-specific
			case TOOL: return "tool";
			case FUNCTION: return "function";
			default: return "user";
		}
	}
};
