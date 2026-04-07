
#pragma once

#include <string>
#include <iostream>
#include "nlohmann/json.hpp"

/// @brief Represents a single message in the conversation
/// Stored in session.messages and sent to API providers
struct Message {
	/// Message roles that are actually stored in session and sent to APIs
	enum Role {
		SYSTEM,         // System prompt
		USER,           // User messages
		ASSISTANT,      // Provider responses
		TOOL_RESPONSE,  // Result of tool execution (modern "tool" role)
		FUNCTION        // Legacy OpenAI function result (role: "function")
	};

	Role role;
	std::string content;
	int tokens;

	// Optional fields for tool/function messages
	std::string tool_name;
	std::string tool_call_id;

	// For assistant messages that make tool calls (OpenAI format)
	// Or for storing structured content (Gemini parts array, etc.)
	std::string tool_calls_json;

	// Original content before RAG injection (empty = no injection applied)
	// Used to restore message content when stripping injection from previous turns
	std::string pre_injection_content;

	Message(Role r, const std::string& c, int tokens = 0) : role(r), content(c), tokens(tokens) {}

	// Helper: is this a tool call response (result of executing a tool)?
	bool is_tool_response() const {
		return role == TOOL_RESPONSE || role == FUNCTION;
	}

	// Convert role string to Role enum
	static Role stringToRole(const std::string& roleStr) {
		if (roleStr == "system") return SYSTEM;
		if (roleStr == "user") return USER;
		if (roleStr == "assistant") return ASSISTANT;
		if (roleStr == "tool") return TOOL_RESPONSE;
		if (roleStr == "function") return FUNCTION;
		return USER;  // Default fallback
	}

	// Get standardized role string for RAG storage
	// Backends will translate these to their specific format
	std::string get_role() const {
		switch (role) {
			case SYSTEM: return "system";
			case USER: return "user";
			case ASSISTANT: return "assistant";
			case TOOL_RESPONSE: return "tool";
			case FUNCTION: return "function";
			default: return "user";
		}
	}
};

inline std::ostream& operator<<(std::ostream& os, const Message& msg) {
	os << msg.get_role() << " (" << msg.tokens << " tokens): ";
	if (msg.content.length() > 100) {
		os << msg.content.substr(0, 100) << "...";
	} else {
		os << msg.content;
	}
	if (!msg.tool_name.empty()) {
		os << " [tool: " << msg.tool_name << "]";
	}
	if (!msg.tool_call_id.empty()) {
		os << " [tool_call_id: " << msg.tool_call_id << "]";
	}
	return os;
}

// JSON serialization for session persistence (nlohmann ADL pattern)
inline void to_json(nlohmann::json& j, const Message& m) {
	j = nlohmann::json{
		{"role", m.get_role()},
		{"content", m.content},
		{"tokens", m.tokens}
	};
	if (!m.tool_name.empty()) j["tool_name"] = m.tool_name;
	if (!m.tool_call_id.empty()) j["tool_call_id"] = m.tool_call_id;
	if (!m.tool_calls_json.empty()) j["tool_calls_json"] = m.tool_calls_json;
	if (!m.pre_injection_content.empty()) j["pre_injection_content"] = m.pre_injection_content;
}

inline void from_json(const nlohmann::json& j, Message& m) {
	m.role = Message::stringToRole(j.at("role").get<std::string>());
	m.content = j.at("content").get<std::string>();
	m.tokens = j.value("tokens", 0);
	m.tool_name = j.value("tool_name", "");
	m.tool_call_id = j.value("tool_call_id", "");
	m.tool_calls_json = j.value("tool_calls_json", "");
	m.pre_injection_content = j.value("pre_injection_content", "");
}
