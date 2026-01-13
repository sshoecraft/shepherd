#include "auth.h"
#include "config.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

using json = nlohmann::json;

// =============================================================================
// KeyStore factory
// =============================================================================

std::unique_ptr<KeyStore> KeyStore::create(const std::string& mode) {
    if (mode == "none" || mode.empty()) {
        return std::make_unique<NoneKeyStore>();
    }
    if (mode == "json") {
        return std::make_unique<JsonKeyStore>();
    }
    // Future: sqlite, vault, managed
    throw std::runtime_error("Unknown auth mode: " + mode + " (valid: none, json)");
}

// =============================================================================
// JsonKeyStore implementation
// =============================================================================

JsonKeyStore::JsonKeyStore() {
    keys = load_keys();
}

std::string JsonKeyStore::get_keys_file_path() {
    // Use XDG base directory (same pattern as config.cpp)
    std::string config_home;
    const char* xdg_config = getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] != '\0') {
        config_home = xdg_config;
    } else {
        config_home = Config::get_home_directory() + "/.config";
    }
    return config_home + "/shepherd/api_keys.json";
}

std::map<std::string, ApiKeyEntry> JsonKeyStore::load_keys() {
    std::map<std::string, ApiKeyEntry> result;
    std::string path = get_keys_file_path();

    if (!std::filesystem::exists(path)) {
        return result;  // No keys file = empty map
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return result;
    }

    try {
        json j;
        file >> j;

        for (auto& [key, value] : j.items()) {
            ApiKeyEntry entry;
            entry.name = value.value("name", "");
            entry.notes = value.value("notes", "");
            entry.created = value.value("created", "");
            entry.permissions = value.value("permissions", json::object());
            result[key] = entry;
        }
    } catch (const json::exception&) {
        throw std::runtime_error("Failed to parse api_keys.json");
    }

    return result;
}

void JsonKeyStore::save_keys(const std::map<std::string, ApiKeyEntry>& keys) {
    std::string path = get_keys_file_path();

    // Ensure directory exists
    std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    // Build JSON
    json j;
    for (const auto& [key, entry] : keys) {
        j[key] = {
            {"name", entry.name},
            {"notes", entry.notes},
            {"created", entry.created},
            {"permissions", entry.permissions}
        };
    }

    // Write to file
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to write api_keys.json");
    }
    file << j.dump(4) << std::endl;
    file.close();

    // Set file permissions to 0600 (owner read/write only)
    chmod(path.c_str(), S_IRUSR | S_IWUSR);
}

std::string JsonKeyStore::generate_key() {
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";

    std::string key = "sk-";

    // Use /dev/urandom for cryptographic randomness
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) {
        throw std::runtime_error("Failed to open /dev/urandom");
    }

    for (int i = 0; i < 32; i++) {
        unsigned char byte;
        urandom.read(reinterpret_cast<char*>(&byte), 1);
        key += charset[byte % (sizeof(charset) - 1)];
    }

    return key;
}

bool JsonKeyStore::validate_key(const std::string& key) {
    if (key.empty() || keys.empty()) {
        return false;
    }

    // Constant-time comparison to prevent timing attacks
    // Check against all keys, accumulating result
    bool found = false;
    for (const auto& [stored_key, entry] : keys) {
        if (stored_key.length() != key.length()) {
            continue;
        }

        // Constant-time comparison
        int result = 0;
        for (size_t i = 0; i < stored_key.length(); i++) {
            result |= stored_key[i] ^ key[i];
        }

        if (result == 0) {
            found = true;
            // Don't break early - continue checking to maintain constant time
        }
    }

    return found;
}

bool JsonKeyStore::is_enabled() {
    return !keys.empty();
}

const ApiKeyEntry* JsonKeyStore::get_entry(const std::string& key) const {
    auto it = keys.find(key);
    return (it != keys.end()) ? &it->second : nullptr;
}

// =============================================================================
// CLI handler: shepherd apikey
// =============================================================================

static std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::gmtime(&time);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

static std::string mask_key(const std::string& key) {
    // Show first 7 chars (sk-XXXX) and last 4 chars
    if (key.length() < 12) return key;
    return key.substr(0, 7) + "..." + key.substr(key.length() - 4);
}

int handle_apikey_args(const std::vector<std::string>& args,
                       std::function<void(const std::string&)> callback) {
    // No args or help
    if (args.empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h") {
        callback("Usage: shepherd apikey <command> [options]\n");
        callback("\nCommands:\n");
        callback("  create <name> [--notes <notes>] [--perms <json>]\n");
        callback("  list             List all keys (masked)\n");
        callback("  show <name>      Show full details of a key\n");
        callback("  set <name> [--notes <notes>] [--perms <json>]\n");
        callback("  remove <name>    Remove key by name\n");
        callback("\nExamples:\n");
        callback("  shepherd apikey create production\n");
        callback("  shepherd apikey create internal --perms '{\"server_tools\":true}'\n");
        callback("  shepherd apikey list\n");
        callback("  shepherd apikey show internal\n");
        callback("  shepherd apikey set internal --perms '{\"server_tools\":false}'\n");
        callback("  shepherd apikey remove production\n");
        return 0;
    }

    std::string subcmd = args[0];

    // List keys
    if (subcmd == "list") {
        auto keys = JsonKeyStore::load_keys();
        if (keys.empty()) {
            callback("No API keys configured.\n");
            callback("Create one with: shepherd apikey create --name <name>\n");
            return 0;
        }

        callback("NAME           KEY               CREATED      NOTES\n");
        for (const auto& [key, entry] : keys) {
            std::string line = entry.name;
            // Pad name to 14 chars
            while (line.length() < 14) line += " ";
            line += " " + mask_key(key);
            // Pad key to 17 chars
            while (line.length() < 32) line += " ";
            // Extract date from ISO timestamp
            std::string date = entry.created.length() >= 10 ? entry.created.substr(0, 10) : entry.created;
            line += " " + date;
            while (line.length() < 45) line += " ";
            line += " " + entry.notes + "\n";
            callback(line);
        }
        return 0;
    }

    // Show key details
    if (subcmd == "show") {
        if (args.size() < 2) {
            callback("Error: Missing key name\n");
            callback("Usage: shepherd apikey show <name>\n");
            return 1;
        }

        std::string name = args[1];
        auto keys = JsonKeyStore::load_keys();

        // Find key by name
        for (const auto& [key, entry] : keys) {
            if (entry.name == name) {
                callback("Name:        " + entry.name + "\n");
                callback("Key:         " + mask_key(key) + "\n");
                callback("Created:     " + entry.created + "\n");
                callback("Notes:       " + (entry.notes.empty() ? "(none)" : entry.notes) + "\n");
                callback("Permissions: " + entry.permissions.dump() + "\n");
                return 0;
            }
        }

        callback("Error: Key '" + name + "' not found\n");
        return 1;
    }

    // Set key (update notes/perms)
    if (subcmd == "set") {
        if (args.size() < 2) {
            callback("Error: Missing key name\n");
            callback("Usage: shepherd apikey set <name> [--notes <notes>] [--perms <json>]\n");
            return 1;
        }

        std::string name = args[1];
        std::string notes;
        json permissions;
        bool has_notes = false;
        bool has_perms = false;

        // Parse optional arguments
        for (size_t i = 2; i < args.size(); i++) {
            if (args[i] == "--notes" && i + 1 < args.size()) {
                notes = args[++i];
                has_notes = true;
            } else if (args[i] == "--perms" && i + 1 < args.size()) {
                try {
                    permissions = json::parse(args[++i]);
                    has_perms = true;
                } catch (const json::exception& e) {
                    callback("Error: Invalid JSON for --perms: " + std::string(e.what()) + "\n");
                    return 1;
                }
            }
        }

        if (!has_notes && !has_perms) {
            callback("Error: Specify --notes or --perms to update\n");
            return 1;
        }

        auto keys = JsonKeyStore::load_keys();

        // Find and update key
        for (auto& [key, entry] : keys) {
            if (entry.name == name) {
                if (has_notes) entry.notes = notes;
                if (has_perms) entry.permissions = permissions;
                JsonKeyStore::save_keys(keys);
                callback("Key '" + name + "' updated.\n");
                return 0;
            }
        }

        callback("Error: Key '" + name + "' not found\n");
        return 1;
    }

    // Remove key
    if (subcmd == "remove") {
        if (args.size() < 2) {
            callback("Error: Missing key name\n");
            callback("Usage: shepherd apikey remove <name>\n");
            return 1;
        }

        std::string name = args[1];
        auto keys = JsonKeyStore::load_keys();

        // Find key by name
        std::string key_to_remove;
        for (const auto& [key, entry] : keys) {
            if (entry.name == name) {
                key_to_remove = key;
                break;
            }
        }

        if (key_to_remove.empty()) {
            callback("Error: Key '" + name + "' not found\n");
            return 1;
        }

        keys.erase(key_to_remove);
        JsonKeyStore::save_keys(keys);
        callback("Key '" + name + "' removed.\n");
        return 0;
    }

    // Create new key
    if (subcmd == "create") {
        if (args.size() < 2) {
            callback("Error: Missing key name\n");
            callback("Usage: shepherd apikey create <name> [--notes <notes>] [--perms <json>]\n");
            return 1;
        }

        std::string name = args[1];
        std::string notes;
        json permissions = json::object();

        // Parse optional arguments
        for (size_t i = 2; i < args.size(); i++) {
            if (args[i] == "--notes" && i + 1 < args.size()) {
                notes = args[++i];
            } else if (args[i] == "--perms" && i + 1 < args.size()) {
                try {
                    permissions = json::parse(args[++i]);
                } catch (const json::exception& e) {
                    callback("Error: Invalid JSON for --perms: " + std::string(e.what()) + "\n");
                    return 1;
                }
            }
        }

        // Check for duplicate name
        auto keys = JsonKeyStore::load_keys();
        for (const auto& [key, entry] : keys) {
            if (entry.name == name) {
                callback("Error: Key with name '" + name + "' already exists\n");
                return 1;
            }
        }

        // Generate key
        std::string new_key = JsonKeyStore::generate_key();

        // Create entry
        ApiKeyEntry entry;
        entry.name = name;
        entry.notes = notes;
        entry.created = get_iso_timestamp();
        entry.permissions = permissions;

        // Save
        keys[new_key] = entry;
        JsonKeyStore::save_keys(keys);

        callback("API key created successfully.\n\n");
        callback("Key: " + new_key + "\n\n");
        callback("Keys are stored in: " + JsonKeyStore::get_keys_file_path() + "\n");
        return 0;
    }

    callback("Error: Unknown command '" + subcmd + "'\n");
    callback("Use 'shepherd apikey help' for usage.\n");
    return 1;
}
