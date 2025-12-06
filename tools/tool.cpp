#include "tool.h"
#include <stdexcept>
#include <algorithm>

namespace tool_utils {
    std::string get_string(const std::map<std::string, std::any>& args, const std::string& key, const std::string& default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<std::string>(it->second);
            } catch (const std::bad_any_cast&) {
                // Try to convert from const char*
                try {
                    return std::string(std::any_cast<const char*>(it->second));
                } catch (const std::bad_any_cast&) {
                    return default_value;
                }
            }
        }
        return default_value;
    }

    int get_int(const std::map<std::string, std::any>& args, const std::string& key, int default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<int>(it->second);
            } catch (const std::bad_any_cast&) {
                try {
                    return static_cast<int>(std::any_cast<double>(it->second));
                } catch (const std::bad_any_cast&) {
                    return default_value;
                }
            }
        }
        return default_value;
    }

    double get_double(const std::map<std::string, std::any>& args, const std::string& key, double default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<double>(it->second);
            } catch (const std::bad_any_cast&) {
                try {
                    return static_cast<double>(std::any_cast<int>(it->second));
                } catch (const std::bad_any_cast&) {
                    return default_value;
                }
            }
        }
        return default_value;
    }

    bool get_bool(const std::map<std::string, std::any>& args, const std::string& key, bool default_value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                return std::any_cast<bool>(it->second);
            } catch (const std::bad_any_cast&) {
                return default_value;
            }
        }
        return default_value;
    }
}
