#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>

namespace test_helpers {

// Get a unique temp file path
inline std::string temp_file_path(const std::string& prefix = "test_") {
    return "/tmp/" + prefix + std::to_string(getpid()) + "_" +
           std::to_string(rand());
}

// Read entire file contents
inline std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Write string to file
inline bool write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    file << content;
    return file.good();
}

// Delete file if exists
inline void remove_file(const std::string& path) {
    std::remove(path.c_str());
}

// Check if file exists
inline bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

// Set environment variable (RAII wrapper)
class ScopedEnv {
public:
    ScopedEnv(const std::string& name, const std::string& value)
        : name_(name) {
        const char* old = getenv(name.c_str());
        if (old) {
            old_value_ = old;
            had_value_ = true;
        }
        setenv(name.c_str(), value.c_str(), 1);
    }

    ~ScopedEnv() {
        if (had_value_) {
            setenv(name_.c_str(), old_value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::string old_value_;
    bool had_value_ = false;
};

} // namespace test_helpers

#endif // TEST_HELPERS_H
