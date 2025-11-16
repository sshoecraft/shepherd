#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

class Logger {
public:
    Logger();
    ~Logger();

    // Configuration
    void set_log_level(LogLevel level);
    bool is_debug_enabled() const { return min_log_level_ <= LogLevel::DEBUG; }
    void set_log_file(const std::string& filename);
    void set_console_output(bool enable);
    void set_file_output(bool enable);

    // Logging methods
    void log(LogLevel level, const std::string& message);
    void trace(const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    void fatal(const std::string& message);

    // Template methods for formatted logging
    template<typename... Args>
    void log_fmt(LogLevel level, const std::string& format, Args... args);

    template<typename... Args>
    void debug_fmt(const std::string& format, Args... args);

    template<typename... Args>
    void info_fmt(const std::string& format, Args... args);

    template<typename... Args>
    void warn_fmt(const std::string& format, Args... args);

    template<typename... Args>
    void error_fmt(const std::string& format, Args... args);

    // Singleton access
    static Logger& instance();

private:
    LogLevel min_log_level_;
    bool console_output_enabled_;
    bool file_output_enabled_;
    std::unique_ptr<std::ofstream> log_file_;
    std::string log_filename_;
    mutable std::mutex log_mutex_;
    bool is_destructing_ = false;
    int mpi_rank_;  // MPI rank (0 if not using MPI)

    std::string get_timestamp() const;
    std::string level_to_string(LogLevel level) const;
    void write_log(LogLevel level, const std::string& message);

    // Simple string formatting helper
    template<typename T>
    std::string format_helper(const std::string& format, T&& value) const;

    template<typename T, typename... Args>
    std::string format_helper(const std::string& format, T&& value, Args&&... args) const;
};

// Template implementations
template<typename... Args>
void Logger::log_fmt(LogLevel level, const std::string& format, Args... args) {
    if (level >= min_log_level_) {
        std::string formatted_message = format_helper(format, args...);
        write_log(level, formatted_message);
    }
}

template<typename... Args>
void Logger::debug_fmt(const std::string& format, Args... args) {
    log_fmt(LogLevel::DEBUG, format, args...);
}

template<typename... Args>
void Logger::info_fmt(const std::string& format, Args... args) {
    log_fmt(LogLevel::INFO, format, args...);
}

template<typename... Args>
void Logger::warn_fmt(const std::string& format, Args... args) {
    log_fmt(LogLevel::WARN, format, args...);
}

template<typename... Args>
void Logger::error_fmt(const std::string& format, Args... args) {
    log_fmt(LogLevel::ERROR, format, args...);
}

// Fixed format_helper implementation
template<typename T>
std::string Logger::format_helper(const std::string& format, T&& value) const {
    size_t pos = format.find("{}", 0);
    if (pos != std::string::npos) {
        std::ostringstream oss;
        oss << value;
        std::string result = format;
        result.replace(pos, 2, oss.str());
        return result;
    }
    return format;
}

// Fixed format_helper implementation for variadic template
template<typename T, typename... Args>
std::string Logger::format_helper(const std::string& format, T&& value, Args&&... args) const {
    size_t pos = format.find("{}", 0);
    if (pos != std::string::npos) {
        std::ostringstream oss;
        oss << value;
        std::string partial = format;
        partial.replace(pos, 2, oss.str());
        return format_helper(partial, args...);
    }
    return format;
}

// Convenience macros for global logger access
#define LOG_TRACE(msg) Logger::instance().trace(msg)
#define LOG_DEBUG(msg) Logger::instance().debug(msg)
#define LOG_INFO(msg) Logger::instance().info(msg)
#define LOG_WARN(msg) Logger::instance().warn(msg)
#define LOG_ERROR(msg) Logger::instance().error(msg)
#define LOG_FATAL(msg) Logger::instance().fatal(msg)

#define LOG_DEBUG_FMT(fmt, ...) Logger::instance().debug_fmt(fmt, __VA_ARGS__)
#define LOG_INFO_FMT(fmt, ...) Logger::instance().info_fmt(fmt, __VA_ARGS__)
#define LOG_WARN_FMT(fmt, ...) Logger::instance().warn_fmt(fmt, __VA_ARGS__)
#define LOG_ERROR_FMT(fmt, ...) Logger::instance().error_fmt(fmt, __VA_ARGS__)
