#include "logger.h"
#include "terminal_io.h"
#include <iostream>
#include <ctime>
#include <cstdlib>

// Helper to detect MPI rank
static int get_mpi_rank() {
    const char* rank_env = getenv("OMPI_COMM_WORLD_RANK");  // Open MPI
    if (!rank_env) rank_env = getenv("PMI_RANK");          // Intel MPI
    if (!rank_env) rank_env = getenv("SLURM_PROCID");      // SLURM
    if (!rank_env) return 0;  // Default to rank 0 if no MPI
    return std::atoi(rank_env);
}

Logger::Logger()
    : min_log_level_(LogLevel::INFO)
    , console_output_enabled_(true)
    , file_output_enabled_(false)
    , log_file_(nullptr)
    , mpi_rank_(get_mpi_rank()) {
}

Logger::~Logger() {
    is_destructing_ = true;
    if (log_file_ && log_file_->is_open()) {
        log_file_->close();
    }
}

void Logger::set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    min_log_level_ = level;
}

void Logger::set_log_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(log_mutex_);

    if (log_file_ && log_file_->is_open()) {
        log_file_->close();
    }

    log_filename_ = filename;
    log_file_ = std::make_unique<std::ofstream>(filename, std::ios::app);

    if (log_file_->is_open()) {
        file_output_enabled_ = true;
        // Write session start marker
        *log_file_ << "\n=== Shepherd Log Session Started at " << get_timestamp() << " ===\n";
        log_file_->flush();
    } else {
        std::cerr << "Failed to open log file: " << filename << std::endl;
        file_output_enabled_ = false;
    }
}

void Logger::set_console_output(bool enable) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    console_output_enabled_ = enable;
}

void Logger::set_file_output(bool enable) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    file_output_enabled_ = enable;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level >= min_log_level_) {
        write_log(level, message);
    }
}

void Logger::trace(const std::string& message) {
    log(LogLevel::TRACE, message);
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warn(const std::string& message) {
    log(LogLevel::WARN, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::fatal(const std::string& message) {
    log(LogLevel::FATAL, message);
}

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

std::string Logger::get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

std::string Logger::level_to_string(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

void Logger::write_log(LogLevel level, const std::string& message) {
    // Don't try to log if we're being destroyed (prevents mutex errors during static destruction)
    if (is_destructing_) {
        return;
    }

    // Suppress console output from non-leader MPI ranks to avoid output corruption
    if (mpi_rank_ > 0 && console_output_enabled_) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex_);

    std::string timestamp = get_timestamp();
    std::string level_str = level_to_string(level);

    // Format: [TIMESTAMP] [LEVEL] MESSAGE
    std::string formatted_message = "[" + timestamp + "] [" + level_str + "] " + message;

    // Output to console if enabled
    if (console_output_enabled_) {
        formatted_message += "\n";

        // WARN, ERROR, and FATAL go to stderr to avoid polluting stdout
        if (level >= LogLevel::WARN) {
            std::cerr << formatted_message;
        } else {
            // DEBUG and INFO go through tio for proper formatting
            Color color = (level == LogLevel::DEBUG) ? Color::GRAY : Color::DEFAULT;
            tio.write(formatted_message.c_str(), formatted_message.length(), color);
        }
    }

    // Output to file if enabled and file is open
    if (file_output_enabled_ && log_file_ && log_file_->is_open()) {
        *log_file_ << formatted_message << std::endl;
        log_file_->flush(); // Ensure immediate write for daemon use
    }
}