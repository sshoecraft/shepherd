#include <gtest/gtest.h>
#include "config.h"
#include "test_helpers.h"
#include "temp_dir.h"
#include <fstream>
#include <cstdlib>

// Test fixture for Config tests
class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test configs
        temp_dir = std::make_unique<test_helpers::TempDir>("config_test_");
        ASSERT_TRUE(temp_dir->valid());
    }

    void TearDown() override {
        temp_dir.reset();
    }

    std::unique_ptr<test_helpers::TempDir> temp_dir;
};

// =============================================================================
// parse_size_string tests (via set_max_db_size)
// =============================================================================

TEST_F(ConfigTest, ParseSizeStringPlainNumber) {
    Config cfg;
    cfg.set_max_db_size("1024");
    EXPECT_EQ(cfg.max_db_size, 1024u);
}

TEST_F(ConfigTest, ParseSizeStringKilobytes) {
    Config cfg;

    cfg.set_max_db_size("10K");
    EXPECT_EQ(cfg.max_db_size, 10u * 1024);

    cfg.set_max_db_size("10KB");
    EXPECT_EQ(cfg.max_db_size, 10u * 1024);
}

TEST_F(ConfigTest, ParseSizeStringMegabytes) {
    Config cfg;

    cfg.set_max_db_size("500M");
    EXPECT_EQ(cfg.max_db_size, 500ULL * 1024 * 1024);

    cfg.set_max_db_size("500MB");
    EXPECT_EQ(cfg.max_db_size, 500ULL * 1024 * 1024);
}

TEST_F(ConfigTest, ParseSizeStringGigabytes) {
    Config cfg;

    cfg.set_max_db_size("10G");
    EXPECT_EQ(cfg.max_db_size, 10ULL * 1024 * 1024 * 1024);

    cfg.set_max_db_size("10GB");
    EXPECT_EQ(cfg.max_db_size, 10ULL * 1024 * 1024 * 1024);
}

TEST_F(ConfigTest, ParseSizeStringTerabytes) {
    Config cfg;

    cfg.set_max_db_size("1T");
    EXPECT_EQ(cfg.max_db_size, 1024ULL * 1024 * 1024 * 1024);

    cfg.set_max_db_size("1TB");
    EXPECT_EQ(cfg.max_db_size, 1024ULL * 1024 * 1024 * 1024);
}

TEST_F(ConfigTest, ParseSizeStringCaseInsensitive) {
    Config cfg;

    cfg.set_max_db_size("10g");
    EXPECT_EQ(cfg.max_db_size, 10ULL * 1024 * 1024 * 1024);

    cfg.set_max_db_size("10gb");
    EXPECT_EQ(cfg.max_db_size, 10ULL * 1024 * 1024 * 1024);

    cfg.set_max_db_size("10Gb");
    EXPECT_EQ(cfg.max_db_size, 10ULL * 1024 * 1024 * 1024);
}

TEST_F(ConfigTest, ParseSizeStringFractional) {
    Config cfg;

    cfg.set_max_db_size("1.5G");
    EXPECT_EQ(cfg.max_db_size, static_cast<size_t>(1.5 * 1024 * 1024 * 1024));

    cfg.set_max_db_size("0.5M");
    EXPECT_EQ(cfg.max_db_size, static_cast<size_t>(0.5 * 1024 * 1024));
}

TEST_F(ConfigTest, ParseSizeStringEmpty) {
    Config cfg;
    EXPECT_THROW(cfg.set_max_db_size(""), ConfigError);
}

TEST_F(ConfigTest, ParseSizeStringInvalidSuffix) {
    Config cfg;
    EXPECT_THROW(cfg.set_max_db_size("10X"), ConfigError);
}

TEST_F(ConfigTest, ParseSizeStringNoNumber) {
    Config cfg;
    EXPECT_THROW(cfg.set_max_db_size("GB"), ConfigError);
}

// =============================================================================
// get_home_directory tests
// =============================================================================

TEST_F(ConfigTest, GetHomeDirectoryReturnsNonEmpty) {
    std::string home = Config::get_home_directory();
    EXPECT_FALSE(home.empty());
    EXPECT_EQ(home[0], '/');  // Absolute path
}

TEST_F(ConfigTest, GetHomeDirectoryRespectsEnv) {
    test_helpers::ScopedEnv env("HOME", "/tmp/test_home_config");

    std::string home = Config::get_home_directory();
    EXPECT_EQ(home, "/tmp/test_home_config");
}

// =============================================================================
// Config defaults tests
// =============================================================================

TEST_F(ConfigTest, DefaultValues) {
    Config cfg;

    EXPECT_TRUE(cfg.streaming);
    EXPECT_FALSE(cfg.thinking);
    EXPECT_FALSE(cfg.stats);
    EXPECT_FALSE(cfg.tui);
    EXPECT_FALSE(cfg.warmup);
    EXPECT_FALSE(cfg.calibration);
    EXPECT_FALSE(cfg.auto_provider);
    EXPECT_FALSE(cfg.raw_output);
    EXPECT_EQ(cfg.backend, "llamacpp");
    EXPECT_EQ(cfg.context_size, 0u);  // Auto-detect
    EXPECT_EQ(cfg.truncate_limit, 0);
    EXPECT_EQ(cfg.tui_history, 10000);
}

TEST_F(ConfigTest, DefaultMaxDbSize) {
    Config cfg;

    // Default is "10G"
    EXPECT_EQ(cfg.max_db_size_str, "10G");
    EXPECT_EQ(cfg.max_db_size, 10ULL * 1024 * 1024 * 1024);
}

// =============================================================================
// get_available_backends tests
// =============================================================================

TEST_F(ConfigTest, GetAvailableBackendsNotEmpty) {
    auto backends = Config::get_available_backends();
    EXPECT_FALSE(backends.empty());

    // CLI backend should always be available
    bool has_cli = std::find(backends.begin(), backends.end(), "cli") != backends.end();
    EXPECT_TRUE(has_cli);
}

// =============================================================================
// set_backend tests
// =============================================================================

TEST_F(ConfigTest, SetBackendValid) {
    Config cfg;

    // CLI is always available
    EXPECT_NO_THROW(cfg.set_backend("cli"));
    EXPECT_EQ(cfg.backend, "cli");
}

TEST_F(ConfigTest, SetBackendInvalid) {
    Config cfg;
    EXPECT_THROW(cfg.set_backend("nonexistent_backend"), ConfigError);
}

// =============================================================================
// validate tests
// =============================================================================

TEST_F(ConfigTest, ValidateDefaultConfig) {
    Config cfg;
    EXPECT_NO_THROW(cfg.validate());
}

TEST_F(ConfigTest, ValidateMinDbSize) {
    Config cfg;
    cfg.max_db_size = 1024;  // Less than 1MB minimum
    EXPECT_THROW(cfg.validate(), ConfigError);
}

// =============================================================================
// Load/Save tests
// =============================================================================

TEST_F(ConfigTest, LoadMissingConfigUsesDefaults) {
    Config cfg;
    cfg.set_config_path(temp_dir->file_path("nonexistent.json"));

    // Should not throw, just use defaults
    EXPECT_NO_THROW(cfg.load());
    EXPECT_TRUE(cfg.streaming);  // Default value
}

TEST_F(ConfigTest, SaveAndLoad) {
    std::string config_path = temp_dir->file_path("test_config.json");

    // Save config with custom values
    {
        Config cfg;
        cfg.set_config_path(config_path);
        cfg.streaming = false;
        cfg.thinking = true;
        cfg.truncate_limit = 5000;
        cfg.set_max_db_size("5G");
        cfg.save();
    }

    // Load and verify
    {
        Config cfg;
        cfg.set_config_path(config_path);
        cfg.load();

        EXPECT_FALSE(cfg.streaming);
        EXPECT_TRUE(cfg.thinking);
        EXPECT_EQ(cfg.truncate_limit, 5000);
        EXPECT_EQ(cfg.max_db_size_str, "5G");
    }
}

TEST_F(ConfigTest, LoadInvalidJson) {
    std::string config_path = temp_dir->file_path("invalid.json");

    // Write invalid JSON
    std::ofstream file(config_path);
    file << "{ invalid json }";
    file.close();

    Config cfg;
    cfg.set_config_path(config_path);
    EXPECT_THROW(cfg.load(), ConfigError);
}

// =============================================================================
// handle_config_args tests
// =============================================================================

TEST_F(ConfigTest, HandleConfigArgsShow) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_config_args({"show"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("streaming") != std::string::npos);
    EXPECT_TRUE(output.find("thinking") != std::string::npos);
}

TEST_F(ConfigTest, HandleConfigArgsEmpty) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    // Empty args should show config
    int result = handle_config_args({}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("streaming") != std::string::npos);
}

TEST_F(ConfigTest, HandleConfigArgsHelp) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_config_args({"help"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("Usage") != std::string::npos);
}

TEST_F(ConfigTest, HandleConfigArgsSetInvalidKey) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_config_args({"set", "invalid_key", "value"}, callback);
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(output.find("Unknown") != std::string::npos);
}

TEST_F(ConfigTest, HandleConfigArgsUnknownSubcommand) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_config_args({"unknown"}, callback);
    EXPECT_EQ(result, 1);
}

// =============================================================================
// XDG paths tests
// =============================================================================

TEST_F(ConfigTest, GetDefaultConfigPathXdg) {
    test_helpers::ScopedEnv home("HOME", "/home/testuser");
    test_helpers::ScopedEnv xdg("XDG_CONFIG_HOME", "/custom/config");

    std::string path = Config::get_default_config_path();
    EXPECT_EQ(path, "/custom/config/shepherd/config.json");
}

TEST_F(ConfigTest, GetDefaultConfigPathNoXdg) {
    test_helpers::ScopedEnv home("HOME", "/home/testuser");
    unsetenv("XDG_CONFIG_HOME");

    std::string path = Config::get_default_config_path();
    EXPECT_EQ(path, "/home/testuser/.config/shepherd/config.json");
}

TEST_F(ConfigTest, GetDefaultMemoryDbPathXdg) {
    test_helpers::ScopedEnv home("HOME", "/home/testuser");
    test_helpers::ScopedEnv xdg("XDG_DATA_HOME", "/custom/data");

    std::string path = Config::get_default_memory_db_path();
    EXPECT_EQ(path, "/custom/data/shepherd/memory.db");
}

TEST_F(ConfigTest, GetDefaultMemoryDbPathNoXdg) {
    test_helpers::ScopedEnv home("HOME", "/home/testuser");
    unsetenv("XDG_DATA_HOME");

    std::string path = Config::get_default_memory_db_path();
    EXPECT_EQ(path, "/home/testuser/.local/share/shepherd/memory.db");
}
