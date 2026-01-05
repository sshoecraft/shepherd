#include <gtest/gtest.h>
#include "auth.h"
#include "test_helpers.h"
#include "temp_dir.h"
#include <fstream>
#include <cstdlib>
#include <sys/stat.h>
#include <regex>

// Test fixture for Auth tests
class AuthTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test keys
        temp_dir = std::make_unique<test_helpers::TempDir>("auth_test_");
        ASSERT_TRUE(temp_dir->valid());

        // Save original XDG_CONFIG_HOME and set to temp dir
        const char* orig = getenv("XDG_CONFIG_HOME");
        if (orig) {
            original_xdg_config = orig;
        }
        setenv("XDG_CONFIG_HOME", temp_dir->path().c_str(), 1);

        // Create shepherd config directory
        std::string shepherd_dir = temp_dir->path() + "/shepherd";
        mkdir(shepherd_dir.c_str(), 0755);
    }

    void TearDown() override {
        // Restore original XDG_CONFIG_HOME
        if (!original_xdg_config.empty()) {
            setenv("XDG_CONFIG_HOME", original_xdg_config.c_str(), 1);
        } else {
            unsetenv("XDG_CONFIG_HOME");
        }
        temp_dir.reset();
    }

    std::unique_ptr<test_helpers::TempDir> temp_dir;
    std::string original_xdg_config;
};

// =============================================================================
// Key Generation Tests (AUTH-001 to AUTH-003)
// =============================================================================

TEST_F(AuthTest, GenerateKeyFormat) {
    // AUTH-001: generate_key() returns sk- + 32 alphanumeric chars
    std::string key = JsonKeyStore::generate_key();

    EXPECT_EQ(key.length(), 35u);  // "sk-" (3) + 32 chars
    EXPECT_EQ(key.substr(0, 3), "sk-");
}

TEST_F(AuthTest, GenerateKeyUniqueness) {
    // AUTH-002: Multiple calls return different keys
    std::string key1 = JsonKeyStore::generate_key();
    std::string key2 = JsonKeyStore::generate_key();
    std::string key3 = JsonKeyStore::generate_key();

    EXPECT_NE(key1, key2);
    EXPECT_NE(key2, key3);
    EXPECT_NE(key1, key3);
}

TEST_F(AuthTest, GenerateKeyCharacterSet) {
    // AUTH-003: Only [a-zA-Z0-9] after prefix
    std::string key = JsonKeyStore::generate_key();
    std::string suffix = key.substr(3);

    std::regex pattern("^[a-zA-Z0-9]+$");
    EXPECT_TRUE(std::regex_match(suffix, pattern));
}

// =============================================================================
// Key Storage Tests (AUTH-004 to AUTH-010)
// =============================================================================

TEST_F(AuthTest, SaveKeysCreatesFile) {
    // AUTH-004: save_keys() creates file at expected path
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    entry.notes = "";
    entry.created = "2026-01-03T12:00:00Z";
    entry.permissions = nlohmann::json::object();
    keys["sk-testkey12345678901234567890123"] = entry;

    JsonKeyStore::save_keys(keys);

    std::string path = JsonKeyStore::get_keys_file_path();
    EXPECT_TRUE(std::filesystem::exists(path));
}

TEST_F(AuthTest, SaveKeysFilePermissions) {
    // AUTH-005: File mode is 0600
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    entry.notes = "";
    entry.created = "2026-01-03T12:00:00Z";
    entry.permissions = nlohmann::json::object();
    keys["sk-testkey12345678901234567890123"] = entry;

    JsonKeyStore::save_keys(keys);

    std::string path = JsonKeyStore::get_keys_file_path();
    struct stat st;
    ASSERT_EQ(stat(path.c_str(), &st), 0);
    EXPECT_EQ(st.st_mode & 0777, 0600);
}

TEST_F(AuthTest, LoadKeysEmptyFile) {
    // AUTH-006: Empty file returns empty map (actually empty JSON object)
    std::string path = JsonKeyStore::get_keys_file_path();
    std::ofstream file(path);
    file << "{}";
    file.close();

    auto keys = JsonKeyStore::load_keys();
    EXPECT_TRUE(keys.empty());
}

TEST_F(AuthTest, LoadKeysValidJson) {
    // AUTH-007: Valid JSON returns populated map
    std::string path = JsonKeyStore::get_keys_file_path();
    std::ofstream file(path);
    file << R"({
        "sk-abc123": {
            "name": "primary",
            "notes": "Test key",
            "created": "2026-01-03T12:00:00Z",
            "permissions": {}
        }
    })";
    file.close();

    auto keys = JsonKeyStore::load_keys();
    EXPECT_EQ(keys.size(), 1u);
    EXPECT_TRUE(keys.count("sk-abc123") > 0);
    EXPECT_EQ(keys["sk-abc123"].name, "primary");
    EXPECT_EQ(keys["sk-abc123"].notes, "Test key");
}

TEST_F(AuthTest, LoadKeysMissingFile) {
    // AUTH-008: Missing file returns empty map (no error)
    auto keys = JsonKeyStore::load_keys();
    EXPECT_TRUE(keys.empty());
}

TEST_F(AuthTest, LoadKeysInvalidJson) {
    // AUTH-009: Invalid JSON throws exception
    std::string path = JsonKeyStore::get_keys_file_path();
    std::ofstream file(path);
    file << "{ invalid json }";
    file.close();

    EXPECT_THROW(JsonKeyStore::load_keys(), std::runtime_error);
}

TEST_F(AuthTest, RoundTripSaveLoad) {
    // AUTH-010: Round-trip save/load preserves data
    std::map<std::string, ApiKeyEntry> original;
    ApiKeyEntry entry;
    entry.name = "production";
    entry.notes = "Production API key";
    entry.created = "2026-01-03T12:00:00Z";
    entry.permissions = nlohmann::json::object();
    original["sk-prod123456789012345678901234"] = entry;

    JsonKeyStore::save_keys(original);
    auto loaded = JsonKeyStore::load_keys();

    EXPECT_EQ(loaded.size(), 1u);
    EXPECT_EQ(loaded["sk-prod123456789012345678901234"].name, "production");
    EXPECT_EQ(loaded["sk-prod123456789012345678901234"].notes, "Production API key");
    EXPECT_EQ(loaded["sk-prod123456789012345678901234"].created, "2026-01-03T12:00:00Z");
}

// =============================================================================
// Key Validation Tests (AUTH-011 to AUTH-014)
// =============================================================================

TEST_F(AuthTest, ValidateKeyValid) {
    // AUTH-011: validate_key() with valid key returns true
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    keys["sk-validkey1234567890123456789012"] = entry;
    JsonKeyStore::save_keys(keys);

    JsonKeyStore store;
    EXPECT_TRUE(store.validate_key("sk-validkey1234567890123456789012"));
}

TEST_F(AuthTest, ValidateKeyInvalid) {
    // AUTH-012: validate_key() with invalid key returns false
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    keys["sk-validkey1234567890123456789012"] = entry;
    JsonKeyStore::save_keys(keys);

    JsonKeyStore store;
    EXPECT_FALSE(store.validate_key("sk-wrongkey1234567890123456789012"));
}

TEST_F(AuthTest, ValidateKeyEmpty) {
    // AUTH-013: validate_key() with empty key returns false
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    keys["sk-validkey1234567890123456789012"] = entry;
    JsonKeyStore::save_keys(keys);

    JsonKeyStore store;
    EXPECT_FALSE(store.validate_key(""));
}

TEST_F(AuthTest, ValidateKeyWrongPrefix) {
    // AUTH-014: validate_key() with wrong prefix returns false
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    keys["sk-validkey1234567890123456789012"] = entry;
    JsonKeyStore::save_keys(keys);

    JsonKeyStore store;
    EXPECT_FALSE(store.validate_key("xx-validkey1234567890123456789012"));
}

// =============================================================================
// KeyStore Factory Tests (AUTH-016 to AUTH-020)
// =============================================================================

TEST_F(AuthTest, CreateNoneKeyStore) {
    // AUTH-016: create("none") returns NoneKeyStore
    auto store = KeyStore::create("none");
    EXPECT_NE(store, nullptr);

    // NoneKeyStore should be disabled and always validate
    EXPECT_FALSE(store->is_enabled());
    EXPECT_TRUE(store->validate_key("any-key"));
}

TEST_F(AuthTest, CreateJsonKeyStore) {
    // AUTH-017: create("json") returns JsonKeyStore
    auto store = KeyStore::create("json");
    EXPECT_NE(store, nullptr);
}

TEST_F(AuthTest, CreateInvalidMode) {
    // AUTH-018: create("invalid") throws exception
    EXPECT_THROW(KeyStore::create("invalid"), std::runtime_error);
}

TEST_F(AuthTest, NoneKeyStoreAlwaysValidates) {
    // AUTH-019: NoneKeyStore validate_key() always returns true
    NoneKeyStore store;
    EXPECT_TRUE(store.validate_key("anything"));
    EXPECT_TRUE(store.validate_key(""));
    EXPECT_TRUE(store.validate_key("sk-fake"));
}

TEST_F(AuthTest, NoneKeyStoreNotEnabled) {
    // AUTH-020: NoneKeyStore is_enabled() returns false
    NoneKeyStore store;
    EXPECT_FALSE(store.is_enabled());
}

// =============================================================================
// CLI Tests (AUTH-021 to AUTH-025)
// =============================================================================

TEST_F(AuthTest, HandleKeygenArgsGenerateKey) {
    // AUTH-021: handle_keygen_args(["--name", "test"]) generates and saves key
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_keygen_args({"--name", "test"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("API key generated successfully") != std::string::npos);
    EXPECT_TRUE(output.find("sk-") != std::string::npos);

    // Verify key was saved
    auto keys = JsonKeyStore::load_keys();
    EXPECT_EQ(keys.size(), 1u);
}

TEST_F(AuthTest, HandleKeygenArgsList) {
    // AUTH-022: handle_keygen_args(["list"]) lists keys
    // First create a key
    std::string output1;
    handle_keygen_args({"--name", "testkey"}, [&output1](const std::string& s) { output1 += s; });

    // Now list
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_keygen_args({"list"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("testkey") != std::string::npos);
}

TEST_F(AuthTest, HandleKeygenArgsRemove) {
    // AUTH-023: handle_keygen_args(["remove", "test"]) removes key
    // First create a key
    std::string output1;
    handle_keygen_args({"--name", "toremove"}, [&output1](const std::string& s) { output1 += s; });

    // Verify it exists
    auto keys = JsonKeyStore::load_keys();
    EXPECT_EQ(keys.size(), 1u);

    // Now remove
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_keygen_args({"remove", "toremove"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("removed") != std::string::npos);

    // Verify it's gone
    keys = JsonKeyStore::load_keys();
    EXPECT_TRUE(keys.empty());
}

TEST_F(AuthTest, HandleKeygenArgsRemoveNonexistent) {
    // AUTH-024: handle_keygen_args(["remove", "nonexistent"]) returns error
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_keygen_args({"remove", "nonexistent"}, callback);
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(output.find("not found") != std::string::npos);
}

TEST_F(AuthTest, HandleKeygenArgsDuplicateName) {
    // AUTH-025: Duplicate name returns error
    std::string output1;
    handle_keygen_args({"--name", "dup"}, [&output1](const std::string& s) { output1 += s; });

    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_keygen_args({"--name", "dup"}, callback);
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(output.find("already exists") != std::string::npos);
}

// =============================================================================
// JsonKeyStore is_enabled Tests
// =============================================================================

TEST_F(AuthTest, JsonKeyStoreEnabledWithKeys) {
    // JsonKeyStore is_enabled() returns true when keys exist
    std::map<std::string, ApiKeyEntry> keys;
    ApiKeyEntry entry;
    entry.name = "test";
    keys["sk-test123"] = entry;
    JsonKeyStore::save_keys(keys);

    JsonKeyStore store;
    EXPECT_TRUE(store.is_enabled());
}

TEST_F(AuthTest, JsonKeyStoreDisabledWithoutKeys) {
    // JsonKeyStore is_enabled() returns false when no keys
    JsonKeyStore store;
    EXPECT_FALSE(store.is_enabled());
}
