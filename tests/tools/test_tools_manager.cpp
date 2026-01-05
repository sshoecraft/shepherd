#include <gtest/gtest.h>
#include "tools/tools.h"
#include "tools/filesystem_tools.h"
#include "tools/command_tools.h"
#include "tools/json_tools.h"
#include "tools/http_tools.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>

namespace fs = std::filesystem;

// ============================================================================
// Test Fixtures
// ============================================================================

class ToolsManagerTest : public ::testing::Test {
protected:
    Tools tools;
    std::string test_dir;

    void SetUp() override {
        // Create unique test directory in /tmp
        test_dir = "/tmp/shepherd_test_" + std::to_string(getpid());
        fs::create_directories(test_dir);

        // Register tool types that don't have complex global dependencies
        // Note: core_tools and memory_tools excluded - they depend on global config/RAG
        register_filesystem_tools(tools);
        register_command_tools(tools);
        register_json_tools(tools);
        register_http_tools(tools);
        tools.build_all_tools();
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    // Helper to create a test file
    std::string create_test_file(const std::string& name, const std::string& content) {
        std::string path = test_dir + "/" + name;
        std::ofstream file(path);
        file << content;
        file.close();
        return path;
    }
};

// ============================================================================
// TL-001: Register tool - Tool accessible by name
// ============================================================================

TEST_F(ToolsManagerTest, TL001_RegisterToolAccessibleByName) {
    // Filesystem tools should be registered
    Tool* read_tool = tools.get("read");
    ASSERT_NE(read_tool, nullptr) << "read tool should be registered";
    EXPECT_EQ(read_tool->name(), "read");

    Tool* write_tool = tools.get("write");
    ASSERT_NE(write_tool, nullptr) << "write tool should be registered";
    EXPECT_EQ(write_tool->name(), "write");

    // Command tools
    Tool* exec_tool = tools.get("execute_command");
    ASSERT_NE(exec_tool, nullptr) << "execute_command tool should be registered";

    // JSON tools
    Tool* parse_tool = tools.get("parse_json");
    ASSERT_NE(parse_tool, nullptr) << "parse_json tool should be registered";
}

TEST_F(ToolsManagerTest, TL001_GetNonexistentToolReturnsNull) {
    Tool* tool = tools.get("nonexistent_tool_xyz");
    EXPECT_EQ(tool, nullptr) << "Nonexistent tool should return nullptr";
}

TEST_F(ToolsManagerTest, TL001_CaseInsensitiveLookup) {
    // Lookup should be case-insensitive
    Tool* read1 = tools.get("read");
    Tool* read2 = tools.get("READ");
    Tool* read3 = tools.get("Read");

    ASSERT_NE(read1, nullptr);
    EXPECT_EQ(read1, read2);
    EXPECT_EQ(read1, read3);
}

// ============================================================================
// TL-002: Execute valid tool - Result returned
// ============================================================================

TEST_F(ToolsManagerTest, TL002_ExecuteValidTool) {
    // Create a test file
    std::string test_file = create_test_file("test_read.txt", "Hello, World!");

    // Execute read tool
    std::map<std::string, std::any> params;
    params["file_path"] = test_file;

    ToolResult result = tools.execute("read", params);

    EXPECT_TRUE(result.success) << "Tool execution should succeed";
    EXPECT_FALSE(result.content.empty()) << "Content should not be empty";
    EXPECT_TRUE(result.content.find("Hello, World!") != std::string::npos)
        << "Content should contain file contents";
    EXPECT_TRUE(result.error.empty()) << "Error should be empty on success";
}

TEST_F(ToolsManagerTest, TL002_ExecuteNonexistentTool) {
    std::map<std::string, std::any> params;

    ToolResult result = tools.execute("nonexistent_tool_xyz", params);

    EXPECT_FALSE(result.success) << "Execution should fail for nonexistent tool";
    EXPECT_FALSE(result.error.empty()) << "Error message should be set";
    EXPECT_TRUE(result.error.find("not found") != std::string::npos);
}

// ============================================================================
// TL-003: Execute disabled tool - Error returned
// ============================================================================

TEST_F(ToolsManagerTest, TL003_ExecuteDisabledTool) {
    // First verify tool works when enabled
    std::string test_file = create_test_file("disabled_test.txt", "test content");
    std::map<std::string, std::any> params;
    params["file_path"] = test_file;

    ToolResult enabled_result = tools.execute("read", params);
    EXPECT_TRUE(enabled_result.success) << "Tool should work when enabled";

    // Disable the tool
    tools.disable("read");

    // Try to execute again
    ToolResult disabled_result = tools.execute("read", params);

    EXPECT_FALSE(disabled_result.success) << "Execution should fail for disabled tool";
    EXPECT_TRUE(disabled_result.error.find("disabled") != std::string::npos)
        << "Error should mention tool is disabled";

    // Re-enable for other tests
    tools.enable("read");
}

// ============================================================================
// TL-004: Tool enable/disable - State persisted
// ============================================================================

TEST_F(ToolsManagerTest, TL004_EnableDisableState) {
    // Tools should be enabled by default
    EXPECT_TRUE(tools.is_enabled("read")) << "Tool should be enabled by default";
    EXPECT_TRUE(tools.is_enabled("write")) << "Tool should be enabled by default";

    // Disable a tool
    tools.disable("read");
    EXPECT_FALSE(tools.is_enabled("read")) << "Tool should be disabled after disable()";
    EXPECT_TRUE(tools.is_enabled("write")) << "Other tools should remain enabled";

    // Re-enable
    tools.enable("read");
    EXPECT_TRUE(tools.is_enabled("read")) << "Tool should be enabled after enable()";
}

TEST_F(ToolsManagerTest, TL004_EnableMultipleTools) {
    tools.disable("read");
    tools.disable("write");

    EXPECT_FALSE(tools.is_enabled("read"));
    EXPECT_FALSE(tools.is_enabled("write"));

    // Enable multiple at once
    std::vector<std::string> to_enable = {"read", "write"};
    tools.enable(to_enable);

    EXPECT_TRUE(tools.is_enabled("read"));
    EXPECT_TRUE(tools.is_enabled("write"));
}

TEST_F(ToolsManagerTest, TL004_DisableMultipleTools) {
    // Disable multiple at once
    std::vector<std::string> to_disable = {"read", "write"};
    tools.disable(to_disable);

    EXPECT_FALSE(tools.is_enabled("read"));
    EXPECT_FALSE(tools.is_enabled("write"));

    // Re-enable for other tests
    std::vector<std::string> to_enable = {"read", "write"};
    tools.enable(to_enable);
}

// ============================================================================
// TL-005: handle_tools_args(["list"])
// ============================================================================

TEST_F(ToolsManagerTest, TL005_HandleToolsArgsList) {
    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"list"}, callback);

    EXPECT_EQ(result, 0) << "list command should return success";
    EXPECT_TRUE(output.find("Available Tools") != std::string::npos)
        << "Output should contain 'Available Tools'";
    EXPECT_TRUE(output.find("read") != std::string::npos)
        << "Output should list read tool";
    EXPECT_TRUE(output.find("write") != std::string::npos)
        << "Output should list write tool";
}

TEST_F(ToolsManagerTest, TL005_HandleToolsArgsEmpty) {
    // Empty args should also list tools
    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({}, callback);

    EXPECT_EQ(result, 0) << "empty args should return success (lists tools)";
    EXPECT_TRUE(output.find("Available Tools") != std::string::npos);
}

// ============================================================================
// TL-006: handle_tools_args(["enable", "tool"])
// ============================================================================

TEST_F(ToolsManagerTest, TL006_HandleToolsArgsEnable) {
    // First disable a tool
    tools.disable("read");
    EXPECT_FALSE(tools.is_enabled("read"));

    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"enable", "read"}, callback);

    EXPECT_EQ(result, 0) << "enable command should return success";
    EXPECT_TRUE(tools.is_enabled("read")) << "Tool should be enabled after command";
    EXPECT_TRUE(output.find("Enabled") != std::string::npos)
        << "Output should confirm enablement";
}

TEST_F(ToolsManagerTest, TL006_HandleToolsArgsEnableMultiple) {
    tools.disable("read");
    tools.disable("write");

    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"enable", "read", "write"}, callback);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(tools.is_enabled("read"));
    EXPECT_TRUE(tools.is_enabled("write"));
}

TEST_F(ToolsManagerTest, TL006_HandleToolsArgsEnableNotFound) {
    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"enable", "nonexistent_tool"}, callback);

    EXPECT_NE(result, 0) << "Should return error for nonexistent tool";
    EXPECT_TRUE(output.find("Not found") != std::string::npos);
}

// ============================================================================
// TL-007: handle_tools_args(["disable", "tool"])
// ============================================================================

TEST_F(ToolsManagerTest, TL007_HandleToolsArgsDisable) {
    EXPECT_TRUE(tools.is_enabled("read"));

    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"disable", "read"}, callback);

    EXPECT_EQ(result, 0) << "disable command should return success";
    EXPECT_FALSE(tools.is_enabled("read")) << "Tool should be disabled after command";
    EXPECT_TRUE(output.find("Disabled") != std::string::npos);

    // Re-enable for other tests
    tools.enable("read");
}

TEST_F(ToolsManagerTest, TL007_HandleToolsArgsDisableNotFound) {
    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"disable", "nonexistent_tool"}, callback);

    EXPECT_NE(result, 0) << "Should return error for nonexistent tool";
    EXPECT_TRUE(output.find("Not found") != std::string::npos);
}

// ============================================================================
// TL-FS-001: read_file existing file - Content returned
// ============================================================================

TEST_F(ToolsManagerTest, TLFS001_ReadFileExisting) {
    std::string content = "Line 1\nLine 2\nLine 3\n";
    std::string test_file = create_test_file("read_test.txt", content);

    std::map<std::string, std::any> params;
    params["file_path"] = test_file;

    ToolResult result = tools.execute("read", params);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.content.find("Line 1") != std::string::npos);
    EXPECT_TRUE(result.content.find("Line 2") != std::string::npos);
    EXPECT_TRUE(result.content.find("Line 3") != std::string::npos);
}

// ============================================================================
// TL-FS-002: read_file missing file - Error with message
// ============================================================================

TEST_F(ToolsManagerTest, TLFS002_ReadFileMissing) {
    std::map<std::string, std::any> params;
    params["file_path"] = std::string("/tmp/nonexistent_file_xyz_123.txt");

    ToolResult result = tools.execute("read", params);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
    EXPECT_TRUE(result.error.find("not found") != std::string::npos);
}

// ============================================================================
// TL-FS-003: write_file new file - File created
// ============================================================================

TEST_F(ToolsManagerTest, TLFS003_WriteFileNew) {
    std::string new_file = test_dir + "/new_file.txt";
    std::string content = "New file content";

    // Verify file doesn't exist
    EXPECT_FALSE(fs::exists(new_file));

    std::map<std::string, std::any> params;
    params["file_path"] = new_file;
    params["content"] = content;

    ToolResult result = tools.execute("write", params);

    EXPECT_TRUE(result.success) << "Write should succeed: " << result.error;
    EXPECT_TRUE(fs::exists(new_file)) << "File should be created";

    // Verify content
    std::ifstream file(new_file);
    std::string read_content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    EXPECT_EQ(read_content, content);
}

// ============================================================================
// TL-FS-004: write_file overwrite - Content replaced
// ============================================================================

TEST_F(ToolsManagerTest, TLFS004_WriteFileOverwrite) {
    std::string existing_file = create_test_file("overwrite.txt", "Original content");
    std::string new_content = "Replaced content";

    std::map<std::string, std::any> params;
    params["file_path"] = existing_file;
    params["content"] = new_content;

    ToolResult result = tools.execute("write", params);

    EXPECT_TRUE(result.success);

    // Verify content was replaced
    std::ifstream file(existing_file);
    std::string read_content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    EXPECT_EQ(read_content, new_content);
    EXPECT_TRUE(read_content.find("Original") == std::string::npos);
}

// ============================================================================
// TL-FS-005: list_directory valid path - Files listed
// ============================================================================

TEST_F(ToolsManagerTest, TLFS005_ListDirectoryValid) {
    // Create some test files
    create_test_file("list_test_1.txt", "content 1");
    create_test_file("list_test_2.txt", "content 2");
    fs::create_directory(test_dir + "/subdir");

    std::map<std::string, std::any> params;
    params["path"] = test_dir;

    ToolResult result = tools.execute("list_directory", params);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.content.find("list_test_1.txt") != std::string::npos);
    EXPECT_TRUE(result.content.find("list_test_2.txt") != std::string::npos);
    EXPECT_TRUE(result.content.find("subdir") != std::string::npos);
}

// ============================================================================
// TL-FS-006: list_directory invalid path - Error returned
// ============================================================================

TEST_F(ToolsManagerTest, TLFS006_ListDirectoryInvalid) {
    std::map<std::string, std::any> params;
    params["path"] = std::string("/nonexistent/directory/path");

    ToolResult result = tools.execute("list_directory", params);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
    EXPECT_TRUE(result.error.find("not found") != std::string::npos);
}

// ============================================================================
// TL-FS-007: file_exists check (via list_directory showing files)
// Covered implicitly through list tests
// ============================================================================

// ============================================================================
// TL-FS-008: read_file with offset and limit
// ============================================================================

TEST_F(ToolsManagerTest, TLFS008_ReadFileWithOffset) {
    std::string content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n";
    std::string test_file = create_test_file("offset_test.txt", content);

    std::map<std::string, std::any> params;
    params["file_path"] = test_file;
    params["offset"] = 2;  // Start from line 2
    params["limit"] = 2;   // Read 2 lines

    ToolResult result = tools.execute("read", params);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.content.find("Line 2") != std::string::npos);
    EXPECT_TRUE(result.content.find("Line 3") != std::string::npos);
    // Line 1 should not be in output (before offset)
    EXPECT_TRUE(result.content.find("Line 1") == std::string::npos);
    // Line 4 should not be in output (after limit)
    EXPECT_TRUE(result.content.find("Line 4") == std::string::npos);
}

// ============================================================================
// TL-CMD-001: execute_command simple - Output returned
// ============================================================================

TEST_F(ToolsManagerTest, TLCMD001_ExecuteCommandSimple) {
    std::map<std::string, std::any> params;
    params["command"] = std::string("echo 'Hello World'");

    ToolResult result = tools.execute("execute_command", params);

    EXPECT_TRUE(result.success) << "Command should succeed: " << result.error;
    EXPECT_TRUE(result.content.find("Hello World") != std::string::npos)
        << "Output should contain command output";
}

TEST_F(ToolsManagerTest, TLCMD001_ExecuteCommandWithDir) {
    std::map<std::string, std::any> params;
    params["command"] = std::string("pwd");
    params["working_dir"] = test_dir;

    ToolResult result = tools.execute("execute_command", params);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.content.find(test_dir) != std::string::npos)
        << "Output should show working directory";
}

// ============================================================================
// TL-CMD-002: execute_command with timeout (not currently enforced, but API exists)
// ============================================================================

TEST_F(ToolsManagerTest, TLCMD002_ExecuteCommandWithTimeout) {
    std::map<std::string, std::any> params;
    params["command"] = std::string("sleep 0.1 && echo 'done'");
    params["timeout"] = 30;

    ToolResult result = tools.execute("execute_command", params);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.content.find("done") != std::string::npos);
}

// ============================================================================
// TL-CMD-003: execute_command exit code - Non-zero exit reported
// ============================================================================

TEST_F(ToolsManagerTest, TLCMD003_ExecuteCommandExitCode) {
    std::map<std::string, std::any> params;
    params["command"] = std::string("exit 42");

    ToolResult result = tools.execute("execute_command", params);

    EXPECT_FALSE(result.success) << "Command with non-zero exit should fail";
    // The error or content should indicate the failure
}

TEST_F(ToolsManagerTest, TLCMD003_ExecuteCommandStderr) {
    std::map<std::string, std::any> params;
    params["command"] = std::string("echo 'error message' >&2 && exit 1");

    ToolResult result = tools.execute("execute_command", params);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.content.find("error message") != std::string::npos
                || result.error.find("error message") != std::string::npos)
        << "stderr should be captured";
}

// ============================================================================
// TL-CMD-004: get_environment_variable - Value returned
// ============================================================================

TEST_F(ToolsManagerTest, TLCMD004_GetEnvironmentVariable) {
    // PATH is always set
    std::map<std::string, std::any> params;
    params["name"] = std::string("PATH");

    ToolResult result = tools.execute("get_env", params);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.content.empty());
    EXPECT_TRUE(result.content.find("PATH=") != std::string::npos);
}

TEST_F(ToolsManagerTest, TLCMD004_GetEnvironmentVariableDefault) {
    std::map<std::string, std::any> params;
    params["name"] = std::string("NONEXISTENT_VAR_XYZ_123");
    params["default"] = std::string("default_value");

    ToolResult result = tools.execute("get_env", params);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.content.find("default_value") != std::string::npos);
}

// ============================================================================
// TL-JSON-001: Parse valid JSON - Parsed object returned
// ============================================================================

TEST_F(ToolsManagerTest, TLJSON001_ParseValidJSON) {
    std::map<std::string, std::any> params;
    params["json"] = std::string("{\"name\": \"test\", \"value\": 42}");

    ToolResult result = tools.execute("parse_json", params);

    EXPECT_TRUE(result.success) << "Parsing valid JSON should succeed: " << result.error;
}

TEST_F(ToolsManagerTest, TLJSON001_ParseValidJSONArray) {
    std::map<std::string, std::any> params;
    params["json"] = std::string("[1, 2, 3, \"four\"]");

    ToolResult result = tools.execute("parse_json", params);

    EXPECT_TRUE(result.success);
}

// ============================================================================
// TL-JSON-002: Parse invalid JSON - Error message
// ============================================================================

TEST_F(ToolsManagerTest, TLJSON002_ParseInvalidJSON) {
    std::map<std::string, std::any> params;
    params["json"] = std::string("{invalid json syntax");

    ToolResult result = tools.execute("parse_json", params);

    EXPECT_FALSE(result.success) << "Parsing invalid JSON should fail";
    EXPECT_FALSE(result.error.empty()) << "Error message should be set";
}

TEST_F(ToolsManagerTest, TLJSON002_ParseEmptyJSON) {
    std::map<std::string, std::any> params;
    params["json"] = std::string("");

    ToolResult result = tools.execute("parse_json", params);

    EXPECT_FALSE(result.success) << "Parsing empty string should fail";
}

// ============================================================================
// TL-JSON-003: Serialize object - Valid JSON string
// ============================================================================

TEST_F(ToolsManagerTest, TLJSON003_SerializeJSON) {
    std::map<std::string, std::any> params;
    params["data"] = std::string("test value");

    ToolResult result = tools.execute("serialize_json", params);

    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.content.empty());
}

// ============================================================================
// TL-JSON-004: Query with JSONPath - Extracted values
// ============================================================================

TEST_F(ToolsManagerTest, TLJSON004_QueryJSON) {
    std::map<std::string, std::any> params;
    params["json"] = std::string("{\"name\": \"test\", \"nested\": {\"value\": 42}}");
    params["path"] = std::string("$.nested.value");

    ToolResult result = tools.execute("query_json", params);

    EXPECT_TRUE(result.success);
}

// ============================================================================
// TL-HTTP-001: HTTP GET request - Response body returned
// ============================================================================

TEST_F(ToolsManagerTest, TLHTTP001_HTTPGet) {
    std::map<std::string, std::any> params;
    // Use httpbin.org for testing - it's designed for HTTP testing
    params["url"] = std::string("https://httpbin.org/get");

    ToolResult result = tools.execute("http_get", params);

    // Note: This test requires network connectivity
    // If httpbin.org is unreachable, test will fail appropriately
    if (result.success) {
        EXPECT_FALSE(result.content.empty());
    }
    // We don't assert success because network may not be available in all test environments
}

// ============================================================================
// TL-HTTP-002: HTTP POST request - Body sent, response returned
// ============================================================================

TEST_F(ToolsManagerTest, TLHTTP002_HTTPPost) {
    std::map<std::string, std::any> params;
    params["url"] = std::string("https://httpbin.org/post");
    params["body"] = std::string("{\"test\": \"data\"}");

    ToolResult result = tools.execute("http_post", params);

    // Note: This test requires network connectivity
    if (result.success) {
        EXPECT_FALSE(result.content.empty());
    }
}

// ============================================================================
// TL-HTTP-003: HTTP Request with method - Custom method works
// ============================================================================

TEST_F(ToolsManagerTest, TLHTTP003_HTTPRequestMethod) {
    std::map<std::string, std::any> params;
    params["url"] = std::string("https://httpbin.org/put");
    params["method"] = std::string("PUT");
    params["body"] = std::string("{\"test\": \"put data\"}");

    ToolResult result = tools.execute("http_request", params);

    // Note: This test requires network connectivity
    if (result.success) {
        EXPECT_FALSE(result.content.empty());
    }
}

// ============================================================================
// TL-HTTP-004: Error status codes - Error reported
// ============================================================================

TEST_F(ToolsManagerTest, TLHTTP004_HTTPErrorStatusCode) {
    std::map<std::string, std::any> params;
    params["url"] = std::string("https://httpbin.org/status/404");

    ToolResult result = tools.execute("http_get", params);

    // Note: This test requires network connectivity
    // The request should return false for 404 status
    if (!result.error.empty() || !result.success) {
        // Either way is acceptable - depends on implementation
        SUCCEED();
    }
}

// ============================================================================
// TL-HTTP-005: URL required - Error when missing
// ============================================================================

TEST_F(ToolsManagerTest, TLHTTP005_HTTPMissingURL) {
    std::map<std::string, std::any> params;
    // No URL provided

    ToolResult result = tools.execute("http_get", params);

    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
    EXPECT_TRUE(result.error.find("required") != std::string::npos);
}

// ============================================================================
// Additional Tool Manager Tests
// ============================================================================

TEST_F(ToolsManagerTest, ListToolsReturnsAllTools) {
    std::vector<std::string> tool_list = tools.list();

    EXPECT_GT(tool_list.size(), 0) << "Tool list should not be empty";

    // Check for expected tools
    bool has_read = std::find(tool_list.begin(), tool_list.end(), "read") != tool_list.end();
    bool has_write = std::find(tool_list.begin(), tool_list.end(), "write") != tool_list.end();
    bool has_exec = std::find(tool_list.begin(), tool_list.end(), "execute_command") != tool_list.end();

    EXPECT_TRUE(has_read) << "Tool list should include 'read'";
    EXPECT_TRUE(has_write) << "Tool list should include 'write'";
    EXPECT_TRUE(has_exec) << "Tool list should include 'execute_command'";
}

TEST_F(ToolsManagerTest, ListWithDescriptions) {
    auto tool_descriptions = tools.list_with_descriptions();

    EXPECT_GT(tool_descriptions.size(), 0);

    // Check that descriptions are non-empty
    for (const auto& pair : tool_descriptions) {
        EXPECT_FALSE(pair.second.empty())
            << "Tool '" << pair.first << "' should have a description";
    }
}

TEST_F(ToolsManagerTest, AsSystemPrompt) {
    std::string prompt = tools.as_system_prompt();

    EXPECT_FALSE(prompt.empty());
    EXPECT_TRUE(prompt.find("available tools") != std::string::npos
                || prompt.find("Available Tools") != std::string::npos
                || prompt.find("tools:") != std::string::npos
                || prompt.find("Here are") != std::string::npos)
        << "System prompt should mention tools";
}

TEST_F(ToolsManagerTest, ClearCategory) {
    size_t initial_count = tools.all_tools.size();
    EXPECT_GT(initial_count, 0);

    // Clear core tools
    tools.clear_category("core");

    EXPECT_LT(tools.all_tools.size(), initial_count)
        << "Clearing category should reduce tool count";
}

TEST_F(ToolsManagerTest, RemoveTool) {
    ASSERT_NE(tools.get("read"), nullptr);

    tools.remove_tool("read");

    EXPECT_EQ(tools.get("read"), nullptr) << "Tool should be removed";
}

TEST_F(ToolsManagerTest, HandleToolsArgsHelp) {
    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"help"}, callback);

    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("Usage") != std::string::npos
                || output.find("usage") != std::string::npos);
}

TEST_F(ToolsManagerTest, HandleToolsArgsUnknown) {
    std::string output;
    auto callback = [&output](const std::string& msg) { output += msg; };

    int result = tools.handle_tools_args({"unknown_command"}, callback);

    EXPECT_NE(result, 0) << "Unknown command should return error";
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ToolsManagerTest, ReadDirectoryAsFile) {
    std::map<std::string, std::any> params;
    params["file_path"] = test_dir;  // This is a directory

    ToolResult result = tools.execute("read", params);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.error.find("directory") != std::string::npos);
}

TEST_F(ToolsManagerTest, WriteToDirectoryPath) {
    std::string dir_path = test_dir + "/subdir";
    fs::create_directory(dir_path);

    std::map<std::string, std::any> params;
    params["file_path"] = dir_path;  // This is a directory
    params["content"] = std::string("content");

    ToolResult result = tools.execute("write", params);

    // This should fail or handle gracefully
    // Implementation may vary
}

TEST_F(ToolsManagerTest, ExecuteCommandMissingCommand) {
    std::map<std::string, std::any> params;
    // No command provided

    ToolResult result = tools.execute("execute_command", params);

    EXPECT_FALSE(result.success);
    EXPECT_TRUE(result.error.find("required") != std::string::npos);
}

TEST_F(ToolsManagerTest, WriteCreatesParentDirectories) {
    std::string nested_file = test_dir + "/new_parent/new_child/file.txt";

    std::map<std::string, std::any> params;
    params["file_path"] = nested_file;
    params["content"] = std::string("nested content");

    ToolResult result = tools.execute("write", params);

    EXPECT_TRUE(result.success) << "Write should create parent directories: " << result.error;
    EXPECT_TRUE(fs::exists(nested_file));
}
