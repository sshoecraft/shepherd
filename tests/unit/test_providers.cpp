#include <gtest/gtest.h>
#include "provider.h"
#include "session.h"
#include "backend.h"
#include "config.h"
#include <memory>
#include <atomic>

// Global config required by provider system
std::unique_ptr<Config> config;

class ProviderIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize global config
        config = std::make_unique<Config>();
        config->load();
    }

    static void TearDownTestSuite() {
        config.reset();
    }

    void SetUp() override {
        providers = Provider::load_providers();
    }

    std::vector<Provider> providers;
};

// Test that we can load providers
TEST_F(ProviderIntegrationTest, LoadProviders) {
    // This test passes if providers load without crashing
    // Empty provider list is valid (user may not have configured any)
    SUCCEED();
}

// Test each configured API provider
class APIProviderTest : public ProviderIntegrationTest,
                        public ::testing::WithParamInterface<std::string> {
protected:
    Provider* FindProvider(const std::string& type) {
        for (auto& p : providers) {
            if (p.type == type) return &p;
        }
        return nullptr;
    }
};

TEST_P(APIProviderTest, ConnectAndGenerate) {
    std::string provider_type = GetParam();
    Provider* prov = FindProvider(provider_type);

    if (!prov) {
        GTEST_SKIP() << "No " << provider_type << " provider configured";
    }

    // Create session
    Session session;
    std::string output;
    std::atomic<bool> got_content{false};
    std::atomic<bool> got_stop{false};
    std::atomic<bool> got_error{false};
    std::string error_msg;

    auto callback = [&](CallbackEvent event, const std::string& content,
                       const std::string& name, const std::string& id) -> bool {
        switch (event) {
            case CallbackEvent::CONTENT:
                output += content;
                got_content = true;
                break;
            case CallbackEvent::STOP:
                got_stop = true;
                break;
            case CallbackEvent::ERROR:
                got_error = true;
                error_msg = content;
                break;
            default:
                break;
        }
        return true;
    };

    // Connect to provider
    std::unique_ptr<Backend> backend;
    try {
        backend = prov->connect(session, callback);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to connect to " << provider_type << ": " << e.what();
    }

    ASSERT_NE(backend, nullptr) << "Backend creation failed for " << provider_type;

    // Set minimal system message
    session.system_message = "You are a helpful assistant. Be very brief.";
    session.backend = backend.get();

    // Send a simple test message
    try {
        session.add_message(Message::USER, "Say 'test passed' and nothing else.");
    } catch (const std::exception& e) {
        FAIL() << "Generation failed for " << provider_type << ": " << e.what();
    }

    // Check results
    if (got_error) {
        FAIL() << provider_type << " returned error: " << error_msg;
    }

    EXPECT_TRUE(got_content) << provider_type << " produced no content";
    EXPECT_TRUE(got_stop) << provider_type << " did not signal stop";
    EXPECT_FALSE(output.empty()) << provider_type << " output was empty";

    // Shutdown cleanly
    backend->shutdown();
}

// Test streaming works
TEST_P(APIProviderTest, StreamingOutput) {
    std::string provider_type = GetParam();
    Provider* prov = FindProvider(provider_type);

    if (!prov) {
        GTEST_SKIP() << "No " << provider_type << " provider configured";
    }

    Session session;
    int chunk_count = 0;
    std::atomic<bool> got_error{false};

    auto callback = [&](CallbackEvent event, const std::string& content,
                       const std::string& name, const std::string& id) -> bool {
        if (event == CallbackEvent::CONTENT && !content.empty()) {
            chunk_count++;
        }
        if (event == CallbackEvent::ERROR) {
            got_error = true;
        }
        return true;
    };

    std::unique_ptr<Backend> backend;
    try {
        backend = prov->connect(session, callback);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to connect to " << provider_type << ": " << e.what();
    }

    ASSERT_NE(backend, nullptr);

    session.system_message = "You are helpful.";
    session.backend = backend.get();

    try {
        // Ask for a longer response to ensure multiple chunks
        session.add_message(Message::USER, "Count from 1 to 10, one number per line.");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Generation failed: " << e.what();
    }

    if (!got_error) {
        // Streaming should produce multiple chunks for this prompt
        EXPECT_GT(chunk_count, 1) << provider_type << " should stream multiple chunks";
    }

    backend->shutdown();
}

// Instantiate tests for each API backend type
INSTANTIATE_TEST_SUITE_P(
    APIBackends,
    APIProviderTest,
    ::testing::Values("openai", "anthropic", "gemini", "ollama"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param;
    }
);

// Test provider switching
TEST_F(ProviderIntegrationTest, SwitchProviders) {
    // Need at least 2 providers to test switching
    std::vector<Provider*> api_providers;
    for (auto& p : providers) {
        if (p.is_api()) {
            api_providers.push_back(&p);
        }
    }

    if (api_providers.size() < 2) {
        GTEST_SKIP() << "Need at least 2 API providers to test switching";
    }

    Session session;
    std::string output;

    auto callback = [&](CallbackEvent event, const std::string& content,
                       const std::string&, const std::string&) -> bool {
        if (event == CallbackEvent::CONTENT) {
            output += content;
        }
        return true;
    };

    // Connect to first provider
    std::unique_ptr<Backend> backend1;
    try {
        backend1 = api_providers[0]->connect(session, callback);
    } catch (...) {
        GTEST_SKIP() << "Failed to connect to first provider";
    }
    ASSERT_NE(backend1, nullptr);

    session.system_message = "Be brief.";
    session.backend = backend1.get();

    // Generate something
    try {
        session.add_message(Message::USER, "Say hello.");
    } catch (...) {
        GTEST_SKIP() << "First provider generation failed";
    }

    EXPECT_FALSE(output.empty()) << "First provider produced no output";
    std::string first_output = output;
    output.clear();

    // Shutdown first backend
    backend1->shutdown();

    // Switch to second provider
    std::unique_ptr<Backend> backend2;
    try {
        backend2 = api_providers[1]->connect(session, callback);
    } catch (...) {
        GTEST_SKIP() << "Failed to connect to second provider";
    }
    ASSERT_NE(backend2, nullptr);

    session.backend = backend2.get();

    // Generate again
    try {
        session.add_message(Message::USER, "Say goodbye.");
    } catch (...) {
        GTEST_SKIP() << "Second provider generation failed";
    }

    EXPECT_FALSE(output.empty()) << "Second provider produced no output";

    backend2->shutdown();
}

// Test handle_provider_args
TEST_F(ProviderIntegrationTest, HandleProviderArgsList) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_provider_args({"list"}, callback);
    EXPECT_EQ(result, 0);

    // Should either show providers or "No providers configured"
    EXPECT_FALSE(output.empty());
}

TEST_F(ProviderIntegrationTest, HandleProviderArgsHelp) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_provider_args({"help"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("Usage") != std::string::npos);
}

TEST_F(ProviderIntegrationTest, HandleProviderArgsShowNonexistent) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_provider_args({"show", "nonexistent_provider_xyz"}, callback);
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(output.find("not found") != std::string::npos);
}
