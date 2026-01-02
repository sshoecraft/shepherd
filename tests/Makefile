# Shepherd Test Suite Makefile
# Separate from main project Makefile

BUILD_DIR := ../build
CMAKE_FLAGS := -DBUILD_TESTS=ON

.PHONY: all config build test test-unit test-tools clean help

all: test

help:
	@echo "Shepherd Test Suite"
	@echo ""
	@echo "Targets:"
	@echo "  make test          - Build and run all tests"
	@echo "  make test-unit     - Build and run unit tests (includes provider tests)"
	@echo "  make test-tools    - Build and run tool tests"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make clean         - Remove test executables"
	@echo ""
	@echo "Run specific test:"
	@echo "  make test-filter FILTER='ConfigTest.*'"
	@echo "  make test-filter FILTER='ProviderTest.*'"

config:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) ..

build: config
	cd $(BUILD_DIR) && $(MAKE) test_unit test_tools

test-unit: build
	@echo "=== Running Unit Tests ==="
	$(BUILD_DIR)/tests/test_unit

test-tools: build
	@echo "=== Running Tool Tests ==="
	$(BUILD_DIR)/tests/test_tools

test: test-unit test-tools

# Run specific test by filter
test-filter: build
	$(BUILD_DIR)/tests/test_unit --gtest_filter=$(FILTER)

# Run with verbose output
test-verbose: build
	$(BUILD_DIR)/tests/test_unit --gtest_output=xml:test_results.xml
	@echo "Results written to test_results.xml"

# List all tests
test-list: build
	$(BUILD_DIR)/tests/test_unit --gtest_list_tests

# Run via ctest
ctest: build
	cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	rm -f $(BUILD_DIR)/tests/test_unit
	rm -f $(BUILD_DIR)/tests/test_tools
	rm -f test_results.xml
