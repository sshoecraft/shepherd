
# Uncomment and set this to your python virtual env if you have one setup and you installed tensorrt-llm in it
#VENV:=/path/to/venv

LLAMACPP?=OFF
TENSORRT?=OFF
TESTS?=OFF
BUILD_TYPE?=Debug

ifneq ("$(wildcard ~/.shepherd_opts)","")
include ~/.shepherd_opts
endif
all:
	(cd build && make -j$(nproc))

# Build activation command if TensorRT is ON and VENV is set
ifeq ($(TENSORRT),ON)
  ifeq ($(VENV),)
    ifneq ("$(wildcard ~/venv)","")
     VENV=~/venv
    endif
  endif
  ifneq ($(VENV),)
    _ACT=. $(VENV)/bin/activate &&
  endif
endif

config:
	rm -rf build; mkdir -p build
	cd build && $(_ACT) cmake -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTS=$(TESTS) -DENABLE_API_BACKENDS=ON -DENABLE_LLAMACPP=$(LLAMACPP) -DENABLE_TENSORRT=$(TENSORRT) ..

install:
	cd build && make install

clean:
	(cd build && make clean)

distclean:
	rm -rf build

# this is just for quick testing - use cmake for full support (and for tensorrt support)
BACKENDS=backends/backend.cpp backends/factory.cpp backends/llamacpp.cpp backends/chat_template.cpp backends/models.cpp backends/api.cpp backends/openai.cpp backends/ollama.cpp backends/anthropic.cpp backends/gemini.cpp
TOOL_SRCS=tools/command_tools.cpp tools/core_tools.cpp tools/filesystem_tools.cpp tools/http_tools.cpp tools/json_tools.cpp tools/mcp_resource_tools.cpp tools/memory_tools.cpp tools/tool.cpp tools/tool_parser.cpp tools/utf8_sanitizer.cpp tools/web_search.cpp
MCP_SRCS=mcp/mcp_client.cpp mcp/mcp_config.cpp mcp/mcp.cpp mcp/mcp_server.cpp mcp/mcp_tool.cpp
API_TOOLS_SRCS=api_tools/api_tool_config.cpp api_tools/api_tool_adapter.cpp api_tools/api_tools.cpp
SERVER_SRCS=server/server.cpp
SRCS=main.cpp cli.cpp session.cpp config.cpp logger.cpp http_client.cpp rag.cpp auth.cpp $(BACKENDS) $(TOOL_SRCS) $(MCP_SRCS) $(API_TOOLS_SRCS) $(SERVER_SRCS)
OBJS=$(SRCS:%.cpp=%$(OBJSUFFIX).o)
DEPS=$(SRCS:%.cpp=.deps/%.d)
DEPDIR=.deps
LIBS=-lcurl -lsqlite3 -lcrypto -L./llama.cpp/build/bin -L./llama.cpp/build/common -lllama -lcommon -lggml -lggml-base -lggml-cpu -lggml-cuda -Wl,-rpath,./llama.cpp/build/bin
CCFLAGS=-DENABLE_API_BACKENDS -DENABLE_LLAMACPP -I./llama.cpp/include -I./llama.cpp/common
#LTOFLAG=-flto
#NDEBUGFFLAG=-DNDEBUG
#CCOPTS=-O3 -pipe -march=native -mtune=native $(LTOFLAG) $(NDEBUGFFLAG) -ffast-math -funroll-loops
CCOPTS=-g -O0 -D_DEBUG
CC=g++ $(CCOPTS)

shepherd: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

.SUFFIXES: .cpp .o

%.o : %.cpp | $(DEPDIR)
	@mkdir -p $(dir $(DEPDIR)/$@)
	$(CC) $(CCFLAGS) -MMD -MP -MF $(DEPDIR)/$*.d -I . -c $< -o $@

$(DEPDIR):
	@mkdir -p $(DEPDIR)/backends $(DEPDIR)/tools $(DEPDIR)/mcp $(DEPDIR)/api_tools $(DEPDIR)/server

oclean:
	rm -f shepherd $(OBJS)
	rm -rf $(DEPDIR)

-include $(DEPS)
