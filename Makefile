

#backends/tensorrt.cpp
BACKENDS=backends/backend.cpp backends/factory.cpp backends/llamacpp.cpp backends/models.cpp backends/api.cpp backends/openai.cpp backends/ollama.cpp backends/anthropic.cpp backends/gemini.cpp
TOOL_SRCS=tools/command_tools.cpp tools/core_tools.cpp tools/filesystem_tools.cpp tools/http_tools.cpp tools/json_tools.cpp tools/mcp_resource_tools.cpp tools/memory_tools.cpp tools/tool.cpp tools/tool_parser.cpp tools/utf8_sanitizer.cpp tools/web_search.cpp
MCP_SRCS=mcp/mcp_client.cpp mcp/mcp_config.cpp mcp/mcp.cpp mcp/mcp_server.cpp mcp/mcp_tool.cpp
SERVER_SRCS=server/server.cpp
SRCS=main.cpp cli.cpp session.cpp config.cpp logger.cpp http_client.cpp rag.cpp $(BACKENDS) $(TOOL_SRCS) $(MCP_SRCS) $(SERVER_SRCS)
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
	@mkdir -p $(DEPDIR)/backends $(DEPDIR)/tools $(DEPDIR)/mcp $(DEPDIR)/server

clean:
	rm -f shepherd $(OBJS)
	rm -rf $(DEPDIR)

-include $(DEPS)
