# GenerationThread Module

## Overview

`GenerationThread` (`generation_thread.cpp/h`) runs LLM generation in a background thread to prevent UI blocking. The main thread submits requests and polls for completion.

## Architecture

```
Main Thread                    Generation Thread
    │                                │
    ├─ Create GenerationThread       │
    │                                │
    ├─ gen_thread.start() ─────────→ │ (thread starts)
    │                                │
    ├─ gen_thread.submit(req) ─────→ │ (request queued)
    │                                ├─ session->add_message()
    │                                ├─ callback invoked per token
    ├─ while (!is_complete())        ├─ ... generating ...
    │      poll TUI, check escape    │
    │                                ├─ done
    ├─ gen_thread.is_complete() ←───┤
    │                                │
    └─ access gen_thread.last_response
```

## Key Classes

### GenerationRequest

```cpp
// Event callback type
using EventCallback = std::function<bool(
    Message::Type type,
    const std::string& content,
    const std::string& tool_name,
    const std::string& tool_call_id
)>;

struct GenerationRequest {
    Message::Type type;
    std::string content;
    std::string tool_name;
    std::string tool_id;
    int prompt_tokens;
    int max_tokens;
    EventCallback callback;  // Streaming callback
};
```

### GenerationThread

```cpp
class GenerationThread {
public:
    void init(Session* session);
    void start();
    void stop();

    void submit(GenerationRequest req);
    bool is_complete() const;
    void reset();

    Response last_response;
    bool busy{false};

private:
    Session* session{nullptr};
    std::thread worker;
    ThreadQueue<GenerationRequest> queue;
    std::atomic<bool> should_stop{false};
    std::atomic<bool> complete{false};
    GenerationRequest current_request;

    void worker_loop();
};
```

## Usage Pattern

### Initialization

```cpp
GenerationThread gen_thread;
gen_thread.init(&session);
gen_thread.start();
```

### Submitting Requests

```cpp
GenerationRequest req;
req.type = Message::USER;
req.content = user_input;
req.callback = [](Message::Type type, const std::string& content, ...) {
    tio.write(content, type);  // Stream to output
    return true;  // Continue
};

gen_thread.submit(req);
```

### Waiting for Completion

```cpp
tio.is_generating = true;

while (!gen_thread.is_complete()) {
    // Non-blocking work while waiting:
    if (tio.tui_mode && g_tui) {
        g_tui->run_once();  // Handle TUI refresh
    }

    if (tio.check_escape_pressed()) {
        g_generation_cancelled = true;  // Signal cancellation
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

tio.is_generating = false;
Response resp = gen_thread.last_response;
gen_thread.reset();
```

## Worker Loop

```cpp
void GenerationThread::worker_loop() {
    while (!should_stop) {
        // Wait for request
        if (!queue.try_pop(current_request, 100)) {
            continue;
        }

        busy = true;
        complete = false;

        try {
            // Unified add_message with optional callback
            last_response = session->add_message(
                current_request.type,
                current_request.content,
                current_request.callback,  // May be nullptr
                current_request.tool_name,
                current_request.tool_id,
                current_request.prompt_tokens,
                current_request.max_tokens
            );

            if (current_request.callback) {
                last_response.was_streamed = true;
            }
        } catch (const std::exception& e) {
            last_response = Response{};
            last_response.success = false;
            last_response.error = e.what();
        }

        complete = true;
        busy = false;
    }
}
```

## Streaming via Callback

The callback is invoked by the backend for each token:

```cpp
// In backend:
for (each token) {
    if (callback) {
        if (!callback(Message::ASSISTANT, token, "", "")) {
            break;  // Cancelled
        }
    }
}

// Frontend's callback:
req.callback = [](Message::Type type, const std::string& content, ...) {
    tio.write(content, type);  // Immediate output
    return !g_generation_cancelled;  // Check for cancel
};
```

## Cancellation

Generation can be cancelled via:
1. User presses Escape → `tio.check_escape_pressed()` returns true
2. Main thread sets `g_generation_cancelled = true`
3. Callback returns `false`
4. Backend stops generating

```cpp
// In main loop:
if (tio.check_escape_pressed()) {
    g_generation_cancelled = true;
}

// In callback:
return !g_generation_cancelled;
```

## Thread Safety

- `ThreadQueue` uses mutex + condition variable
- `complete` and `busy` are atomic
- `session` access is serialized (only one thread at a time)
- `last_response` is only read after `is_complete()` returns true

## Tool Call Loop

After generation completes, check for tool calls:

```cpp
Response resp = gen_thread.last_response;

while (has_tool_calls(resp)) {
    // Execute tool
    auto result = tools.execute(tool_call);

    // Submit result
    GenerationRequest req;
    req.type = Message::TOOL_RESPONSE;
    req.content = result.content;
    req.tool_name = tool_call.name;
    req.tool_id = tool_call.id;
    req.callback = streaming_callback;

    gen_thread.submit(req);

    // Wait for next response
    while (!gen_thread.is_complete()) { ... }
    resp = gen_thread.last_response;
}
```

## Error Handling

```cpp
if (!gen_thread.last_response.success) {
    tio.error("Generation failed: " + gen_thread.last_response.error);
}

if (gen_thread.last_response.finish_reason == "cancelled") {
    cli.show_cancelled();
}
```
