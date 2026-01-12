** STARTUP AND SHUTDOWN:
- To startup a llamacpp openai api server for testing: shepherd --provider llama-gpt-oss --apiserver (add --debug if needed)
- To shutdown the shepherd server (api and cli), use: shepherd ctl shutdown

** BUILDING:
- To change cmake build options: ~/.shepherd_opts (see: Makefile)
- To configure with cmake: make config
- To build: make
