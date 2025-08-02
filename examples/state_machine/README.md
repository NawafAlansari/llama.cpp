# State Machine Example for llama.cpp

This example demonstrates a state machine-based implementation of text generation using llama.cpp. It provides the same functionality as the main tool but with a cleaner, more maintainable architecture.

## Features

- All features of the standard llama.cpp inference tool
- Cleaner code structure with explicit state transitions
- Improved error handling and debugging
- Easy to extend with new functionality

## Usage

Basic text generation:
```bash
./state_machine_example -m models/7B/ggml-model-q4_0.bin -p "Once upon a time" -n 128
```

Interactive chat:
```bash
./state_machine_example -m models/7B/ggml-model-q4_0.bin -i
```

Chat with system prompt:
```bash
./state_machine_example -m models/7B/ggml-model-q4_0.bin -sys "You are a helpful assistant"
```

## Implementation Details

This example implements a state machine with the following states:

- **INITIALIZING**: Initial setup and parameter parsing
- **LOADING_MODEL**: Loading the model from file
- **PREPARING_PROMPT**: Tokenizing and preparing the initial prompt
- **LOADING_SESSION**: Loading session data if available
- **GENERATING_TOKENS**: Main token generation loop
- **MANAGING_CONTEXT**: Handling context window management
- **WAITING_FOR_INPUT**: Waiting for user input in interactive mode
- **PROCESSING_INPUT**: Processing received user input
- **FINISHING**: Final cleanup and shutdown
- **ERROR**: Error state

Each state is handled by a dedicated function, making the code easier to understand and maintain.

## State Transitions

The state machine follows this general flow:

```
INITIALIZING → LOADING_MODEL → PREPARING_PROMPT → LOADING_SESSION → GENERATING_TOKENS
                                                                           ↓
FINISHING ← WAITING_FOR_INPUT ← MANAGING_CONTEXT ← ←←←←←←←←←←←←←←←←←←←←←←←← ← ← ←
     ↑                                ↓
     ←←←←←←←←←←←←←←←←←←← PROCESSING_INPUT
```

Error handling is available from any state, transitioning to ERROR and then to FINISHING.

## Building

This example is built as part of the main llama.cpp build process when examples are enabled:

```bash
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
make -j state_machine_example
```

The executable will be available at `build/bin/state_machine_example`.

## Architecture Benefits

1. **Separation of Concerns**: Each state handler focuses on a specific aspect of the generation process
2. **Easier Debugging**: State transitions are explicit and can be logged
3. **Maintainability**: Adding new features or modifying behavior is easier when logic is organized by state
4. **Error Handling**: Centralized error handling with clear state transitions
5. **Extensibility**: New states can be added easily without affecting existing logic

## Comparison with Original Implementation

This state machine version reorganizes the original `main.cpp` logic without changing the core functionality:

- Same command-line interface and parameters
- Same performance characteristics
- Same output format and behavior
- Identical functionality for all use cases (text generation, chat, interactive mode)

The main difference is in code organization and maintainability, not in end-user functionality.