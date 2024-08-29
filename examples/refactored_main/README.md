# Refactored Main

This directory contains a refactored version of the main.cpp file from the llama.cpp project. The refactoring aims to improve code organization, readability, and maintainability by separating concerns into different modules.

## File Structure

- `CMakeLists.txt`: CMake configuration file for building the refactored project.
- `refactored_main.cpp`: The main entry point of the application, orchestrating the other modules.
- `config.cpp` and `config.h`: Handles configuration and parameter parsing.
- `model_manager.cpp` and `model_manager.h`: Manages the llama model loading and operations.
- `session_manager.cpp` and `session_manager.h`: Manages chat sessions and token caching.
- `input_handler.cpp` and `input_handler.h`: Handles user input and prompt processing.
- `output_handler.cpp` and `output_handler.h`: Manages output generation and display.
- `error_handler.cpp` and `error_handler.h`: Provides error handling and logging functionality.

## Refactoring Goals

1. Improve code organization by separating different functionalities into modules.
2. Enhance readability by breaking down the monolithic main.cpp into smaller, focused files.
3. Increase maintainability by reducing coupling between different parts of the code.
4. Facilitate easier testing and debugging of individual components.

## Design Plan and Structure

The refactored project follows a modular design approach, with each module responsible for a specific aspect of the application. The main components and their interactions are as follows:

1. **Config**: Central configuration structure that holds all parameters and settings.
   - Implements parsing of command-line arguments and configuration files.
   - Provides access to configuration data for other modules.

2. **Model Manager**: Handles the lifecycle of the llama model.
   - Responsible for loading and initializing the model.
   - Manages model-related operations and queries.

3. **Session Manager**: Manages chat sessions and token caching.
   - Implements session saving and loading functionality.
   - Handles token caching for improved performance.

4. **Input Handler**: Processes user input and manages prompts.
   - Handles different input modes (interactive, file-based, etc.).
   - Tokenizes input for model processing.

5. **Output Handler**: Generates and displays model output.
   - Manages the generation of text based on model predictions.
   - Handles formatting and display of output to the user.

6. **Error Handler**: Centralized error management and logging.
   - Provides consistent error handling across modules.
   - Implements logging functionality for debugging and monitoring.

The main application flow is coordinated in `refactored_main.cpp`, which initializes and orchestrates the interactions between these modules.

### Key Data Structures

- `Config`: A struct that encapsulates all configuration parameters.
- `llama_model` and `llama_context`: Core structures from the llama.cpp library for model representation.
- `std::vector<llama_token>`: Used for storing tokenized input and output.
- Custom classes for each module (e.g., `ModelManager`, `SessionManager`) that encapsulate module-specific data and methods.

## Building and Running

To build the refactored project:

1. Navigate to the `examples/refactored_main` directory.
2. Run `cmake .` to generate the build files.
3. Run `make` to compile the project.
4. Execute the resulting binary to run the refactored application.

Note: Ensure that you have the necessary dependencies installed, including the llama.cpp library.


## Detailed To-Do List

1. Set up the project structure:
   - [X] Create all necessary .cpp and .h files for each module
   - [X] Update CMakeLists.txt to include all new files

2. Implement the Config module:
   - [ ] Define the Config struct in config.h
   - [ ] Implement command-line argument parsing in config.cpp
   - [ ] Add methods for accessing configuration data

3. Develop the Model Manager:
   - [ ] Create ModelManager class with methods for loading and initializing the model
   - [ ] Implement model querying and interaction functions

4. Build the Session Manager:
   - [ ] Develop SessionManager class with session saving/loading capabilities
   - [ ] Implement token caching functionality

5. Create the Input Handler:
   - [ ] Design InputHandler class to process various input modes
   - [ ] Implement tokenization of user input

6. Construct the Output Handler:
   - [ ] Develop OutputHandler class for text generation and display
   - [ ] Implement formatting and user output functions

7. Establish the Error Handler:
   - [ ] Create ErrorHandler class for centralized error management
   - [ ] Implement logging functionality

8. Refactor main.cpp into refactored_main.cpp:
   - [ ] Initialize and coordinate all modules
   - [ ] Implement the main application flow

9. Testing and Debugging:
   - [ ] Write unit tests for each module
   - [ ] Perform integration testing
   - [ ] Debug and fix any issues

10. Documentation and Code Review:
    - [ ] Update README.md with final project structure and instructions
    - [ ] Document all classes and important functions
    - [ ] Conduct a thorough code review

11. Performance Optimization:
    - [ ] Profile the refactored code
    - [ ] Optimize critical paths and resource usage

12. Final Testing and Validation:
    - [ ] Ensure all original functionality is preserved
    - [ ] Validate against the original main.cpp behavior

## Tips and Advice

1. Start with a skeleton implementation of each module, focusing on the interface first. This will help you see how the pieces fit together before diving into the details.

2. Use consistent naming conventions and coding style across all modules to maintain readability.

3. Implement robust error handling from the beginning. It's easier to add it as you go rather than retrofitting it later.

4. Make extensive use of C++11 features like smart pointers, move semantics, and lambda functions to write more efficient and safer code.

5. Keep the interfaces between modules as clean and minimal as possible to reduce coupling.

6. Use dependency injection where appropriate to make your code more testable and flexible.

7. Write unit tests for each module as you develop them. This will help catch bugs early and ensure each component works as expected in isolation.

8. Use version control (git) effectively. Commit often and use meaningful commit messages. This will help you track changes and revert if necessary.

9. Don't try to optimize prematurely. Get the basic functionality working first, then profile and optimize.

10. Document as you go. Writing documentation helps clarify your thinking and makes it easier for others (or future you) to understand the code.

11. Consider using static analysis tools to catch potential issues early in the development process.

12. Regularly build and test the entire project to catch integration issues early.

13. Use TODO comments for minor tasks or improvements you think of while working on other parts.

14. Keep the original main.cpp as a reference to ensure you're not missing any functionality in the refactored version.

15. Consider using design patterns where appropriate, but don't over-engineer. Keep it simple where possible.

## Design Patterns to Consider

1. Singleton: For the Config and ErrorHandler modules to ensure a single, global point of access.
2. Factory Method: For creating different types of input or output handlers based on configuration.
3. Observer: To notify different parts of the application about changes in the model or session state.
4. Strategy: For implementing different tokenization or text generation algorithms.
5. Command: To encapsulate and parameterize different user commands or model operations.
6. Facade: To provide a simplified interface to the complex subsystem of the llama model.
7. Adapter: To make the llama.cpp library compatible with the new modular structure.
8. Decorator: To add additional functionality to the input or output handlers dynamically.
9. Proxy: To control access to the resource-intensive llama model.
10. Template Method: To define the skeleton of the text generation algorithm in the OutputHandler.
