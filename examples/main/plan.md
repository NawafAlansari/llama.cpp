# Refactoring Plan for `llama.cpp`

## Setup
- [X] Create a new branch in the version control system (e.g., Git) for the refactoring process.

## Command-line Argument Parsing
- [ ] Extract the code responsible for parsing command-line arguments into a separate function `parse_command_line_arguments()`.
- [ ] Modify `parse_command_line_arguments()` to take `argc` and `argv` as parameters and return the initialized `gpt_params` struct.
- [ ] Update the `main()` function to call `parse_command_line_arguments()` with the appropriate arguments.
- [ ] Compile and test the code to ensure the argument parsing functionality remains intact.

## Model Loading and Initialization
- [ ] Extract the code responsible for loading and initializing the Llama model into a separate function `load_and_initialize_model()`.
- [ ] Modify `load_and_initialize_model()` to take the `gpt_params` struct as a parameter and return the loaded `llama_model` and initialized `llama_context` objects.
- [ ] Update the `main()` function to call `load_and_initialize_model()` with the appropriate arguments.
- [ ] Compile and test the code to ensure the model loading and initialization functionality works correctly.

## Tokenization Functions
- [ ] Extract the code responsible for tokenizing the prompt and antiprompts into separate functions `tokenize_prompt()` and `tokenize_antiprompts()`.
- [ ] Modify `tokenize_prompt()` and `tokenize_antiprompts()` to take the respective input strings as parameters and return the tokenized representations.
- [ ] Update the `main()` function to call these functions with the appropriate arguments.
- [ ] Compile and test the code to ensure the tokenization functionality remains intact.

## Sampling Context Initialization
- [ ] Extract the code responsible for initializing the sampling context into a separate function `initialize_sampling_context()`.
- [ ] Modify `initialize_sampling_context()` to take the `llama_sampling_params` struct as a parameter and return the initialized `llama_sampling_context` object.
- [ ] Update the `main()` function to call `initialize_sampling_context()` with the appropriate arguments.
- [ ] Compile and test the code to ensure the sampling context initialization works correctly.

## Main Generation Loop
- [ ] Break down the main generation loop into smaller functions based on the pseudocode.
- [ ] Implement functions like `perform_group_attention_context_extension()`, `perform_infinite_context_shifting()`, `reuse_matching_prefix_from_session()`, `evaluate_tokens_in_context()`, `process_user_input()`, `sample_next_token()`, and `display_generated_text()`.
- [ ] Modify these functions to take the necessary parameters and return the appropriate values.
- [ ] Update the main generation loop in the `main()` function to call these functions with the appropriate arguments.
- [ ] Compile and test the code incrementally to ensure the generation loop functionality remains intact.

## Session Management and Logging
- [ ] Extract the code responsible for loading and saving session state into separate functions `load_session_from_file()` and `save_session_to_file()`.
- [ ] Modify `load_session_from_file()` and `save_session_to_file()` to take the necessary parameters, such as the session file path and session tokens.
- [ ] Extract the code for printing timing information and writing log files into separate functions `print_timing_information()` and `write_log_file()`.
- [ ] Update the `main()` function to call these functions at the appropriate places.
- [ ] Compile and test the code to ensure the session management and logging functionality works correctly.

## Resource Cleanup
- [ ] Extract the code responsible for cleaning up and freeing allocated resources into a separate function `cleanup_and_free_resources()`.
- [ ] Modify `cleanup_and_free_resources()` to take the necessary objects, such as `llama_model`, `llama_context`, and `llama_sampling_context`, as parameters.
- [ ] Update the `main()` function to call `cleanup_and_free_resources()` with the appropriate arguments before exiting.
- [ ] Compile and test the code to ensure the resource cleanup functionality works correctly.

## Testing and Validation
- [ ] Perform thorough testing of the refactored code with various inputs and scenarios.
- [ ] Compare the output and behavior of the refactored code with the original code to ensure they match.
- [ ] Address any issues or bugs that arise during testing and make necessary adjustments.

## Finalization
- [ ] Once the refactored code is thoroughly tested and validated, merge the refactoring branch into the main branch.
- [ ] Update any relevant documentation or comments to reflect the changes made during refactoring.
- [ ] Celebrate your achievement and take pride in the improved code structure and readability!


# Pseudocode
```
function main():
    parse_command_line_arguments()
    initialize_logging()

    model, context = load_and_initialize_model()
    if guidance_enabled:
        guidance_context = create_guidance_context(model)

    if session_file_exists:
        load_session_from_file()

    tokenized_prompt = tokenize_prompt(prompt)
    tokenized_antiprompts = tokenize_antiprompts(antiprompts)

    sampling_context = initialize_sampling_context()

    while (tokens_remaining > 0 or interactive_mode):
        if tokenized_prompt is not empty:
            if group_attention_enabled:
                perform_group_attention_context_extension()
            else:
                perform_infinite_context_shifting()

            reuse_matching_prefix_from_session()

            for batch in tokenized_prompt:
                evaluate_tokens_in_context(context, batch)
                if guidance_enabled:
                    evaluate_tokens_in_guidance_context(guidance_context, batch)

        if interactive_mode:
            user_input = process_user_input()
            tokenized_prompt.append(user_input)

        if no_more_input_tokens and not_interactive_mode:
            next_token = sample_next_token(sampling_context)
            tokenized_prompt.append(next_token)

        display_generated_text()

        if end_of_text_token_reached and not_interactive_mode:
            break

        if interactive_mode and tokens_remaining <= 0:
            tokens_remaining = reset_tokens_remaining()
            enter_interactive_mode()

    save_session_to_file()
    print_timing_information()
    write_log_file()

    cleanup_and_free_resources()
```