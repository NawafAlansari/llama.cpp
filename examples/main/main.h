#include "common.h"

#include "console.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


// Function to parse command-line arguments
gpt_params parse_command_line_arguments(int argc, char** argv);

// Function to initialize logging
void initialize_logging(int argc, char ** argv);

// Function to load and initialize the model
std::tuple<llama_model*, llama_context*, llama_context*> load_and_initialize_model(gpt_params& params);
// Function to load session from file
std::vector<llama_token> load_session_from_file(llama_context* ctx, gpt_params &params);

// Function to tokenize the prompt
std::vector<llama_token> tokenize_prompt(const std::string& prompt);

// Function to tokenize the antiprompts
std::vector<std::vector<llama_token>> tokenize_antiprompts(const std::vector<std::string>& antiprompts);

// Function to initialize the sampling context
llama_sampling_context* initialize_sampling_context(const llama_sampling_params& sampling_params);

// Function to perform group attention context extension
void perform_group_attention_context_extension(llama_context* ctx, int n_ctx, int n_batch);

// Function to perform infinite context shifting
void perform_infinite_context_shifting(llama_context* ctx, int n_ctx, int n_batch);

// Function to reuse matching prefix from session
std::vector<llama_token> reuse_matching_prefix_from_session(llama_context* ctx, const std::vector<llama_token>& session_tokens, const std::vector<llama_token>& input_tokens);

// Function to evaluate tokens in context
void evaluate_tokens_in_context(llama_context* ctx, const std::vector<llama_token>& tokens, int n_past);

// Function to evaluate tokens in guidance context
void evaluate_tokens_in_guidance_context(llama_context* guidance_ctx, const std::vector<llama_token>& tokens, int n_past);

// Function to process user input
std::vector<llama_token> process_user_input(const std::string& input_prefix, const std::string& input_suffix);

// Function to sample the next token
llama_token sample_next_token(llama_sampling_context* sampling_ctx);

// Function to display the generated text
void display_generated_text(const std::string& generated_text);

// Function to save session to file
void save_session_to_file(const std::string& session_file_path, const std::vector<llama_token>& session_tokens);

// Function to print timing information
void print_timing_information(const llama_model* model);

// Function to write log file
void write_log_file(const gpt_params& params, const llama_model* model, const std::vector<llama_token>& input_tokens, const std::vector<llama_token>& output_tokens);

// Function to clean up and free resources
void cleanup_and_free_resources(llama_model* model, llama_context* ctx, llama_sampling_context* sampling_ctx);