// helpers.h

#pragma once

#include "llama.h"  // Assuming you have a file named "llama.h" that contains the definition of gpt_params
#include "common.h"
#include "grammar-parser.h"
#include "build-info.h"
#include "console.h"

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

void write_logfile(
    const llama_context * ctx, const gpt_params & params, const llama_model * model,
    const std::vector<llama_token> & input_tokens, const std::string & output,
    const std::vector<llama_token> & output_tokens);

/**
 * @brief Initializes the gpt_params structure.
 * 
 * This function handles the initialization of the gpt_params structure. It parses
 * the command-line arguments provided to the program and loads the relevant settings
 * and configurations into the gpt_params structure.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @param params Reference to the gpt_params structure to be initialized.
 * @return true if parameters were successfully initialized, false otherwise.
 */
bool initializeParams(int argc, char *argv[], gpt_params& params);

/**
 * @brief Sets up the logging system.
 * 
 * Configures the logging target and logs the start of the program. If logging is enabled,
 * it also dumps the provided command-line arguments to the log for future reference.
 * 
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 */
void configureLogging(int argc, char *argv[]);

/**
 * @brief Initializes the console for user interaction.
 * 
 * Sets up the console based on the configurations provided in the gpt_params structure.
 * This includes deciding whether to use color in the console output. Additionally, a cleanup
 * function is registered to be executed upon program termination.
 * 
 * @param params The gpt_params structure containing console-related settings.
 */
void configureConsole(const gpt_params& params);

/**
 * @brief Performs various checks on the gpt_params structure.
 * 
 * Checks the values of certain parameters and adjusts them if necessary. This includes
 * verifying whether certain tools (like 'perplexity' or 'embedding') should be used and
 * issuing warnings if specific parameters deviate from their default values. Also, handles
 * the generation of random prompts if required.
 * 
 * @param params The gpt_params structure containing the parameters to be checked.
 * @return true if checks were successful, false if the program should terminate.
 */
bool checkAndAdjustParams(gpt_params& params);


/**
 * Initializes the llama backend with the provided NUMA configuration.
 * 
 * @param numa: The NUMA configuration for the llama backend.
 */
void initializeBackend(int numa);

/**
 * Initializes pointers for the model and context based on the provided parameters.
 * 
 * @param params: Parameters for the model and context initialization.
 * @param model: Pointer to the model structure.
 * @param ctx: Pointer to the primary context.
 * @param ctx_guidance: Pointer to the guidance context, if any.
 */
bool initializeModelAndContext(gpt_params& params, llama_model** model, llama_context** ctx, llama_context** ctx_guidance);


/**
 * Loads the machine learning model based on the provided parameters and applies the Lora adapter if necessary.
 * 
 * @param params: Parameters that might influence how the model is loaded.
 * @param model: Pointer to the model structure.
 * @param ctx: Pointer to the primary context.
 * @param ctx_guidance: Pointer to the guidance context, if any.
 */
void loadModelAndApplyLora(const gpt_params& params, llama_model* model, llama_context* ctx, llama_context** ctx_guidance);


/**
 * Checks the context size based on the provided parameters and adjusts it if necessary.
 * 
 * @param params: Parameters that specify the desired context size.
 * @param ctx: Pointer to the primary context.
 */
void checkContextSize(gpt_params& params, llama_context* ctx);


/**
 * Logs system-related information, such as the number of threads in use.
 * 
 * @param params: Parameters that might contain system-related configurations.
 *
 */
void logSystemInfo(gpt_params& params);


/**
 * Tests the memory usage for given batch and context parameters. It's mainly for diagnostic purposes.
 * 
 * @param params: Parameters that specify the batch and context size.
 * @param ctx: Pointer to the primary context.
 * @param model: Pointer to the model structure.
 */
void testMemoryUsage(const gpt_params& params, llama_context* ctx, llama_model* model);


/**
 * Exports the computational graph (cgraph) of the model.
 * 
 * @param ctx: Pointer to the primary context.
 */
void exportCgraph(llama_context* ctx, llama_model *model);

/**
 * Attempts to load a saved session from the provided path.
 * 
 * @param ctx: Pointer to the primary context.
 * @param path_session: Path to the saved session file.
 * @return: A vector containing session tokens.
 */
bool loadSavedSession(llama_context* ctx, gpt_params &params, const std::string& path_session, std::vector<llama_token> &session_tokens);


/**
 * Determine the main input tokens based on the provided prompt or from a loaded session.
 * 
 * @param ctx: Pointer to the primary context.
 * @param params: The GPT parameters containing information about the prompt.
 * @param embd_inp: Pointer to the main input token vector.
 * @param session_tokens: Pointer to the loaded session tokens vector.
 * @param add_bos: Flag to indicate if a beginning-of-sentence token should be added.
 * @return: A vector containing the main input tokens, either tokenized from the prompt or taken from the session.
 */
std::vector<llama_token> initializeInput(llama_context* ctx, const gpt_params& params, std::vector<llama_token> session_tokens, bool add_bos);


/**
 * Tokenizes the provided prompt.
 * 
 * @param ctx: Pointer to the primary context.
 * @param prompt: The text prompt to tokenize.
 * @param add_bos: Flag to indicate if a beginning-of-sentence token should be added.
 * @return: A vector containing the tokens of the prompt.
 */
std::vector<llama_token> tokenizePrompt(llama_context* ctx, const std::string& prompt, bool add_bos);




/**
 * Tokenizes the negative prompt for guidance.
 * 
 * @param ctx_guidance: Pointer to the guidance context.
 * @param ctx: Pointer to the primary context.
 * @param negative_prompt: The negative text prompt to tokenize.
 * @param add_bos: Flag to indicate if a beginning-of-sentence token should be added.
 * @param original_prompt: The original text prompt.
 * @return: A tuple containing the tokens of the negative prompt, guidance offset, and original prompt length.
 */
std::tuple<std::vector<llama_token>, int, int> tokenizeNegativePrompt(llama_context* ctx_guidance, llama_context* ctx, const std::string& negative_prompt, bool add_bos, const std::string& original_prompt);


/**
 * Tokenizes the negative prompt for guidance.
 * 
 * @param ctx_guidance: Pointer to the guidance context.
 * @param ctx: Pointer to the primary context.
 * @param negative_prompt: The negative text prompt to tokenize.
 * @param add_bos: Flag to indicate if a beginning-of-sentence token should be added.
 * @param original_prompt: The original text prompt.
 * @return: A tuple containing the tokens of the negative prompt, guidance offset, and original prompt length.
 */
std::tuple<std::vector<llama_token>, int, int> tokenizeNegativePrompt(llama_context* ctx_guidance, llama_context* ctx, const std::string& negative_prompt, bool add_bos, const std::string& original_prompt);

/**
 * Checks if the tokenized prompt is within the permissible limits of the model's context size.
 * 
 * @param embd_inp: The tokenized prompt.
 * @param ctx: Pointer to the primary context.
 * @return: A boolean indicating if the token length is acceptable (true) or not (false).
 */
bool checkTokenLength(const std::vector<llama_token>& embd_inp, llama_context* ctx);


/**
 * Recalculates the cached logits based on the session tokens and the tokenized prompt.
 * 
 * @param session_tokens: The tokens from the loaded session.
 * @param embd_inp: The tokenized prompt.
 */
size_t recalculateCachedLogits(gpt_params &params, std::vector<llama_token>& session_tokens, const std::vector<llama_token>& embd_inp);


/**
 * Tokenizes the instruction prefix.
 * 
 * @param ctx: Pointer to the primary context.
 * @param add_bos: Flag to indicate if a beginning-of-sentence token should be added.
 * @return: A vector containing the tokens of the instruction prefix.
 */
std::vector<llama_token> tokenizeInstructPrefix(llama_context* ctx, bool add_bos);

/**
 * Tokenizes the instruction suffix.
 * 
 * @param ctx: Pointer to the primary context.
 * @return: A vector containing the tokens of the instruction suffix.
 */
std::vector<llama_token> tokenizeInstructSuffix(llama_context* ctx);

/**
 * Sets up the parameters and variables related to the instruction mode.
 * 
 * @param params: The GPT parameters.
 * @param inp_pfx: The tokenized instruction prefix.
 * @param inp_sfx: The tokenized instruction suffix.
 */
void setupInstructMode(gpt_params& params, const std::vector<llama_token>& inp_pfx, const std::vector<llama_token>& inp_sfx);

/**
 * Initializes and sets up the interactive mode based on given parameters.
 * 
 * @param params: The GPT parameters.
 */
void setupInteractiveMode(gpt_params& params);

/**
 * Logs detailed information about the prompts and tokens, especially when verbose mode is enabled.
 * 
 * @param ctx: Pointer to the primary context.
 * @param params: The GPT parameters.
 * @param embd_inp: The main input token vector.
 * @param guidance_inp: The tokenized guidance input vector.
 * @param ctx_guidance: Pointer to the guidance context.
 */
void logVerbosePromptsAndParams(llama_context* ctx, const gpt_params& params, const std::vector<llama_token>& embd_inp, const std::vector<llama_token>& guidance_inp, llama_context* ctx_guidance);

/**
 * Logs various settings and details when the interactive mode is enabled.
 * 
 * @param params: The GPT parameters.
 */
void logInteractiveModeDetails(const gpt_params& params);

/**
 * Logs parameters related to the token generation process.
 * 
 * @param ctx: Pointer to the primary context.
 * @param params: The GPT parameters.
 */
void logGenerationParams(llama_context* ctx, const gpt_params& params);


/**
 * Sets up the grammar using the provided grammar string.
 * 
 * @param ctx: Pointer to the primary context.
 * @param grammar_str: The grammar string to parse and set up.
 * @return: A pointer to the initialized grammar structure.
 */
bool setupGrammar(llama_context* ctx, gpt_params &params, llama_grammar * grammar, grammar_parser::parse_state &parsed_grammar);

/**
 * Sets up the details and messages for the interactive mode based on the given parameters.
 * 
 * @param params: The GPT parameters.
 */
bool setupInteractiveModeDetails(const gpt_params& params);

/**
 * Initializes various vectors and variables needed for token management in the main loop.
 * 
 * @param ctx: Pointer to the primary context.
 * @param params: The GPT parameters.
 * @param embd: Reference to the main embedding vector.
 * @param embd_guidance: Reference to the guidance embedding vector.
 * @param last_tokens: Reference to the vector of last tokens.
 * @param candidates: Reference to the vector of token candidates.
 * @param n_past: Reference to the count of past tokens.
 * @param n_remain: Reference to the remaining tokens count.
 * @param n_consumed: Reference to the consumed tokens count.
 * @param n_session_consumed: Reference to the consumed tokens from the session.
 * @param n_past_guidance: Reference to the past guidance tokens count.
 */
void initializeTokenManagement(llama_context* ctx, const gpt_params& params, 
                               std::vector<llama_token>& embd, std::vector<llama_token>& embd_guidance, 
                               std::vector<llama_token>& last_tokens, std::vector<llama_token>& candidates, 
                               int& n_past, int& n_remain, int& n_consumed, int& n_session_consumed, int& n_past_guidance);

/**
 * Prepares the necessary vectors and streams for the main token generation loop.
 * 
 * @param ctx: Pointer to the primary context.
 * @param params: The GPT parameters.
 * @param input_tokens: Reference to the input tokens vector.
 * @param output_tokens: Reference to the output tokens vector.
 * @param output_ss: Reference to the output string stream.
 */
void prepareForMainLoop(llama_context* ctx, const gpt_params& params, 
                        std::vector<int>& input_tokens, std::vector<int>& output_tokens, std::ostringstream& output_ss);


/**
 * Executes the main sampling and interaction loop.
 * 
 * @param ctx: Pointer to the primary context.
 * @param params: The GPT parameters.
 * @param embd_inp: The input tokens to process.
 * @param ctx_guidance: Pointer to the guidance context.
 * @param n_ctx: The maximum context size.
 * @param inp_pfx: Prefix for the instruction mode.
 * @param inp_sfx: Suffix for the instruction mode.
 * @param grammar: Pointer to the grammar structure.
 * @param parsed_grammar: The parsed grammar state.
 * @return: Status of the execution.
 */
int executeMainLoop(llama_context* ctx, const gpt_params& params, std::vector<llama_token>& embd_inp, 
                    llama_context* ctx_guidance, 
                    const std::vector<llama_token>& inp_pfx, const std::vector<llama_token>& inp_sfx,
                    struct llama_grammar*& grammar, grammar_parser::parse_state& parsed_grammar);

/**
 * Performs cleanup operations after main loop execution. 
 * This includes saving the session, printing timings, writing to logfile, releasing memory, 
 * and ending logs.
 * 
 * @param ctx: Pointer to the main llama context.
 * @param params: Struct containing various parameters used in the application.
 * @param model: Pointer to the llama model.
 * @param input_tokens: Vector of tokens that were input during execution.
 * @param output_ss: String stream containing the output.
 * @param output_tokens: Vector of tokens that were output during execution.
 * @param ctx_guidance: Pointer to the guidance context.
 * @param grammar: Pointer to the grammar used.
 */
void cleanupAndExit(llama_context* ctx, const gpt_params& params, llama_model* model, 
                    std::vector<int> input_tokens, std::ostringstream& output_ss, 
                    std::vector<int> output_tokens, llama_context* ctx_guidance, 
                    llama_grammar* grammar);
