// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"

#include "console.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"
#include "helpers.h"
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

static llama_context ** g_ctx; 
static llama_model ** g_model;
static gpt_params * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream * g_output_ss; 
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting = false; 



#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting = true;
        } else {
            console::cleanup();
            printf("\n");
            llama_print_timings(*g_ctx);
            write_logfile(*g_ctx, *g_params, *g_model, *g_input_tokens, g_output_ss->str(), *g_output_tokens);
            _exit(130);
        }
    }
}
#endif


void logInteractiveModeDetails(const gpt_params &params)
{
    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

        LOG_TEE("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_TEE("Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (params.input_prefix_bos) {
            LOG_TEE("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_TEE("Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            LOG_TEE("Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }
}




int main(int argc, char** argv){

    gpt_params params; 
    g_params = &params;



    if(!initializeParams(argc, argv, params)){
        return 1; 
    }

//configure logging 
#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("main", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS


    configureConsole(params);

    if (!checkAndAdjustParams(params)) {
        return 0;
    }




    // Initialize the backend 
    initializeBackend(params.numa);



    //model and context initialization
     llama_model *model; 
     llama_context *ctx;
     llama_context * ctx_guidance = NULL;
     g_model = &model; 
     g_ctx = &ctx;
    
    if (!initializeModelAndContext(params, &model, &ctx, &ctx_guidance)) {
        return 1; 
    }




    checkContextSize(params, ctx);


    logSystemInfo(params);

    if (params.mem_test){
        testMemoryUsage(params, ctx, model); 
        return 0;
    }

    if (params.export_cgraph){
        exportCgraph(ctx, model);
        return 0; 
    }

    std::string path_session = params.path_prompt_cache; 
    std::vector<llama_token> session_tokens;
    std::vector<llama_token> embd_inp; 
    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;


    loadSavedSession(ctx, params, path_session, session_tokens);
    embd_inp = initializeInput(ctx, params, session_tokens, add_bos);

    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;

    if (ctx_guidance) {
        std::tie(guidance_inp, guidance_offset, original_prompt_len) = tokenizeNegativePrompt(ctx_guidance, ctx, params.cfg_negative_prompt, add_bos, params.prompt);
    }
    LOG_TEE("WE ARE HERE\n");


    if (!checkTokenLength(embd_inp, ctx)) {
        return 1;
    }

    size_t n_matching_session_tokens = recalculateCachedLogits(params, session_tokens, embd_inp);

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // Tokenize prefix & suffix for instruct mode
    const auto inp_pfx = tokenizeInstructPrefix(ctx, add_bos);
    const auto inp_sfx = tokenizeInstructSuffix(ctx);

    LOG("inp_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_pfx));
    LOG("inp_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_sfx));

    setupInstructMode(params, inp_pfx, inp_sfx);

    // Setup interactive mode if required
    setupInteractiveMode(params);

    // Log verbose prompts and parameters
    logVerbosePromptsAndParams(ctx, params, embd_inp, guidance_inp, ctx_guidance);

    // Log interactive mode details
    logInteractiveModeDetails(params);

    // Log generation parameters
    logGenerationParams(ctx, params);

    // Parsing and setting up the grammar
    struct llama_grammar * grammar = NULL;
    grammar_parser::parse_state parsed_grammar;

    if(!setupGrammar(ctx, params, grammar, parsed_grammar)){
        return 1; 
    }
    /*
    This part of the code is responsible for setting up the chat session managemet
    what we can do here instead is to abstract this into a structure that basically 
    is responsible for managing the session's states and parameters. 
    */

    // Setting up interactive mode details
    is_interacting = setupInteractiveModeDetails(params);

    // Initializing variables for token management and main loop
    int n_ctx = llama_n_ctx(ctx);
    std::vector<llama_token> last_tokens(n_ctx);
    std::fill(last_tokens.begin(), last_tokens.end(), 0);

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;



    
    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    const int n_vocab = llama_n_vocab(ctx);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    // Eneter the main loop 
    int status = executeMainLoop(ctx, params, embd_inp, ctx_guidance, inp_pfx, inp_sfx, grammar, parsed_grammar);

    cleanupAndExit(ctx, params, model, input_tokens, output_ss, output_tokens, ctx_guidance, grammar);
    return 0;

}


