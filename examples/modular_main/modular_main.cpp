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
static std::vector<llama_token> g_tokens;
static std::ostringstream g_output_ss; 
static std::vector<llama_token> g_output_tokens;
static bool is_interacting = false; 


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
        
    std::cout << "We are here" << std::endl;

    checkContextSize(params, ctx); 

    logSystemInfo(params); 

    if (params.mem_test){
        testMemoryUsage(params, ctx, model); 
        return 0;
    }

    if (params.export_cgraph){
        exportCgraph(ctx);
        return 0; 
    }

    std::string path_session = params.path_prompt_cache; 
    std::vector<llama_token> session_tokens;
    std::vector<llama_token> embd_inp; 
    const bool add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM;


    session_tokens = loadSavedSession(ctx, path_session);
    embd_inp = initializeInput(ctx, params, &embd_inp, &session_tokens, add_bos);
    
    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;

    if (ctx_guidance) {
        std::tie(guidance_inp, guidance_offset, original_prompt_len) = tokenizeNegativePrompt(ctx_guidance, ctx, params.cfg_negative_prompt, add_bos, params.prompt);
    }

    if (!checkTokenLength(embd_inp, ctx)) {
        return 1;
    }

    recalculateCachedLogits(session_tokens, embd_inp);

    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct) {
        params.n_keep = (int)embd_inp.size();
    }

    // Tokenize prefix & suffix for instruct mode
    const auto inp_pfx = tokenizeInstructPrefix(ctx, add_bos);
    const auto inp_sfx = tokenizeInstructSuffix(ctx);

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

    std::tie(grammar, parsed_grammar) = setupGrammar(ctx, params.grammar);

    // Setting up interactive mode details
    setupInteractiveModeDetails(params);

    // Initializing variables for token management and main loop
    std::vector<llama_token> embd, embd_guidance, last_tokens, candidates;
    int n_past, n_remain, n_consumed, n_session_consumed, n_past_guidance;
    initializeTokenManagement(ctx, params, embd, embd_guidance, last_tokens, candidates, n_past, n_remain, n_consumed, n_session_consumed, n_past_guidance);

    // Preparing for main generation loop
    std::vector<int> input_tokens, output_tokens;
    std::ostringstream output_ss;
    prepareForMainLoop(ctx, params, input_tokens, output_tokens, output_ss);

    // Eneter the main loop 
    int status = executeMainLoop(ctx, params, embd_inp, ctx_guidance, inp_pfx, inp_sfx, grammar, parsed_grammar);

    cleanupAndExit(ctx, params, model, input_tokens, output_ss, output_tokens, ctx_guidance, grammar);
    return 0;

}


