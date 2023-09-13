#include "helpers.h"
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
    const std::vector<llama_token> & output_tokens
) {
    if (params.logdir.empty()) { 
        return;
    }
    const std::string timestamp = get_sortable_timestamp(); 

    const bool sucess = create_directory_with_parents(params.logdir); 
    if(!sucess){ 
        fprintf(stderr, "%s: warning: failed to create logdir %s", "cannot write logfile\n",
         __func__, params.logdir.c_str());
         return;
    }

    const std::string logfile_path = params.logdir + "/" + timestamp + ".yml";
    FILE * logfile = fopen(logfile_path.c_str(), "w");

    if (logfile == NULL) {
        fprintf(stderr, "%s: failed to open logfile %s\n", __func__, logfile_path.c_str());
         return;
    }

    fprintf(logfile, "binary: modular_main\n");
    char model_desc[128]; 
    llama_model_desc(model, model_desc, sizeof(model_desc)); 
    dump_non_result_info_yaml(logfile, params, ctx, timestamp, input_tokens, model_desc);

    fprintf(logfile, "\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "# Generation Results #\n");
    fprintf(logfile, "######################\n");
    fprintf(logfile, "\n");

    dump_string_yaml_multiline(logfile, "output", output.c_str()); 
    dump_vector_int_yaml(logfile, "output_tokens", output_tokens); 
    
    llama_dump_timing_info_yaml(logfile, ctx);
    fclose(logfile); 
} 


bool initializeParams(int argc, char *argv[], gpt_params &params)
{
    return gpt_params_parse(argc, argv, params); 
    
}

void configureLogging(int argc, char *argv[])
{

}

void configureConsole(const gpt_params &params)
{
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup();});

}

bool checkAndAdjustParams(gpt_params &params)
{
    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 10000.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
         params.seed = time(NULL); 
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed)

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    return 1; 
}

void initializeBackend(int numa)
{
    LOG("%s: llama backend init\n", __func__);
    llama_backend_init(numa);
}

bool initializeModelAndContext(gpt_params &params, llama_model **model, llama_context **ctx, llama_context **ctx_guidance)
{
    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(*model, *ctx) = llama_init_from_gpt_params(params);
    if (params.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        *ctx_guidance = llama_new_context_with_model(*model, lparams);
    }


    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return false;
    }



    return true;
}


void loadModelAndApplyLora(const gpt_params &params, llama_model *model, llama_context *ctx, llama_context **ctx_guidance)
{

}


void checkContextSize(gpt_params &params, llama_context *ctx)
{
    const int n_ctx_train = llama_n_ctx_train(ctx);
    if (params.n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    } else if (params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

}


void logSystemInfo(gpt_params &params)
{
    {
        LOG_TEE("\n");
        LOG_TEE("system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }
}


void testMemoryUsage(const gpt_params &params, llama_context *ctx, llama_model *model)
{
    {
        LOG_TEE("%s: testing memory usage for n_batch = %d, n_ctx = %d\n", __func__, params.n_batch, params.n_ctx);

        const std::vector<llama_token> tmp(params.n_batch, llama_token_bos(ctx));
        llama_eval(ctx, tmp.data(), tmp.size(), params.n_ctx, params.n_threads);
    }

    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
}


void exportCgraph(llama_context *ctx, llama_model *model)
{
    llama_eval_export(ctx, "llama.ggml");
    llama_free(ctx);
    llama_free_model(model);

}


bool loadSavedSession(llama_context *ctx, gpt_params &params, const std::string &path_session, std::vector<llama_token> &session_tokens)
{
    if (!path_session.empty()){
        LOG_TEE("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());

        // fopen to check for existing session
        FILE * fp = std::fopen(path_session.c_str(), "rb");
        if (fp != NULL) {
            std::fclose(fp);

            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_TEE("%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
                return false;
            }
            session_tokens.resize(n_token_count_out);
            llama_set_rng_seed(ctx, params.seed);

            LOG_TEE("%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
        } else {
            LOG_TEE("%s: session file does not exist, will create\n", __func__);
        }
    }

    return true; 
    
}


std::vector<llama_token> initializeInput(llama_context *ctx, const gpt_params &params, std::vector<llama_token> session_tokens, bool add_bos)
{
    LOG("add_bos: %d\n", add_bos);

    std::vector<llama_token> embedding_input;
    if (params.interactive_first || params.instruct || !params.prompt.empty() || session_tokens.empty()) {
        LOG("tokenize the prompt\n");
        embedding_input = ::llama_tokenize(ctx, params.prompt, add_bos);
    } else {
        LOG("use session tokens\n");
        embedding_input = session_tokens;
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embedding_input));

    // Should not run without any tokens
    if (embedding_input.empty()) {
        embedding_input.push_back(llama_token_bos(ctx));
        LOG("embedding_input was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embedding_input));
    }

    return embedding_input;
}


std::vector<llama_token> tokenizePrompt(llama_context *ctx, const std::string &prompt, bool add_bos)
{
    return std::vector<llama_token>();
}


std::tuple<std::vector<llama_token>, int, int> tokenizeNegativePrompt(llama_context *ctx_guidance, llama_context *ctx, const std::string &negative_prompt, bool add_bos, const std::string &original_prompt)
{
    std::vector<llama_token> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;

    LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(negative_prompt));

    guidance_inp = ::llama_tokenize(ctx_guidance, negative_prompt, add_bos);
    LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp));

    std::vector<llama_token> original_inp = ::llama_tokenize(ctx, original_prompt, add_bos);
    LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp));

    original_prompt_len = original_inp.size();
    guidance_offset = (int)guidance_inp.size() - original_prompt_len;
    LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
    LOG("guidance_offset:     %s", log_tostr(guidance_offset));

    return std::make_tuple(guidance_inp, guidance_offset, original_prompt_len);
}   


bool checkTokenLength(const std::vector<llama_token> &embd_inp, llama_context *ctx)
{
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return false;
    }

    return true;
}


size_t recalculateCachedLogits(gpt_params &params, std::vector<llama_token> &session_tokens, const std::vector<llama_token> &embd_inp)
{
    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_TEE("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }
    }

    LOGLN(
            "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
            log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }
    return n_matching_session_tokens;
}


std::vector<llama_token> tokenizeInstructPrefix(llama_context *ctx, bool add_bos)
{
        return ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos);

}


std::vector<llama_token> tokenizeInstructSuffix(llama_context *ctx)
{
    return ::llama_tokenize(ctx, "\n\n### Response:\n\n",false);
}

void setupInstructMode(gpt_params &params, const std::vector<llama_token> &inp_pfx, const std::vector<llama_token> &inp_sfx)
{
    if (params.instruct) {
        params.interactive_first = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }
}

void setupInteractiveMode(gpt_params &params)
{
    if (params.interactive_first) {
        params.interactive = true;
    }
}

void logVerbosePromptsAndParams(llama_context *ctx, const gpt_params &params, const std::vector<llama_token> &embd_inp, const std::vector<llama_token> &guidance_inp, llama_context *ctx_guidance){
    if (params.verbose_prompt) {
        LOG_TEE("\n");
        LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }
    }
    if (ctx_guidance) {
            LOG_TEE("\n");
            LOG_TEE("%s: negative prompt: '%s'\n", __func__, params.cfg_negative_prompt.c_str());
            LOG_TEE("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int) guidance_inp.size(); i++) {
                LOG_TEE("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
            }
        }

        if (params.n_keep > 0) {
        LOG_TEE("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_TEE("'\n");
        }
        LOG_TEE("\n");
        LOG_TEE("logVerbosePromptsAndParams: params.n_keep = %d\n", params.n_keep);
}




void logGenerationParams(llama_context *ctx, const gpt_params &params)
{
    const int n_ctx = llama_n_ctx(ctx);

    LOG_TEE("sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
            params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

}

bool setupGrammar(llama_context *ctx, gpt_params &params, llama_grammar *grammar, grammar_parser::parse_state &parsed_grammar)
{
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return false;
        }
        LOG_TEE("%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        LOG_TEE("\n");

        {
            auto it = params.logit_bias.find(llama_token_eos(ctx));
            if (it != params.logit_bias.end() && it->second == -INFINITY) {
                LOG_TEE("%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
            }
        }

        std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
        grammar = llama_grammar_init(
            grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    }
    return true; 
}

bool setupInteractiveModeDetails(const gpt_params &params)
{
     if (params.interactive) {
        const char *control_message;
        if (params.multiline_input) {
            control_message = " - To return control to LLaMa, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to LLaMa.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_TEE("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_TEE(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_TEE(       "%s\n", control_message);

        return params.interactive_first;
    }

    return false; 
}

void initializeTokenManagement(llama_context *ctx, const gpt_params &params, std::vector<llama_token> &embd, std::vector<llama_token> &embd_guidance, std::vector<llama_token> &last_tokens, std::vector<llama_token> &candidates, int &n_past, int &n_remain, int &n_consumed, int &n_session_consumed, int &n_past_guidance)
{
}

void prepareForMainLoop(llama_context *ctx, const gpt_params &params, std::vector<int> &input_tokens, std::vector<int> &output_tokens, std::ostringstream &output_ss)
{
}

int executeMainLoop(llama_context *ctx, const gpt_params &params, std::vector<llama_token> &embd_inp, llama_context *ctx_guidance, const std::vector<llama_token> &inp_pfx, const std::vector<llama_token> &inp_sfx, llama_grammar *&grammar, grammar_parser::parse_state &parsed_grammar)
{
    return 0;
}

void cleanupAndExit(llama_context *ctx, const gpt_params &params, llama_model *model, std::vector<int> input_tokens, std::ostringstream &output_ss, std::vector<int> output_tokens, llama_context *ctx_guidance, llama_grammar *grammar)
{
}
