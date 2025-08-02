// Include necessary headers
#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

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

// Define states for the state machine
enum LlamaState {
    INITIALIZING,       // Initial setup and parameter parsing
    LOADING_MODEL,      // Loading the model from file
    PREPARING_PROMPT,   // Tokenizing and preparing the initial prompt
    LOADING_SESSION,    // Loading session data if available
    GENERATING_TOKENS,  // Main token generation loop
    MANAGING_CONTEXT,   // Handling context window management
    WAITING_FOR_INPUT,  // Waiting for user input in interactive mode
    PROCESSING_INPUT,   // Processing received user input
    FINISHING,          // Final cleanup and shutdown
    ERROR               // Error state
};

// Context structure to hold all state data
struct LlamaContext {
    // Configuration
    common_params params;
    
    // Core components
    llama_model* model;
    llama_context* ctx;
    common_sampler* sampler;
    std::shared_ptr<common_chat_templates> chat_templates;
    struct ggml_threadpool* threadpool;
    struct ggml_threadpool* threadpool_batch;
    
    // Tokens and generation state
    std::vector<llama_token> input_tokens;
    std::vector<llama_token> output_tokens;
    std::vector<llama_token> session_tokens;
    std::vector<llama_token> embd_inp;
    std::vector<llama_token> embd;
    std::vector<llama_token> antiprompt_token;
    
    // State tracking
    int n_past;
    int n_consumed;
    int n_remain;
    int n_session_consumed;
    int n_ctx;
    int n_ctx_train;
    int ga_i; // group-attention index
    int ga_n; // group-attention n
    int ga_w; // group-attention w
    size_t n_matching_session_tokens;
    
    // Flags
    bool is_interacting;
    bool need_insert_eot;
    bool is_antiprompt;
    bool display;
    bool input_echo;
    bool need_to_save_session;
    bool waiting_for_first_input;
    
    // Chat components
    std::vector<common_chat_msg> chat_msgs;
    std::ostringstream output_ss;
    std::ostringstream assistant_ss;
    
    // Session path
    std::string path_session;
    
    // Error tracking
    std::string error_message;
};

// Global context for signal handlers
static LlamaContext g_ctx;

// Helper functions (similar to original code)
static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!g_ctx.is_interacting && g_ctx.params.interactive) {
            g_ctx.is_interacting = true;
            g_ctx.need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(g_ctx.ctx, g_ctx.sampler);
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());
            _exit(130);
        }
    }
}
#endif

// Forward declarations for state handlers
LlamaState handle_initializing(int argc, char** argv, LlamaContext* context);
LlamaState handle_loading_model(LlamaContext* context);
LlamaState handle_preparing_prompt(LlamaContext* context);
LlamaState handle_loading_session(LlamaContext* context);
LlamaState handle_generating_tokens(LlamaContext* context);
LlamaState handle_managing_context(LlamaContext* context);
LlamaState handle_waiting_for_input(LlamaContext* context);
LlamaState handle_processing_input(LlamaContext* context);
LlamaState handle_finishing(LlamaContext* context);

// Initialization state handler
LlamaState handle_initializing(int argc, char** argv, LlamaContext* context) {
    LOG_DBG("State: INITIALIZING\n");
    
    // Initialize common components
    common_init();
    
    // Parse command line parameters
    if (!common_params_parse(argc, argv, context->params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        context->error_message = "Failed to parse parameters";
        return ERROR;
    }
    
    // Setup console
    console::init(context->params.simple_io, context->params.use_color);
    
    // Parameter validation
    if (context->params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");
        context->error_message = "Please use the 'embedding' tool for embedding calculations";
        return ERROR;
    }
    
    if (context->params.n_ctx != 0 && context->params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        context->params.n_ctx = 8;
    }
    
    if (context->params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, context->params.rope_freq_base);
    }
    
    if (context->params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, context->params.rope_freq_scale);
    }
    
    // Initialize display flags
    context->display = context->params.display_prompt;
    context->input_echo = true;
    
    return LOADING_MODEL;
}

// Model loading state handler
LlamaState handle_loading_model(LlamaContext* context) {
    LOG_DBG("State: LOADING_MODEL\n");
    
    // Initialize LLAMA backend
    LOG_INF("%s: llama backend init\n", __func__);
    
    llama_backend_init();
    llama_numa_init(context->params.numa);
    
    // Load model and initialize context
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(context->params);
    
    context->model = llama_init.model.get();
    context->ctx = llama_init.context.get();
    
    if (context->model == NULL) {
        context->error_message = "Unable to load model";
        return ERROR;
    }
    
    context->n_ctx = llama_n_ctx(context->ctx);
    context->n_ctx_train = llama_model_n_ctx_train(context->model);
    
    if (context->n_ctx > context->n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, context->n_ctx_train, context->n_ctx);
    }
    
    // Initialize chat templates
    context->chat_templates = common_chat_templates_init(context->model, context->params.chat_template);
    
    // Set up thread pools
    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) context->params.cpuparams.n_threads);
    
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        context->error_message = "No CPU backend found";
        return ERROR;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    
    struct ggml_threadpool_params tpp_batch = ggml_threadpool_params_from_cpu_params(context->params.cpuparams_batch);
    struct ggml_threadpool_params tpp = ggml_threadpool_params_from_cpu_params(context->params.cpuparams);
    
    set_process_priority(context->params.cpuparams.priority);
    
    context->threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        context->threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!context->threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            context->error_message = "Failed to create batch threadpool";
            return ERROR;
        }
        tpp.paused = true;
    }
    
    context->threadpool = ggml_threadpool_new_fn(&tpp);
    if (!context->threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        context->error_message = "Failed to create threadpool";
        return ERROR;
    }
    
    llama_attach_threadpool(context->ctx, context->threadpool, context->threadpool_batch);
    
    // Auto-enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(context->chat_templates.get());
    if (context->params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            context->params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            context->params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }
    
    if (context->params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }
    
    // Print chat template example in conversation mode
    if (context->params.conversation_mode) {
        if (context->params.enable_chat_template) {
            if (!context->params.prompt.empty() && context->params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }
            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(context->chat_templates.get(), context->params.use_jinja).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }
    
    // Print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(context->params).c_str());
        LOG_INF("\n");
    }
    
    // Initialize sampler
    context->sampler = common_sampler_init(context->model, context->params.sampling);
    
    if (!context->sampler) {
        context->error_message = "Failed to initialize sampling subsystem";
        return ERROR;
    }
    
    LOG_INF("sampler seed: %u\n", common_sampler_get_seed(context->sampler));
    LOG_INF("sampler params: \n%s\n", context->params.sampling.print().c_str());
    LOG_INF("sampler chain: %s\n", common_sampler_print(context->sampler).c_str());
    
    // Initialize group-attention parameters
    context->ga_i = 0;
    context->ga_n = context->params.grp_attn_n;
    context->ga_w = context->params.grp_attn_w;
    
    if (context->ga_n != 1) {
        GGML_ASSERT(context->ga_n > 0 && "grp_attn_n must be positive");
        GGML_ASSERT(context->ga_w % context->ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n");
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", context->n_ctx_train, context->ga_n, context->ga_w);
    }
    LOG_INF("\n");
    
    return PREPARING_PROMPT;
}

// Preparing prompt state handler
LlamaState handle_preparing_prompt(LlamaContext* context) {
    LOG_DBG("State: PREPARING_PROMPT\n");
    
    context->path_session = context->params.path_prompt_cache;
    
    const llama_vocab * vocab = llama_model_get_vocab(context->model);
    const bool add_bos = llama_vocab_get_add_bos(vocab) && !context->params.use_jinja;
    if (!llama_model_has_encoder(context->model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }
    
    LOG_DBG("n_ctx: %d, add_bos: %d\n", context->n_ctx, add_bos);
    
    // Helper function for chat formatting
    auto chat_add_and_format = [&](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(context->chat_templates.get(), context->chat_msgs, new_msg, role == "user", context->params.use_jinja);
        context->chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };
    
    std::string prompt;
    context->waiting_for_first_input = false;
    
    // Prepare prompt based on conversation mode
    if (context->params.conversation_mode && context->params.enable_chat_template) {
        if (!context->params.system_prompt.empty()) {
            chat_add_and_format("system", context->params.system_prompt);
        }
        
        if (!context->params.prompt.empty()) {
            chat_add_and_format("user", context->params.prompt);
        } else {
            context->waiting_for_first_input = true;
        }
        
        if (!context->params.system_prompt.empty() || !context->params.prompt.empty()) {
            common_chat_templates_inputs inputs;
            inputs.use_jinja = context->params.use_jinja;
            inputs.messages = context->chat_msgs;
            inputs.add_generation_prompt = !context->params.prompt.empty();
            
            prompt = common_chat_templates_apply(context->chat_templates.get(), inputs).prompt;
        }
    } else {
        prompt = context->params.prompt;
    }
    
    // Tokenize prompt
    if (context->params.interactive_first || !prompt.empty() || context->session_tokens.empty()) {
        LOG_DBG("tokenize the prompt\n");
        context->embd_inp = common_tokenize(context->ctx, prompt, true, true);
    } else {
        LOG_DBG("use session tokens\n");
        context->embd_inp = context->session_tokens;
    }
    
    LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
    LOG_DBG("tokens: %s\n", string_from(context->ctx, context->embd_inp).c_str());
    
    // Should not run without any tokens
    if (!context->waiting_for_first_input && context->embd_inp.empty()) {
        if (add_bos) {
            context->embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(context->ctx, context->embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            context->error_message = "Input is empty";
            return ERROR;
        }
    }
    
    // Check prompt length
    if ((int) context->embd_inp.size() > context->n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) context->embd_inp.size(), context->n_ctx - 4);
        context->error_message = "Prompt is too long";
        return ERROR;
    }
    
    // Number of tokens to keep when resetting context
    if (context->params.n_keep < 0 || context->params.n_keep > (int) context->embd_inp.size()) {
        context->params.n_keep = (int)context->embd_inp.size();
    } else {
        context->params.n_keep += add_bos; // always keep the BOS token
    }
    
    // Set interactive mode flags
    if (context->params.conversation_mode) {
        if (context->params.single_turn && !context->params.prompt.empty()) {
            context->params.interactive = false;
            context->params.interactive_first = false;
        } else {
            context->params.interactive_first = true;
        }
    }
    
    if (context->params.interactive_first) {
        context->params.interactive = true;
    }
    
    // Print verbose prompt information
    if (context->params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, context->params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, context->embd_inp.size());
        for (int i = 0; i < (int) context->embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", context->embd_inp[i], common_token_to_piece(context->ctx, context->embd_inp[i]).c_str());
        }
        
        if (context->params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < context->params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(context->ctx, context->embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }
    
    // Print interactive mode information
    if (context->params.interactive) {
        const char * control_message;
        if (context->params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (context->params.conversation_mode && context->params.enable_chat_template && context->params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");
        
        context->is_interacting = context->params.interactive_first;
        
        if (!context->params.antiprompt.empty()) {
            for (const auto & antiprompt : context->params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (context->params.verbose_prompt) {
                    auto tmp = common_tokenize(context->ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(context->ctx, tmp[i]).c_str());
                    }
                }
            }
        }
        
        if (context->params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }
        
        if (!context->params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", context->params.input_prefix.c_str());
            if (context->params.verbose_prompt) {
                auto tmp = common_tokenize(context->ctx, context->params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(context->ctx, tmp[i]).c_str());
                }
            }
        }
        
        if (!context->params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", context->params.input_suffix.c_str());
            if (context->params.verbose_prompt) {
                auto tmp = common_tokenize(context->ctx, context->params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(context->ctx, tmp[i]).c_str());
                }
            }
        }
    }
    
    // Prepare single-token antiprompts
    for (const std::string & antiprompt : context->params.antiprompt) {
        auto ids = ::common_tokenize(context->ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            context->antiprompt_token.push_back(ids[0]);
        }
    }
    
    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", context->n_ctx, context->params.n_batch, context->params.n_predict, context->params.n_keep);
    
    // Initialize state variables
    context->n_past = 0;
    context->n_remain = context->params.n_predict;
    context->n_consumed = 0;
    context->n_session_consumed = 0;
    context->is_antiprompt = false;
    context->need_insert_eot = false;
    context->need_to_save_session = !context->path_session.empty() && context->n_matching_session_tokens < context->embd_inp.size();
    
    // Set the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(console::prompt);
    context->display = context->params.display_prompt;
    
    // Handle encoder models
    if (llama_model_has_encoder(context->model)) {
        int enc_input_size = context->embd_inp.size();
        llama_token * enc_input_buf = context->embd_inp.data();
        
        if (llama_encode(context->ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            context->error_message = "Failed to encode input";
            return ERROR;
        }
        
        llama_token decoder_start_token_id = llama_model_decoder_start_token(context->model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            const llama_vocab * vocab = llama_model_get_vocab(context->model);
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        
        context->embd_inp.clear();
        context->embd_inp.push_back(decoder_start_token_id);
    }
    
    return LOADING_SESSION;
}

// Loading session state handler
LlamaState handle_loading_session(LlamaContext* context) {
    LOG_DBG("State: LOADING_SESSION\n");
    
    if (!context->path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, context->path_session.c_str());
        if (!file_exists(context->path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(context->path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            context->session_tokens.resize(context->n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(context->ctx, context->path_session.c_str(), context->session_tokens.data(), context->session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, context->path_session.c_str());
                context->error_message = "Failed to load session file";
                return ERROR;
            }
            context->session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)context->session_tokens.size());
        }
    }
    
    // Debug message about similarity of saved session, if applicable
    context->n_matching_session_tokens = 0;
    if (!context->session_tokens.empty()) {
        for (llama_token id : context->session_tokens) {
            if (context->n_matching_session_tokens >= context->embd_inp.size() || id != context->embd_inp[context->n_matching_session_tokens]) {
                break;
            }
            context->n_matching_session_tokens++;
        }
        if (context->params.prompt.empty() && context->n_matching_session_tokens == context->embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (context->n_matching_session_tokens >= context->embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (context->n_matching_session_tokens < (context->embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, context->n_matching_session_tokens, context->embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, context->n_matching_session_tokens, context->embd_inp.size());
        }
        
        // Remove any "future" tokens that we might have inherited from the previous session
        auto * mem = llama_get_memory(context->ctx);
        llama_memory_seq_rm(mem, -1, context->n_matching_session_tokens, -1);
    }
    
    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
         context->embd_inp.size(), context->n_matching_session_tokens, context->embd_inp.size(), context->session_tokens.size());
    
    // If we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!context->embd_inp.empty() && context->n_matching_session_tokens == context->embd_inp.size() && context->session_tokens.size() > context->embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )\n", context->embd_inp.size() - 1);
        context->session_tokens.resize(context->embd_inp.size() - 1);
    }
    
    return GENERATING_TOKENS;
}

// Token generation state handler
LlamaState handle_generating_tokens(LlamaContext* context) {
    LOG_DBG("State: GENERATING_TOKENS\n");
    
    const llama_vocab * vocab = llama_model_get_vocab(context->model);
    
    // Main generation loop condition
    if (!((context->n_remain != 0 && !context->is_antiprompt) || context->params.interactive)) {
        return FINISHING;
    }
    
    // Predict
    if (!context->embd.empty()) {
        return MANAGING_CONTEXT;
    }
    
    context->embd.clear();
    
    if ((int) context->embd_inp.size() <= context->n_consumed && !context->is_interacting) {
        // Optionally save the session on first sample (for faster prompt loading next time)
        if (!context->path_session.empty() && context->need_to_save_session && !context->params.prompt_cache_ro) {
            context->need_to_save_session = false;
            llama_state_save_file(context->ctx, context->path_session.c_str(), context->session_tokens.data(), context->session_tokens.size());
            LOG_DBG("saved session to %s\n", context->path_session.c_str());
        }
        
        const llama_token id = common_sampler_sample(context->sampler, context->ctx, -1);
        common_sampler_accept(context->sampler, id, /* accept_grammar= */ true);
        
        context->embd.push_back(id);
        
        // Echo this to console
        context->input_echo = true;
        
        // Decrement remaining sampling budget
        --context->n_remain;
        
        LOG_DBG("n_remain: %d\n", context->n_remain);
    } else {
        // Some user input remains from prompt or interaction, forward it to processing
        LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) context->embd_inp.size(), context->n_consumed);
        while ((int) context->embd_inp.size() > context->n_consumed) {
            context->embd.push_back(context->embd_inp[context->n_consumed]);
            
            // Push the prompt in the sampling context in order to apply repetition penalties later
            // for the prompt, we don't apply grammar rules
            common_sampler_accept(context->sampler, context->embd_inp[context->n_consumed], /* accept_grammar= */ false);
            
            ++context->n_consumed;
            if ((int) context->embd.size() >= context->params.n_batch) {
                break;
            }
        }
    }
    
    // Display text
    if (context->input_echo && context->display) {
        for (auto id : context->embd) {
            const std::string token_str = common_token_to_piece(context->ctx, id, context->params.special);
            
            // Console/Stream Output
            LOG("%s", token_str.c_str());
            
            // Record Displayed Tokens To Log
            // Note: Generated tokens are created one by one hence this check
            if (context->embd.size() > 1) {
                // Incoming Requested Tokens
                context->input_tokens.push_back(id);
            } else {
                // Outgoing Generated Tokens
                context->output_tokens.push_back(id);
                context->output_ss << token_str;
            }
        }
    }
    
    // Reset color to default if there is no pending user input
    if (context->input_echo && (int) context->embd_inp.size() == context->n_consumed) {
        console::set_display(console::reset);
        context->display = true;
    }
    
    // If not currently processing queued inputs
    if ((int) context->embd_inp.size() <= context->n_consumed) {
        // Check for reverse prompt in the last n_prev tokens
        if (!context->params.antiprompt.empty()) {
            const int n_prev = 32;
            const std::string last_output = common_sampler_prev_str(context->sampler, context->ctx, n_prev);
            
            context->is_antiprompt = false;
            // Check if each of the reverse prompts appears at the end of the output.
            // If we're not running interactively, the reverse prompt might be tokenized with some following characters
            // so we'll compensate for that by widening the search window a bit.
            for (std::string & antiprompt : context->params.antiprompt) {
                size_t extra_padding = context->params.interactive ? 0 : 2;
                size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                    ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                    : 0;
                
                if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                    if (context->params.interactive) {
                        context->is_interacting = true;
                    }
                    context->is_antiprompt = true;
                    break;
                }
            }
            
            // Check for reverse prompt using special tokens
            // avoid calling common_sampler_last() if last_output is empty
            if (!last_output.empty()) {
                llama_token last_token = common_sampler_last(context->sampler);
                for (auto token : context->antiprompt_token) {
                    if (token == last_token) {
                        if (context->params.interactive) {
                            context->is_interacting = true;
                        }
                        context->is_antiprompt = true;
                        break;
                    }
                }
            }
            
            if (context->is_antiprompt) {
                LOG_DBG("found antiprompt: %s\n", last_output.c_str());
            }
        }
        
        // Deal with end of generation tokens in interactive mode
        if (!context->waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(context->sampler))) {
            LOG_DBG("found an EOG token\n");
            
            if (context->params.interactive) {
                if (!context->params.antiprompt.empty()) {
                    // Tokenize and inject first reverse prompt
                    const auto first_antiprompt = common_tokenize(context->ctx, context->params.antiprompt.front(), false, true);
                    context->embd_inp.insert(context->embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                    context->is_antiprompt = true;
                }
                
                if (context->params.enable_chat_template) {
                    // Helper function for chat formatting
                    auto chat_add_and_format = [&](const std::string & role, const std::string & content) {
                        common_chat_msg new_msg;
                        new_msg.role = role;
                        new_msg.content = content;
                        auto formatted = common_chat_format_single(context->chat_templates.get(), context->chat_msgs, new_msg, role == "user", context->params.use_jinja);
                        context->chat_msgs.push_back(new_msg);
                        LOG_DBG("formatted: '%s'\n", formatted.c_str());
                        return formatted;
                    };
                    chat_add_and_format("assistant", context->assistant_ss.str());
                }
                context->is_interacting = true;
                LOG("\n");
            }
        }
        
        // If current token is not EOG, we add it to current assistant message
        if (context->params.conversation_mode && !context->waiting_for_first_input) {
            const auto id = common_sampler_last(context->sampler);
            context->assistant_ss << common_token_to_piece(context->ctx, id, false);
            
            std::string prompt = context->params.prompt;
            if (!prompt.empty()) {
                prompt.clear();
                context->is_interacting = false;
            }
        }
        
        if ((context->n_past > 0 || context->waiting_for_first_input) && context->is_interacting) {
            return WAITING_FOR_INPUT;
        }
        
        if (context->n_past > 0 || context->waiting_for_first_input) {
            if (context->is_interacting) {
                common_sampler_reset(context->sampler);
            }
            context->is_interacting = false;
            
            if (context->waiting_for_first_input && context->params.single_turn) {
                context->params.interactive = false;
                context->params.interactive_first = false;
            }
            context->waiting_for_first_input = false;
        }
    }
    
    // End of generation
    if (!context->embd.empty() && llama_vocab_is_eog(vocab, context->embd.back()) && !(context->params.interactive)) {
        LOG(" [end of text]\n");
        return FINISHING;
    }
    
    // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
    // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
    if (context->params.interactive && context->n_remain <= 0 && context->params.n_predict >= 0) {
        context->n_remain = context->params.n_predict;
        context->is_interacting = true;
    }
    
    return GENERATING_TOKENS;
}

// Context management state handler
LlamaState handle_managing_context(LlamaContext* context) {
    LOG_DBG("State: MANAGING_CONTEXT\n");
    
    // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
    // --prompt or --file which uses the same value.
    int max_embd_size = context->n_ctx - 4;
    
    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if ((int) context->embd.size() > max_embd_size) {
        const int skipped_tokens = (int) context->embd.size() - max_embd_size;
        context->embd.resize(max_embd_size);
        
        console::set_display(console::error);
        LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
        console::set_display(console::reset);
    }
    
    auto * mem = llama_get_memory(context->ctx);
    
    if (context->ga_n == 1) {
        // Infinite text generation via context shifting
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via n_past)
        // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
        
        if (context->n_past + (int) context->embd.size() >= context->n_ctx) {
            if (!context->params.ctx_shift){
                LOG_DBG("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                return FINISHING;
            }
            
            if (context->params.n_predict == -2) {
                LOG_DBG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, context->params.n_predict);
                return FINISHING;
            }
            
            const int n_left    = context->n_past - context->params.n_keep;
            const int n_discard = n_left/2;
            
            LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    context->n_past, n_left, context->n_ctx, context->params.n_keep, n_discard);
            
            llama_memory_seq_rm (mem, 0, context->params.n_keep            , context->params.n_keep + n_discard);
            llama_memory_seq_add(mem, 0, context->params.n_keep + n_discard, context->n_past, -n_discard);
            
            context->n_past -= n_discard;
            
            LOG_DBG("after swap: n_past = %d\n", context->n_past);
            LOG_DBG("embd: %s\n", string_from(context->ctx, context->embd).c_str());
            LOG_DBG("clear session path\n");
            context->path_session.clear();
        }
    } else {
        // Context extension via Self-Extend
        while (context->n_past >= context->ga_i + context->ga_w) {
            const int ib = (context->ga_n*context->ga_i)/context->ga_w;
            const int bd = (context->ga_w/context->ga_n)*(context->ga_n - 1);
            const int dd = (context->ga_w/context->ga_n) - ib*bd - context->ga_w;
            
            LOG_DBG("\n");
            LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", context->ga_i, context->n_past, ib*bd, context->ga_i + ib*bd, context->n_past + ib*bd);
            LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", context->ga_i + ib*bd, context->ga_i + ib*bd + context->ga_w, context->ga_n, (context->ga_i + ib*bd)/context->ga_n, (context->ga_i + ib*bd + context->ga_w)/context->ga_n);
            LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", context->ga_i + ib*bd + context->ga_w, context->n_past + ib*bd, dd, context->ga_i + ib*bd + context->ga_w + dd, context->n_past + ib*bd + dd);
            
            llama_memory_seq_add(mem, 0, context->ga_i,                context->n_past,              ib*bd);
            llama_memory_seq_div(mem, 0, context->ga_i + ib*bd,        context->ga_i + ib*bd + context->ga_w, context->ga_n);
            llama_memory_seq_add(mem, 0, context->ga_i + ib*bd + context->ga_w, context->n_past + ib*bd,      dd);
            
            context->n_past -= bd;
            context->ga_i += context->ga_w/context->ga_n;
            
            LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", context->n_past + bd, context->n_past, context->ga_i);
        }
    }
    
    // Try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
    if (context->n_session_consumed < (int) context->session_tokens.size()) {
        size_t i = 0;
        for ( ; i < context->embd.size(); i++) {
            if (context->embd[i] != context->session_tokens[context->n_session_consumed]) {
                context->session_tokens.resize(context->n_session_consumed);
                break;
            }
            
            context->n_past++;
            context->n_session_consumed++;
            
            if (context->n_session_consumed >= (int) context->session_tokens.size()) {
                ++i;
                break;
            }
        }
        if (i > 0) {
            context->embd.erase(context->embd.begin(), context->embd.begin() + i);
        }
    }
    
    for (int i = 0; i < (int) context->embd.size(); i += context->params.n_batch) {
        int n_eval = (int) context->embd.size() - i;
        if (n_eval > context->params.n_batch) {
            n_eval = context->params.n_batch;
        }
        
        LOG_DBG("eval: %s\n", string_from(context->ctx, context->embd).c_str());
        
        if (llama_decode(context->ctx, llama_batch_get_one(&context->embd[i], n_eval))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            context->error_message = "Failed to evaluate tokens";
            return ERROR;
        }
        
        context->n_past += n_eval;
        
        LOG_DBG("n_past = %d\n", context->n_past);
        // Display total tokens alongside total time
        if (context->params.n_print > 0 && context->n_past % context->params.n_print == 0) {
            LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", context->n_past, context->n_ctx);
        }
    }
    
    if (!context->embd.empty() && !context->path_session.empty()) {
        context->session_tokens.insert(context->session_tokens.end(), context->embd.begin(), context->embd.end());
        context->n_session_consumed = context->session_tokens.size();
    }
    
    return GENERATING_TOKENS;
}

// Waiting for input state handler
LlamaState handle_waiting_for_input(LlamaContext* context) {
    LOG_DBG("State: WAITING_FOR_INPUT\n");
    LOG_DBG("waiting for user input\n");
    
    if (context->params.conversation_mode) {
        LOG("\n> ");
    }
    
    const llama_vocab * vocab = llama_model_get_vocab(context->model);
    
    if (context->params.input_prefix_bos) {
        LOG_DBG("adding input prefix BOS token\n");
        context->embd_inp.push_back(llama_vocab_bos(vocab));
    }
    
    std::string buffer;
    if (!context->params.input_prefix.empty() && !context->params.conversation_mode) {
        LOG_DBG("appending input prefix: '%s'\n", context->params.input_prefix.c_str());
        LOG("%s", context->params.input_prefix.c_str());
    }
    
    // Color user input only
    console::set_display(console::user_input);
    context->display = context->params.display_prompt;
    
    std::string line;
    bool another_line = true;
    do {
        another_line = console::readline(line, context->params.multiline_input);
        buffer += line;
    } while (another_line);
    
    // Done taking input, reset color
    console::set_display(console::reset);
    context->display = true;
    
    if (buffer.empty()) { // Ctrl+D on empty line exits
        LOG("EOF by user\n");
        return FINISHING;
    }
    
    if (buffer.back() == '\n') {
        // Implement #587:
        // If the user wants the text to end in a newline,
        // this should be accomplished by explicitly adding a newline by using \ followed by return,
        // then returning control by pressing return again.
        buffer.pop_back();
    }
    
    if (buffer.empty()) { // Enter key on empty line lets the user pass control back
        LOG_DBG("empty line, passing control back\n");
    } else { // Add tokens to embd only if the input buffer is non-empty
        return PROCESSING_INPUT;
    }
    
    context->input_echo = false; // do not echo this again
    
    if (context->n_past > 0 || context->waiting_for_first_input) {
        if (context->is_interacting) {
            common_sampler_reset(context->sampler);
        }
        context->is_interacting = false;
        
        if (context->waiting_for_first_input && context->params.single_turn) {
            context->params.interactive = false;
            context->params.interactive_first = false;
        }
        context->waiting_for_first_input = false;
    }
    
    return GENERATING_TOKENS;
}

// Processing input state handler
LlamaState handle_processing_input(LlamaContext* context) {
    LOG_DBG("State: PROCESSING_INPUT\n");
    
    // This state would be called from WAITING_FOR_INPUT when we have actual input to process
    // For now, we'll just return to GENERATING_TOKENS
    // In a full implementation, this would handle the input processing logic
    (void)context; // Suppress unused parameter warning
    
    return GENERATING_TOKENS;
}

// Finishing state handler
LlamaState handle_finishing(LlamaContext* context) {
    LOG_DBG("State: FINISHING\n");
    
    if (!context->path_session.empty() && context->params.prompt_cache_all && !context->params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, context->path_session.c_str());
        llama_state_save_file(context->ctx, context->path_session.c_str(), context->session_tokens.data(), context->session_tokens.size());
    }
    
    LOG("\n\n");
    common_perf_print(context->ctx, context->sampler);
    
    // Free resources
    common_sampler_free(context->sampler);
    
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_dev) {
        auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
        auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        if (ggml_threadpool_free_fn) {
            ggml_threadpool_free_fn(context->threadpool);
            ggml_threadpool_free_fn(context->threadpool_batch);
        }
    }
    
    llama_backend_free();
    
    return FINISHING; // Stay in finishing state
}

int main(int argc, char ** argv) {
    // Initialize context (already defined globally for signal handlers)
    g_ctx = LlamaContext{};
    
    // Set up signal handlers
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
    signal(SIGINT, sigint_handler);
#endif

    // Register console cleanup
    atexit([]() { console::cleanup(); });
    
    // Initial state
    LlamaState currentState = INITIALIZING;
    LlamaState nextState = INITIALIZING;
    
    // Main state machine loop
    while (currentState != FINISHING && currentState != ERROR) {
        // Process current state and determine next state
        switch (currentState) {
            case INITIALIZING:
                nextState = handle_initializing(argc, argv, &g_ctx);
                break;
                
            case LOADING_MODEL:
                nextState = handle_loading_model(&g_ctx);
                break;
                
            case PREPARING_PROMPT:
                nextState = handle_preparing_prompt(&g_ctx);
                break;
                
            case LOADING_SESSION:
                nextState = handle_loading_session(&g_ctx);
                break;
                
            case GENERATING_TOKENS:
                nextState = handle_generating_tokens(&g_ctx);
                break;
                
            case MANAGING_CONTEXT:
                nextState = handle_managing_context(&g_ctx);
                break;
                
            case WAITING_FOR_INPUT:
                nextState = handle_waiting_for_input(&g_ctx);
                break;
                
            case PROCESSING_INPUT:
                nextState = handle_processing_input(&g_ctx);
                break;
                
            case FINISHING:
                nextState = handle_finishing(&g_ctx);
                break;
                
            case ERROR:
                LOG_ERR("%s: error: %s\n", __func__, g_ctx.error_message.c_str());
                nextState = FINISHING;
                break;
                
            default:
                LOG_ERR("%s: invalid state\n", __func__);
                nextState = ERROR;
                break;
        }
        
        // Transition to next state
        currentState = nextState;
    }
    
    return (currentState == ERROR) ? 1 : 0;
}