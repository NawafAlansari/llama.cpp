#include "helpers.h"

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

    fprintf(logfile, "binary: main\n");
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

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

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

int initializeModelAndContext(const gpt_params &params, llama_model **model, llama_context **ctx, llama_context **ctx_guidance)
{
    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (params.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return 1;
    }

    return 0; 
}


void loadModelAndApplyLora(const gpt_params &params, llama_model *model, llama_context *ctx, llama_context **ctx_guidance)
{
}


void checkContextSize(const gpt_params &params, llama_context *ctx)
{
}


void logSystemInfo(const gpt_params &params)
{
}


void testMemoryUsage(const gpt_params &params, llama_context *ctx, llama_model *model)
{
}


void exportCgraph(llama_context *ctx)
{
}


std::vector<llama_token> loadSavedSession(llama_context *ctx, const std::string &path_session)
{
    return std::vector<llama_token>();
}


std::vector<llama_token> initializeInput(llama_context *ctx, const gpt_params &params, std::vector<llama_token> *embd_inp, std::vector<llama_token> *session_tokens, bool add_bos)
{
    return std::vector<llama_token>();
}


std::vector<llama_token> tokenizePrompt(llama_context *ctx, const std::string &prompt, bool add_bos)
{
    return std::vector<llama_token>();
}


std::tuple<std::vector<llama_token>, int, int> tokenizeNegativePrompt(llama_context *ctx_guidance, llama_context *ctx, const std::string &negative_prompt, bool add_bos, const std::string &original_prompt)
{
    return std::tuple<std::vector<llama_token>, int, int>();
}


bool checkTokenLength(const std::vector<llama_token> &embd_inp, llama_context *ctx)
{
    return false;
}


void recalculateCachedLogits(std::vector<llama_token> &session_tokens, const std::vector<llama_token> &embd_inp)
{
}


std::vector<llama_token> tokenizeInstructPrefix(llama_context *ctx, bool add_bos)
{
    return std::vector<llama_token>();
}


std::vector<llama_token> tokenizeInstructSuffix(llama_context *ctx)
{
    return std::vector<llama_token>();
}

void setupInstructMode(gpt_params &params, const std::vector<llama_token> &inp_pfx, const std::vector<llama_token> &inp_sfx)
{
}

void setupInteractiveMode(gpt_params &params)
{
}

void logVerbosePromptsAndParams(llama_context *ctx, const gpt_params &params, const std::vector<llama_token> &embd_inp, const std::vector<llama_token> &guidance_inp, llama_context *ctx_guidance)
{
}

void logInteractiveModeDetails(const gpt_params &params)
{
}

void logGenerationParams(llama_context *ctx, const gpt_params &params)
{
}

std::tuple<struct llama_grammar *, grammar_parser::parse_state> setupGrammar(llama_context *ctx, const std::string &grammar_str)
{
    return std::tuple<struct llama_grammar *, grammar_parser::parse_state>();
}

void setupInteractiveModeDetails(const gpt_params &params)
{
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
