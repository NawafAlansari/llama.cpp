#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "common.h"
#include "common.h"
#include "train.h"

#include <vector>
#include <cstring> 
#include <ctime>
#include <algorithm>
#include <string> 
#include <stdio.h>


struct soft_prompt_params{
    int n_tokens; 
    uint32_t n_embd; 
};


struct soft_prompt{
    struct ggml_context * ctx = NULL; 
    ggml_backend_buffer_t data; 
    soft_prompt_params params;

    struct ggml_tensor * soft_tensors; 
};


struct train_params {
    struct train_params_common common; 

    const char * fn_model; 
    const char * fn_soft_prompt_out;

    int n_prompt_tokens; 
    int n_embd_dim; 
    
}; 

static struct train_params get_default_train_params(){
    struct train_params params; 
    params.common = get_default_train_params_common(); 
    params.fn_model = "./models/phi-2.Q3_K_S.gguf"; 
    params.fn_soft_prompt_out = NULL; 
    
    return params; 
}

static std::pair<llama_model*, llama_context*> initModelAndContext(std::string model_path, llama_model_params mparams, llama_context_params params){
     LOG("%s: llama backend init\n", __func__);
    llama_backend_init();
    
    llama_model *model; 
    model =  llama_load_model_from_file(model_path.c_str(), mparams);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        exit(1);
    }
    
    llama_context *ctx = llama_new_context_with_model(model, params); 
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        exit(1);
    }

    return {model, ctx}; 
}

static void init_soft_prompt(struct llama_model * model, struct soft_prompt * soft_prompt){
    const soft_prompt_params & params = soft_prompt->params;

    const uint32_t n_embd = params.n_embd;
    const uint32_t n_tokens = params.n_tokens;

    struct ggml_init_params sp_ctx_params; 
    sp_ctx_params.mem_size = ggml_tensor_overhead()*2*(n_tokens); //TODO: this might be wrong! 
    sp_ctx_params.mem_buffer = NULL; 
    sp_ctx_params.no_alloc = true; 

    struct ggml_context * ctx = ggml_init(sp_ctx_params);
    soft_prompt->ctx = ctx;

    soft_prompt->soft_tensors = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_embd); //TODO: The type might be wrong, we need to get model embeddoing type. 
    ggml_set_name(soft_prompt->soft_tensors, "soft_tensors");
    ggml_set_param(ctx, soft_prompt->soft_tensors); 
    
    soft_prompt->data = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type()); 

}

static void randomize_soft_prompt(struct soft_prompt * soft_prompt, int seed, float mean, float std, float min, float max){
    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, std, mean, min, max);

    randomize_tensor_normal(soft_prompt->soft_tensors, rnd); 
    free_random_normal_distribution(rnd);
    
}


int main(int argc, char **argv){
    //Step0: Set up constants and parameters.
    struct train_params params = get_default_train_params(); 
    
    if (params.common.seed == LLAMA_DEFAULT_SEED){
        params.common.seed = time(NULL); 
    }
    printf("%s: seed: %u\n", __func__, params.common.seed); 
    srand(params.common.seed);


    //Step1: Initialize the model and context
    struct llama_model_params mparams = llama_model_default_params(); 
    struct llama_context_params llama_cparams = llama_context_default_params();
    mparams.vocab_only = false; 
    printf("%s: model_base = '%s'\n", __func__, params.fn_model);   

    auto [model, lctx] = initModelAndContext(params.fn_model, mparams, llama_cparams);
    printf("\n"); 


  
    //Step2: Initialize the soft prompt and its context and set parameters. 
    struct soft_prompt soft_prompt;
    soft_prompt.params.n_embd = llama_n_embd(model); 
    soft_prompt.params.n_tokens = 4;

    //Step3: Set up Optimizer parameters. 
    struct train_state * train = init_train_state(); 
    struct ggml_opt_context * opt = train->opt;

    opt->params = ggml_opt_default_params(GGML_OPT_TYPE_ADAM); 
    opt->params.print_forward_graph = false; 
    opt->params.print_backward_graph = false;
    opt->params.graph_size = LLAMA_TRAIN_MAX_NODES; 
    opt->params.n_threads = params.common.n_threads;
    opt->params.past = params.common.opt_past;
    opt->params.delta = params.common.opt_delta;
    opt->params.max_no_improvement = params.common.opt_max_no_improvement;
    opt->params.n_gradient_accumulation = params.common.n_gradient_accumulation;
    opt->params.adam.n_iter = params.common.adam_n_iter;
    opt->params.adam.sched = 1.0f; 
    opt->params.adam.alpha = params.common.adam_alpha;
    opt->params.adam.decay = params.common.adam_decay;
    opt->params.adam.decay_min_ndim = params.common.adam_decay_min_ndim;
    opt->params.adam.beta1 = params.common.adam_beta1;
    opt->params.adam.beta2 = params.common.adam_beta2;
    opt->params.adam.gclip = params.common.adam_gclip;
    opt->params.adam.eps_f = params.common.adam_eps_f;

    printf("%s: Optemizer is set up!\n", __func__); 
    bool exited = false; 
    if (exited){
        //TODO: Implement this later once the whole thing is running. 
    }else{ 
        init_soft_prompt(model, &soft_prompt);
        randomize_soft_prompt(&soft_prompt, params.common.seed, 0.0f, 1.0f, -1.0f, 1.0f);
        ggml_opt_init(opt->ctx, opt, opt->params, 1024l*1024l /*get_parameter_count(soft_prompt*/); 
    }
    opt->iter = train->train_its; 
    printf("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its); 
    printf("%s: seen train tokens %llu\n", __func__, (long long unsigned) train->train_tokens);
    printf("%s: completed train epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
    
    printf("%s: opt_size = %zu bytes (%1f MB)\n", __func__, ggml_get_mem_size(opt->ctx), (float) ggml_get_mem_size(opt->ctx) / (1024.0f*1024.0f));

    //Step4: Allocate memory for inputs context, compute context. 
    //Input context without data.
    int n_ctx = llama_n_ctx(lctx);
    int n_tokens = n_ctx - soft_prompt.params.n_tokens;
    int n_vocab = llama_n_vocab(model);
    int n_batch = params.common.n_batch;

    printf("%s: n_ctx = %d\n", __func__, n_ctx);
    printf("%s: n_tokens = %d\n", __func__, n_tokens);
    printf("%s: n_vocab = %d\n", __func__, n_vocab);
    printf("%s: n_batch = %d\n", __func__, n_batch);

    struct ggml_init_params ctx_input_params = {
        ggml_tensor_overhead()*2, 
        NULL, 
        true
    }; 
    struct ggml_context * ctx_input = ggml_init(ctx_input_params);
    struct ggml_tensor * tokens_input = ggml_new_tensor_2d(ctx_input, GGML_TYPE_I32, n_tokens, n_batch); 
    struct ggml_tensor * target_probs = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab, n_tokens, n_batch); 

    ggml_backend_buffer_t input_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx_input, ggml_backend_cpu_buffer_type());
    size_t max_input_size = ggml_backend_buffer_get_size(input_data);
    printf("%s: input_size = %zu bytes (%1f MB)\n", __func__, max_input_size, (float) max_input_size / (1024.0f*1024.0f));
    
    //compute context without data.
    const size_t estimated_compute_size_wo_data = {
        2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
        (params.common.use_checkpointing? 3: 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true))
    }; 

    struct ggml_init_params ctx_compute_params = {
        estimated_compute_size_wo_data, 
        NULL, 
        true
    }; 
    struct ggml_context * ctx_compute = NULL; 

    struct ggml_tensor * loss = NULL; 
    struct ggml_tensor * logits = NULL; 

    struct ggml_cgraph * gf = NULL; 
    struct ggml_cgrapgh * gb = NULL; 
    struct ggml_cgraph * gb_tmp = NULL; 

    size_t best_compute_size = SIZE_MAX; 
    enum ggml_cgraph_eval_order best_order = GGML_CGRAPH_EVAL_ORDER_COUNT; 
    for (unsigned order=0; order<(unsigned) GGML_CGRAPH_EVAL_ORDER_COUNT; ++order){
        
    }
    
    //Step5: Tokenize the training data from a file and shuffle and so on. 

    //Step6: Prepare the training state. 

    //Step7: Begin trainign. 

    //Step8: Clean up.
    ggml_free(ctx_input);
    ggml_free(ctx_compute);
    ggml_free(opt->ctx); 
    free_train_state(train); 
    llama_free(lctx);
    llama_free_model(model);  
    
    printf("End of prompt-tuning example!\n");
    return 0; 
}