#include "generation.h"
#include "helpers.h"

/*
What do I want to do: 
    - Start with a prompt and decode it. 
    - Save the results to the kv cache. 
    - Use this cache to with different prompts, say two for now, without having redecode the original prompt. 

How to do it: 
    - Define a structure that represents the decoded prompt, this should containt the following info: 
        - The prompt.
        - The decoded tokens. 
        - sequence id. 
        - the first and last position of the kv of the decoded prompt in the kv cache. 
*/


//batching 
struct batch_element{
    std::vector<llama_token> tokens; 
    int32_t n_token; 

}; 


struct batch { 
    std::vector<batch_element> elements = {}; 
    int32_t number_batches = 0; 
    int32_t n_tokens = 0;


    void add_element(std::vector<llama_token> tokens){
        batch_element be = {tokens, (int32_t) tokens.size()}; 
        elements.push_back(be); 
        n_tokens += be.n_token; 
        number_batches += 1;
    }

    
    batch_element get(int32_t i){
        return elements[i]; 
    }


    llama_batch make_llama_batch(int32_t i){
        
        std::vector<llama_token> tokens = elements[i].tokens;
        int32_t i_n_tokens = elements[i].n_token;
        
        llama_batch new_batch = llama_batch_init(i_n_tokens, 0, 1);
        for (int32_t j=0; j<i_n_tokens; j++){
            llama_batch_add(new_batch, tokens[j], j, {i}, true); 
        }

        return new_batch;
    }


    bool decode(llama_context *ctx){
        for(int32_t i=0; i<number_batches; i++){
            llama_batch batch_view = make_llama_batch(i);
            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0){
                LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", 1, ret);
                return false; 
            }
            llama_synchronize(ctx);
        }

        return true; 
    }


    void init_from_tokens(std::vector<std::vector<llama_token>> tokens) {
        for (auto t : tokens) {
            add_element(t);
        }
    }

    
    void clear(){
        elements.clear(); 
        n_tokens = 0; 
    }

};







std::vector<std::vector<llama_token>> tokenize_prompts(llama_context *ctx, std::vector<std::string> prompts){
    std::vector<std::vector<llama_token>> tokenized_prompts; 
    for (auto p: prompts){
        std::vector<llama_token> tokens = llama_tokenize(ctx, p, true); 
        tokenized_prompts.push_back(tokens);
    }
    return tokenized_prompts;

}

int main(int argc, char **argv){

    std::string model_path = "./models/phi-2.Q3_K_S.gguf"; 
    int n_len = 64; 

    auto [mparams, params] = getParams(argc, argv);
    auto [model, ctx] = initModelAndContext(model_path, mparams, params);
    
    std::vector<std::string> prompts = { 
        "The greatest adventure is what lies ahead, ",
        "A string in python is",
        "The comedy of louis ck is one of the "
    }; 
    
    auto tokenized_prompts = tokenize_prompts(ctx, prompts); 
    batch b; b.init_from_tokens(tokenized_prompts);

    if (!b.decode(ctx)){
        fprintf(stderr, "%s: error: failed to decode\n", __func__);
        exit(1); 
    }


    //auto * logits = llama_get_logits_ith(ctx, batch.n_tokens-1);


    
    int32_t n_seq_max = llama_n_seq_max(ctx); 
    llama_kv_cache_view kv_cache_view = llama_kv_cache_view_init(ctx, n_seq_max);
    llama_kv_cache_view_update(ctx, &kv_cache_view);

    int32_t n_seq_max_cache = kv_cache_view.n_seq_max;
    printf("n_seq_max_cache: %d\n", n_seq_max_cache);
    dump_kv_cache_view(kv_cache_view, 32); 
    
    llama_kv_cache_view_free(&kv_cache_view);
    
    llama_print_timings(ctx);
    fprintf(stderr, "\n");
    clean(ctx, model);
    
    return 0; 
}