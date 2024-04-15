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
        new_batch.logits[new_batch.n_tokens-1] = true;
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



struct decoding_batch {
    std::vector<llama_token> tokens; 
    std::vector<int32_t> seq_ids; 
    int32_t n_batches; 

    decoding_batch(std::vector<llama_token> tokens, std::vector<int32_t> seq_ids): tokens(tokens), seq_ids(seq_ids){
        n_batches = tokens.size(); 
    }

    llama_batch make_llama_batch(){
        llama_batch new_batch = llama_batch_init(n_batches, 0, n_batches); 
        for (int32_t i=0; i<n_batches; i++){
            llama_batch_add(new_batch, tokens[i], i, {seq_ids[i]}, true); 
            new_batch.logits[new_batch.n_tokens-1] = true;
        }
        return new_batch;
    }

    bool decode(llama_context *ctx){ 
        llama_batch batch_view = make_llama_batch(); 
        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0){
            LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", 1, ret);
            return false; 
        }
        llama_synchronize(ctx);
        return true; 
    }

    void clear(){
        tokens.clear(); 
        seq_ids.clear(); 
        n_batches = 0; 
    }

}; 





struct decoding_sequence {
    std::vector<llama_token> tokens; 
    int32_t tokens_limit; 
    bool stop = false; 
    
    int32_t n_decoded; 
    int32_t n_tokens;

    decoding_sequence(std::vector<llama_token> tokens, int32_t tokens_limit): tokens(tokens), tokens_limit(tokens_limit){
        n_tokens = tokens.size(); 
        n_decoded = 0; 
    }; 

}; 


struct decoder {
    llama_context *ctx; 
    llama_model *model;
    std::vector<decoding_sequence> sequences; 

    decoder(llama_context *ctx, llama_model *model): ctx(ctx), model(model){}

    void decode(){ 
        
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


void dump_cache(llama_context *ctx, bool verbose=false){
    int32_t n_seq_max = llama_n_seq_max(ctx); 
    llama_kv_cache_view kv_cache_view = llama_kv_cache_view_init(ctx, n_seq_max);
    llama_kv_cache_view_update(ctx, &kv_cache_view);

    int32_t n_seq_max_cache = kv_cache_view.n_seq_max;
    printf("n_seq_max_cache: %d\n", n_seq_max_cache);
    if (verbose){
        dump_kv_cache_view_seqs(kv_cache_view, 32); 
    } else {
        dump_kv_cache_view(kv_cache_view, 32);
    }
    
    llama_kv_cache_view_free(&kv_cache_view);
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


    llama_kv_cache_seq_cp(ctx, 0, 1, -1, -1); 
    dump_cache(ctx, false);

    

    //auto * logits = llama_get_logits_ith(ctx, batch.n_tokens-1);
    
    dump_cache(ctx, false); 

    llama_pos seq_0_max_pos = llama_kv_cache_seq_pos_max(ctx, 0);
    llama_pos seq_1_max_pos = llama_kv_cache_seq_pos_max(ctx, 1);
    llama_pos seq_2_max_pos = llama_kv_cache_seq_pos_max(ctx, 2);

    fprintf(stderr, "seq_0_max_pos: %d\n", seq_0_max_pos);
    fprintf(stderr, "seq_1_max_pos: %d\n", seq_1_max_pos);
    fprintf(stderr, "seq_2_max_pos: %d\n", seq_2_max_pos);
    
    llama_print_timings(ctx);
    
    fprintf(stderr, "\n");
    
    clean(ctx, model);
    
    return 0; 
}