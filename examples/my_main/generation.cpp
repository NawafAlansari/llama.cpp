#include "generation.h"

llama_batch getBatch(llama_context *ctx, llama_model *model, std::string prompt){
    std::vector<llama_token> tokens = llama_tokenize(ctx, prompt, true); 

    printf("Prompt: ");
    for (auto id: tokens){
        printf("%s", llama_token_to_piece(ctx, id).c_str());
    }
    printf("\n");
    printf("%s: n_input_tokens: %lu\n", __func__, tokens.size());

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1); 
    for (size_t i=0; i<tokens.size(); i++){
        llama_batch_add(batch, tokens[i], i, {0}, false); 
    }
    batch.logits[batch.n_tokens-1] = true; 

    if(llama_decode(ctx, batch) !=0){
        fprintf(stderr, "%s: error: failed to decode\n", __func__);
        exit(1); 
    }

    return batch;
}


int generate(llama_context *ctx, llama_model *model, std::string prompt, int n_len){ 
    llama_batch batch = getBatch(ctx, model, prompt);
    
    int32_t n_seq_max = llama_n_seq_max(ctx);
    int32_t n_vocab = llama_n_vocab(model);
    
    int n_curr = batch.n_tokens;
    int n_decode = 0; 

    const auto t_main_start = ggml_time_us();
    while(n_curr < n_len){
        auto * logits = llama_get_logits_ith(ctx, batch.n_tokens-1);

        std::vector<llama_token_data> candidates; 
        candidates.reserve(n_vocab);

        for (llama_token id=0; id<n_vocab; id++){
            candidates.emplace_back(llama_token_data{id, logits[id], 0.0f});
        }
        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};
        const llama_token new_token_id = llama_sample_token(ctx, &candidates_p);
        if (new_token_id == llama_token_eos(model) || n_curr == n_seq_max){
            printf("Ending\n");
            break; 
        }

        LOG_TEE("%s", llama_token_to_piece(ctx, new_token_id).c_str());
        fflush(stdout);
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token_id, n_curr, {0}, true);

        n_decode += 1; 

        n_curr += 1;

        if (llama_decode(ctx, batch)){
            fprintf(stderr, "%s: error: failed to decode\n", __func__);
            exit(1); 
        }

        
    }  
    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_batch_free(batch);

    return n_decode; 
} 




std::pair<llama_pos, llama_pos> cached_prompt_find_space(llama_context *ctx, int n_tokens){
    llama_pos p0 = 0; 
    llama_pos p1 = n_tokens+1;
     
    
    return {p0, p1}; 
}