#include "chat.h"
#include "generation.h"
#include "helpers.h"
#include "console.h"

//********************** Debugging **********************//
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


void dump_chat_context(chat_context * chat_ctx){
    fprintf(stderr, "chat_ctx->state: %d\n", (int) chat_ctx->state);
    fprintf(stderr, "chat_ctx->done: %d\n", chat_ctx->done);
    int cur_input_size = chat_ctx->current_user_input.size();
    printf("chat_ctx->input.size(): %d\n", cur_input_size);
    fprintf(stderr, "chat_ctx->input: ");
    for (auto id: chat_ctx->current_user_input){
        fprintf(stderr, "%s", llama_token_to_piece(chat_ctx->ctx, id).c_str());
    }
    fprintf(stderr, "\n");
}

/********************* Chat & Main *********************/
//TODO: There is a problem (maybe?) when we just add ONE token, two are added! (oh it is the enter key! maybe...)
void start_chat(chat_context * chat_ctx){
    while(!chat_ctx->done){
        update_chat_context(chat_ctx);

        switch(chat_ctx->state){
            case chat_state::WAITING_FOR_USER:
                get_user_input(chat_ctx); 
                dump_chat_context(chat_ctx);
                break; 
                
            case chat_state::PROCESSING:
                process_inputs(chat_ctx); 
                dump_cache(chat_ctx->ctx);
                break; 

            case chat_state::GENERATING:
                generate(chat_ctx); 
                break; 
        }

        change_state(chat_ctx);
    }
}

int main(int argc, char **argv){
    std::string model_path = "./models/phi-2.Q3_K_S.gguf"; 
    auto [mparams, params] = getParams(argc, argv);
    params.n_ctx = 32; 
    auto [model, ctx] = initModelAndContext(model_path, mparams, params);
    
    llama_sampling_params sparams;  
    llama_sampling_context * sampling_ctx = llama_sampling_init(sparams); 

    chat_context * chat_ctx = init_chat_context(ctx, model, sampling_ctx);

    start_chat(chat_ctx);
    
    llama_print_timings(ctx);   
    fprintf(stderr, "\n");
    free_chat_context(chat_ctx);
    clean(ctx, model);
    return 0; 
}