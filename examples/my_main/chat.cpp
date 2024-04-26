#include "chat.h"
#include "console.h" 

/********************** Chat Context **********************/
chat_context * init_chat_context(llama_context *ctx, llama_model *model, llama_sampling_context * sampling_ctx){
    chat_context * chat_ctx = new chat_context();
    chat_ctx->ctx = ctx; 
    chat_ctx->model = model; 
    chat_ctx->sampling_ctx = sampling_ctx;  
    chat_ctx->state = chat_state::WAITING_FOR_USER; 
    chat_ctx->done = false; 


    chat_ctx->n_ctx = llama_n_ctx(ctx);
    return chat_ctx;
}


void free_chat_context(chat_context * chat_ctx){
    delete chat_ctx; 
}


std::pair<llama_model_params, llama_context_params> getParams(int argc, char **argv){
    llama_model_params mparams = llama_model_default_params(); 
    llama_context_params params = llama_context_default_params();

    return {mparams, params}; 
}

/********************* Updating *********************/
void change_state(chat_context * chat_ctx){
    if(chat_ctx->state == chat_state::WAITING_FOR_USER){
        if (chat_ctx->current_user_input.size() > 0){
            chat_ctx->state = chat_state::PROCESSING; 
        }

    } else if (chat_ctx->state == chat_state::PROCESSING){
        chat_ctx->state = chat_state::GENERATING;
        
    } else{
        chat_ctx->state = chat_state::WAITING_FOR_USER;
    }
}


void update_chat_context(chat_context * chat_ctx){
    switch (chat_ctx->state)
    {
        case chat_state::WAITING_FOR_USER:
            chat_ctx->current_user_input.clear();
            break;

        case chat_state::PROCESSING:
            break;

        case chat_state::GENERATING:
            break;
    }
}

//********************** User Input **********************//
static std::vector<llama_token> add_prefix_suffix(std::vector<llama_token> &tokens, std::vector<llama_token> &prefix, std::vector<llama_token> &suffix){

    std::vector<llama_token> new_tokens;
    if (prefix.size() > 0){
        new_tokens.insert(new_tokens.end(), prefix.begin(), prefix.end());
    }
    new_tokens.insert(new_tokens.end(), tokens.begin(), tokens.end());
    if (suffix.size() > 0){
        new_tokens.insert(new_tokens.end(), suffix.begin(), suffix.end());
    }
    return new_tokens;     
}


static void get_console_user_input(std::string &buffer){
    console::set_display(console::user_input); 
    std::string line; 
    console::readline(line, false); 
    buffer += line; 
    console::set_display(console::reset); 
}


void get_user_input(chat_context * chat_ctx){ 
    std::string buffer; 
    get_console_user_input(buffer);

    if(buffer.length() > 1) {
        std::vector<llama_token> line_inp = llama_tokenize(chat_ctx->ctx, buffer, false, false);
        line_inp = add_prefix_suffix(line_inp, chat_ctx->input_prefix, chat_ctx->input_suffix);
        chat_ctx->current_user_input.insert(chat_ctx->current_user_input.end(), line_inp.begin(), line_inp.end());
    }
}

//********************** Processing **********************// 
static void truncate_input(std::vector<llama_token> &input, const int max_input_size){
    int n_input = input.size(); 

    if (n_input > max_input_size){
        input.resize(max_input_size);
    }
}

//TODO: There might be problems with the logic here! 
static void context_shift(llama_context *ctx, 
                          const int input_size, 
                          int &n_tokens_in_ctx, 
                          const int n_ctx, 
                          const int n_keep)
{

    if(input_size + n_tokens_in_ctx > n_ctx){
        const int n_left = n_tokens_in_ctx - n_keep;
        const int n_discard = n_left/2; 

        llama_kv_cache_seq_rm(ctx, 0, n_keep, n_keep+n_discard);
        llama_kv_cache_seq_add(ctx, 0, n_keep+n_discard, n_tokens_in_ctx, -n_discard); 

        n_tokens_in_ctx -= n_discard;
        
    }

}

//TODO: There might be a bug here when context is full and context switch happens 
static int decode_batch(llama_context *ctx, std::vector<llama_token> &tokens, int i, const int n_inputs, const int n_batch, int n_past){ 
    int n_eval = n_inputs - i;
    if (n_eval > n_batch){
        n_eval = n_batch;
    }
    llama_batch batch = llama_batch_get_one(&tokens[i], n_eval, n_past, 0);
    if(llama_decode(ctx, batch)){
        printf("%s : failed to eval\n", __func__); 
        exit(1);  
    }

    return n_eval; 
}


void process_inputs(chat_context * chat_ctx){
    llama_context * ctx = chat_ctx->ctx;
    const int n_ctx = chat_ctx->n_ctx;
    const int n_keep = chat_ctx->n_keep;
    const int n_inputs = chat_ctx->current_user_input.size(); 
    const int n_batch = chat_ctx->n_batch; 

    truncate_input(chat_ctx->current_user_input, n_ctx);
    context_shift(ctx, n_inputs, chat_ctx->n_tokens_in_ctx, n_ctx, n_keep);
    
    for (int i=0; i<n_inputs; i+=n_batch){
        int n_eval = decode_batch(ctx, chat_ctx->current_user_input, i, n_inputs, n_batch, chat_ctx->n_tokens_in_ctx);
        chat_ctx->n_tokens_in_ctx += n_eval; 
    }
}

//********************** Generation **********************// 
void generate(chat_context * chat_ctx){
}


