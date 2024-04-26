#include "common.h"
#include "llama.h"
#include "sampling.h"

/********************** Chat Context **********************/
enum struct chat_state{
    WAITING_FOR_USER, 
    PROCESSING, 
    GENERATING 
}; 

struct chat_context {
    llama_context *ctx; 
    llama_model *model; 
    llama_sampling_context * sampling_ctx; 

    //states 
    chat_state state; 
    bool done = false; 
    bool generating = false; 

    int n_tokens_in_ctx = 0;
    int n_consumed = 0; 

    //Input tokens managements 
    std::vector<llama_token> all_input;  
    std::vector<llama_token> current_user_input;  
    
    //output tokens management
    llama_token current_output_token; 
    std::vector<llama_token> current_output;
    
    //params 
    uint32_t n_ctx; 
    int n_batch = 512; 
    int n_keep = 0; 
    
    std::vector<llama_token> input_prefix;
    std::vector<llama_token> input_suffix;
}; 


chat_context * init_chat_context(llama_context *ctx, llama_model *model, llama_sampling_context * sampling_ctx); 
void free_chat_context(chat_context * chat_ctx);

void change_state(chat_context * chat_ctx);
void update_chat_context(chat_context * chat_ctx);
void get_user_input(chat_context * chat_ctx);
void process_inputs(chat_context * chat_ctx); 
void generate(chat_context * chat_ctx); 
