#include "helpers.h"




/**************************** Model and Context **************************/
std::pair<llama_model*, llama_context*> initModelAndContext(std::string model_path, llama_model_params mparams, llama_context_params params){
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



void clean(llama_context *ctx, llama_model *model){
    llama_free_model(model); 
    llama_free(ctx);
    llama_backend_free(); 
}