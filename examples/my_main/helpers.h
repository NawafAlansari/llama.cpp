#include "common.h"
#include "llama.h"
#include "sampling.h"

std::pair<llama_model_params, llama_context_params> getParams(int argc, char **argv); 
std::pair<llama_model*, llama_context*> initModelAndContext(std::string model_path, llama_model_params mparams, llama_context_params params);
void clean(llama_context *ctx, llama_model *model); 