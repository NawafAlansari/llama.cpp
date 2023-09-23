#include "inference.h"

InferenceEngine::InferenceEngine(gpt_params params)
{
    this->params = params;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    n_vocab = llama_n_vocab(ctx); 
    n_past = 0;
    eos = llama_token_eos(ctx);
    add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM; 

    return;
}

void InferenceEngine::tokenizeInput(const char *text)
{
    input_tokens = ::llama_tokenize(ctx, text, add_bos); 
    addInputTokensToContext(); 
}

std::string InferenceEngine::complete()
{
    getNextToken(); 
    updateHasNextToken();
        
    addNextTokenToContext();
    updateLastOutputTokens();
        
    return llama_token_to_piece(ctx, next_token);
}

void InferenceEngine::addInputTokensToContext()
{
    context_tokens.insert(context_tokens.end(), input_tokens.begin(), input_tokens.end());
        if (context_tokens.size() > params.n_ctx){
            context_tokens.erase(context_tokens.begin(), context_tokens.begin() + context_tokens.size() - params.n_ctx );

        }
}

void InferenceEngine::addNextTokenToContext()
{
    if (has_next_token && next_token != eos){        
            context_tokens.push_back(next_token);
        }
        
        if (context_tokens.size() > params.n_ctx){
            context_tokens.erase(context_tokens.begin());

        }
}

void InferenceEngine::updateLastOutputTokens()
{
    if (has_next_token && next_token != eos){
            last_output_tokens.push_back(next_token);

        }
        if (last_output_tokens.size() > params.n_ctx){
            last_output_tokens.erase(last_output_tokens.begin());
        }
}

void InferenceEngine::updateHasNextToken()
{
    has_next_token = (next_token != eos);

}

int InferenceEngine::evaluateModel()
{
    int old_n_past = n_past;
    n_past = std::min((int)context_tokens.size(), params.n_ctx)-1;
    int n_eval = context_tokens.size() - old_n_past;

    return llama_eval(ctx, &context_tokens[old_n_past], n_eval, old_n_past, params.n_threads); 
        
}

void InferenceEngine::sampleNextToken(llama_token_data_array *candidates)
{
    if (params.temp < 0){
            next_token = llama_sample_token_greedy(ctx, candidates);
        
        } else {
            if (params.mirostat == 1){
                static float mirostat_mu = 2.0f * params.mirostat_tau; 
                const int mirostat_m = 100; 
                llama_sample_temperature(ctx, candidates, params.temp); 
                next_token = llama_sample_token_mirostat(ctx, candidates, params.mirostat_tau, params.mirostat_eta, mirostat_m, &mirostat_mu);


            }else if (params.mirostat == 2){
                static float mirostat_mu = 2.0f * params.mirostat_tau;
                llama_sample_temperature(ctx, candidates, params.temp);
                next_token = llama_sample_token_mirostat_v2(ctx, candidates, params.mirostat_tau, params.mirostat_eta, &mirostat_mu);

            }else {
                // Temperature sampling
                    size_t min_keep = std::max(1, params.n_probs);
                    llama_sample_top_k(ctx, candidates, params.top_k, min_keep);
                    llama_sample_tail_free(ctx, candidates, params.tfs_z, min_keep);
                    llama_sample_typical(ctx, candidates, params.typical_p, min_keep);
                    llama_sample_top_p(ctx, candidates, params.top_p, min_keep);
                    llama_sample_temperature(ctx, candidates, params.temp);
                    next_token = llama_sample_token(ctx, candidates);

            }
        }
}

void InferenceEngine::applyPenalties(llama_token_data_array *candidates)
{
    auto last_n_repeat = std::min(std::min((int)last_output_tokens.size(), params.repeat_last_n), params.n_ctx);
        llama_sample_repetition_penalty(ctx, candidates,
                                        last_output_tokens.data() + last_output_tokens.size() - last_n_repeat,
                                        last_n_repeat, params.repeat_penalty);

        llama_sample_frequency_and_presence_penalties(ctx, candidates,
                                        last_output_tokens.data() + last_output_tokens.size() - last_n_repeat,
                                        last_n_repeat, params.frequency_penalty, params.presence_penalty);   
}

void InferenceEngine::getNextToken()
{
    //Evaluate the model  
    evaluateModel();
    //logits 
    float *logits = llama_get_logits(ctx);
    std::vector<llama_token_data>  candidates; 

    //applyLogitsBiasMap(&logits, params.logit_bias); 
    
    
    //Construct candidates 
    candidates.reserve(n_vocab); 
    for (llama_token token_id=0; token_id < n_vocab; token_id++){
        candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }


    //prepare candidates for llama_sample_token_greedy
    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false}; 
    applyPenalties(&candidates_p);
    sampleNextToken(&candidates_p);
    return;  
}
