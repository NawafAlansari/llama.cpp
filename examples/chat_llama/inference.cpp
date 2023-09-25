#include "inference.h"
#include <fstream> 
#include <sstream> 
#include <istream>
#include <iterator>

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

std::string InferenceEngine::getNextTokenText()
{      
    getNextToken(); 
    updateHasNextToken();

    if (has_next_token){ 
        addNextTokenToContext();
        updateLastOutputTokens();
    }
    else {

    }
        
    return llama_token_to_piece(ctx, next_token);
}

std::string InferenceEngine::serialize()
{   
    //Serialize context_tokens
    std::ostringstream oss;
    for(auto token: context_tokens){
        oss << token << " ";
    }
    oss << "\n";

    // Serialize input_tokens
    for (auto token: input_tokens){
        oss << token << " ";
    }

    oss << "\n";

    // Serialize last_output_tokens
    for (auto token: last_output_tokens){
        oss << token << " ";
    }
    oss << "\n";

    // Serialize other simple types
    oss << next_token << "\n";
    oss << n_past << "\n";
    oss << n_vocab << "\n";
    oss << eos << "\n";
    oss << add_bos << "\n";
    oss << has_next_token << "\n";

    return oss.str(); 
}

void InferenceEngine::deserialize(const std::string &serializedState)
{
    std::istringstream iss(serializedState);
    std::string line;

    // Deserialize context_tokens
    std::getline(iss, line);
    std::istringstream context_stream(line);
    context_tokens.clear();
    std::copy(std::istream_iterator<int>(context_stream), std::istream_iterator<int>(), std::back_inserter(context_tokens));

    // Deserialize input_tokens
    std::getline(iss, line);
    std::istringstream input_stream(line);
    input_tokens.clear();
    std::copy(std::istream_iterator<int>(input_stream), std::istream_iterator<int>(), std::back_inserter(input_tokens));

    // Deserialize last_output_tokens
    std::getline(iss, line);
    std::istringstream last_output_stream(line);
    last_output_tokens.clear();
    std::copy(std::istream_iterator<int>(last_output_stream), std::istream_iterator<int>(), std::back_inserter(last_output_tokens));

    // Deserialize other simple types
    std::getline(iss, line); next_token = std::stoi(line);
    std::getline(iss, line); n_past = std::stoi(line);
    std::getline(iss, line); n_vocab = std::stoi(line);
    std::getline(iss, line); eos = std::stoi(line);
    std::getline(iss, line); add_bos = std::stoi(line);
    std::getline(iss, line); has_next_token = std::stoi(line);
}

void InferenceEngine::addInputTokensToContext()
{   
    context_tokens.insert(context_tokens.end(), input_tokens.begin(), input_tokens.end());

        if (context_tokens.size() >= params.n_ctx){
            context_tokens.erase(context_tokens.begin(), context_tokens.begin() + context_tokens.size() - params.n_ctx);
        }
}

void InferenceEngine::addNextTokenToContext()
{
    if (has_next_token){        
            context_tokens.push_back(next_token);
        }
        
        if (context_tokens.size() > params.n_ctx){
            context_tokens.erase(context_tokens.begin());

        }
}

void InferenceEngine::updateLastOutputTokens()
{
    if (has_next_token){
            last_output_tokens.push_back(next_token);

        }
        if (last_output_tokens.size() > params.n_ctx){
            last_output_tokens.erase(last_output_tokens.begin());
        }
}

void InferenceEngine::updateHasNextToken()
{
    if (next_token == eos){
            has_next_token = false;
            next_token = -1; 
        }
    else{
            has_next_token = true;
    }

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

SessionManager::SessionManager()
{
}

SessionManager::~SessionManager()
{
}


bool SessionManager::saveSession(InferenceEngine &engine, const std::string &filePath){
    std::string selializedState = serialize(engine);
    std::ofstream outFile(filePath); 
    if (!outFile.is_open()){
        return false; 
    }
    outFile << selializedState;
    outFile.close();
    return true;
}

bool SessionManager::loadSession(InferenceEngine &engine, const std::string &filePath){
    std::ifstream inFile(filePath); 
    if (!inFile.is_open()){
        return false; 
    }
    std::stringstream buffer; 
    buffer << inFile.rdbuf();
    std::string serializedState = buffer.str();
    inFile.close(); 

    deserialize(serializedState, engine);
    return true; 
}

std::string SessionManager::serialize(InferenceEngine &engine){
     std::string serializedState = engine.serialize(); 
     return serializedState;

}

void SessionManager::deserialize(const std::string &serializedState, InferenceEngine &engine)
{
    engine.deserialize(serializedState);
}
