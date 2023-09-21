/*
What is the plan? 
    -1) Take one token, generate the next token.(DONE);
    0) take a hardcoded string, generate the output. (DONE);
    1) Make a small program that takes user input, and just generates one response then exists.(DONE);
    2) Make a small program that takes user input, and generates a conversation. (DONE);
    2.5) add colors using to conversation using the console library.(DONE) 
    3) If no token budget, handle until stopping; (DONE) 
    4) Implement context swapping; 
    5) Implement applyPanalties for the candidates; (DONE)
    6) Implement applyLogitsBiasMap for the logits; (What is it ?)
    7) Add diagnostics and testing runs; 
    8) Allow adding prefix for the AI response (for example "AI: ")
    

*/

#include "llama.h"
#include "common.h"
#include "console.h"

#include <iostream>
#include <string> 
#include <vector>
#include <algorithm>



struct inference_engine{
    gpt_params params; 
    llama_model *model; 
    llama_context *ctx;

    std::vector<llama_token> context_tokens;
    std::vector<llama_token> input_tokens;
    std::vector<llama_token> last_output_tokens;

    llama_token next_token;

    int n_past;
    int n_vocab;  
    llama_token eos;  
    
    bool has_next_token; 
    bool add_bos; 



    
    inference_engine(gpt_params params){
        this->params = params;
        std::tie(model, ctx) = llama_init_from_gpt_params(params);
        n_vocab = llama_n_vocab(ctx); 
        n_past = 0;
        eos = llama_token_eos(ctx);
        add_bos = llama_vocab_type(ctx) == LLAMA_VOCAB_TYPE_SPM; 

        return; 
    }


    void tokenizeInput(const char *text){
        input_tokens = ::llama_tokenize(ctx, text, add_bos); 
        addInputTokensToContext(); 
    }


    void addInputTokensToContext(){
        context_tokens.insert(context_tokens.end(), input_tokens.begin(), input_tokens.end());
        if (context_tokens.size() > params.n_ctx){
            context_tokens.erase(context_tokens.begin(), context_tokens.begin() + context_tokens.size() - params.n_ctx);
        }
    }

    void addNextTokenToContext(){
        if (has_next_token){
            context_tokens.push_back(next_token);
        }
        if (context_tokens.size() > params.n_ctx){
            context_tokens.erase(context_tokens.begin());
        }
    }


    int evaluateModel(){
        int old_n_past = n_past;
        n_past = context_tokens.size();
        
        int n_eval = context_tokens.size() - old_n_past;
        
        return llama_eval(ctx, &context_tokens[old_n_past], n_eval, old_n_past, params.n_threads); 
        
}

    void sampleNextToken(llama_token_data_array *candidates){
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


    void applyPenalties(llama_token_data_array *candidates){
        auto last_n_repeat = std::min(std::min((int)last_output_tokens.size(), params.repeat_last_n), params.n_ctx);
        llama_sample_repetition_penalty(ctx, candidates,
                                        last_output_tokens.data() + last_output_tokens.size() - last_n_repeat,
                                        last_n_repeat, params.repeat_penalty);

        llama_sample_frequency_and_presence_penalties(ctx, candidates,
                                        last_output_tokens.data() + last_output_tokens.size() - last_n_repeat,
                                        last_n_repeat, params.frequency_penalty, params.presence_penalty);
    }


    void updateLastOutputTokens(){
        if (has_next_token){
            last_output_tokens.push_back(next_token);
        }
        if (last_output_tokens.size() > params.n_ctx){
            last_output_tokens.erase(last_output_tokens.begin());
        }
    }

    void getNextToken(){
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
        updateLastOutputTokens();
        
        return;  

    }

    void updateHasNextToken(){
        has_next_token = (next_token != eos);
    }

    std::string complete(){
        getNextToken(); 
        updateHasNextToken();
        addNextTokenToContext();
        return llama_token_to_piece(ctx, next_token);
        
    }
};


void readUserInput(std::string &user_input, bool multiline_input){
    std::cout << "User: "; 
    console::readline(user_input, multiline_input);
    console::set_display(console::reset);
    return; 
}

std::string streamResponse(inference_engine engine){   
    std::ostringstream output_ss;
    
    while(engine.has_next_token){        
        std::string token_text = engine.complete();     
        
        fprintf(stdout, "%s", token_text.c_str());
        output_ss << token_text;
        fflush(stdout);

    }

    return output_ss.str();
}



std::string getUserInput(){
    std::string user_input = "";
    std::cout << "User: "; 
    std::getline(std::cin, user_input);
    return user_input;
}


int main(int argc, char** argv){
    gpt_params params; 
    if (!gpt_params_parse(argc, argv, params)){
        return 1; 
    }
    console::init(params.simple_io, params.use_color);
    llama_backend_init(params.numa); 
    inference_engine engine  = inference_engine(params);

    
    //Prepare for generation
    std::string user_input; 
    std::string conversation = "";
    bool chatting = true;

    while(chatting){
        //get user input 
        readUserInput(user_input, params.multiline_input);
        engine.tokenizeInput(user_input.c_str());
        conversation += streamResponse(engine);
        
        std::cout << std::endl;
    }
    fprintf(stdout, "Conversation: %s\n", conversation.c_str());       


    // save session file 
    char* session_path = "chat_session_file.bin";
    llama_save_session_file(engine.ctx, session_path, nullptr, 0);

    llama_backend_free(); 
    console::cleanup();
    return 0; 
}