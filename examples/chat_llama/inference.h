#ifndef INFERENCE_H_
#define INFERENCE_H_

#include "llama.h"
#include "common.h"
#include <vector> 
#include <string> 
#include <tuple>


class InferenceEngine {
public: 
    InferenceEngine(gpt_params params); 
    void tokenizeInput(const char *text);
    std::string getNextTokenText();

    std::string serialize(); 
    void deserialize(const std::string &serializedState);

    bool has_next_token; 
    //TODO: build the interface for the inference engine


private: 
    
    gpt_params params; 
    llama_model *model;
    llama_context *ctx;
    
    //Token Management
    std::vector<llama_token> context_tokens;
    std::vector<llama_token> input_tokens;
    std::vector<llama_token> last_output_tokens;
    llama_token next_token;
    int n_past;
    int n_vocab;  
    llama_token eos;  

    //Flags    
    bool add_bos;

    //Token Management Functions 
    void addInputTokensToContext();
    void addNextTokenToContext();
    void updateLastOutputTokens(); 
    void updateHasNextToken(); 
    
    //Inference Functions
    int evaluateModel(); 
    void sampleNextToken(llama_token_data_array *candidates);
    void applyPenalties(llama_token_data_array *candidates); 
    void getNextToken();

};



struct ChatSessionState{
    std::string name; 
    std::string timestamp; 
    std::string filePath; 
}; 


class ChatModel{
public:
    std::vector<ChatSessionState> getChatSessions();
    bool saveChatSession(const ChatSessionState &session);
    bool deleteChatSession(const std::string name);
     
}; 





class  SessionManager{
public:
    SessionManager();
    ~SessionManager();

    bool saveSession(InferenceEngine &engine, const std::string &filePath);
    bool loadSession(InferenceEngine &engine, const std::string &filePath);


private: 
    std::string serialize(InferenceEngine &engine);
    void deserialize(const std::string &serializedState, InferenceEngine &engine);

    

}; 






#endif // INFERENCE_H_


