#ifndef CONTROLLERS_H_
#define CONTROLLERS_H_

#include "inference.h"
#include "CLIView.h"

class ChatController{
public:
    ChatController(gpt_params params);
    void runChat();
    gpt_params params; 


private: 
    InferenceEngine engine;
    void streamResponse();
    void sendToInferenceEngine(std::string &user_input); 

}; 




#endif // CONTROLLERS_H_