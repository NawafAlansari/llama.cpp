#ifndef CONTROLLERS_H_
#define CONTROLLERS_H_

#include "inference.h"
#include "CLIView.h"

class ChatController{
public:
    ChatController(gpt_params params);
    void runChat();
    
    gpt_params params; 
    SessionManager session;

private: 
    InferenceEngine engine;

    //Chat Management
    void streamResponse();
    void sendToInferenceEngine(std::string &user_input);
    
    //Saving and Loading Sessions
    void saveSession(const std::string filePath); 
    void loadSession(const std::string filePath); 
    
    //IO Management and Formatting.  
    void UserInputHandler(std::string &user_input, bool &chatting);
    void outputHandler(std::string &output);
    void formatOutput(std::string &output);
    void cleanUserInput(std::string &user_input);
    void cleanSpaces(std::string &text); 
    void cleanNewLines(std::string &text);
}; 




#endif // CONTROLLERS_H_