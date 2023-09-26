#ifndef CONTROLLERS_H_
#define CONTROLLERS_H_

#include "inference.h"
#include "CLIView.h"

#include <stack> 


class IController {
public:
    virtual void run() = 0;
    virtual ~IController() {} 
};



class ApplicationController: public IController{
public: 
    ApplicationController(); 
    void run() override;

private: 
    bool running = true;

}; 




class ChatListController: public IController{
public: 
    ChatListController(); 
    void run() override; 


private: 
    ChatModel chatModel; 
    ChatListView chatListView;
    void getCommand(std::string &commandBuffer); 
    void executeCommand(std::string &command);
    char getNavigationInput();
    void handleNavigationInput(int selected_chat);
    void cleanCommand(std::string &command);
    void startNewChat(int argc, char **argv);

    bool inCommandMode = false; 
}; 


class ChatController: public IController{
public:
    ChatController(gpt_params params);
    void run() override;
    
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