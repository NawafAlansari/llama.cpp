#include "controllers.h"
#include "CLIView.h"




ChatController::ChatController(gpt_params params): engine(params){
    this->params = params;
    return;  
}

void ChatController::runChat(){
    std::string user_input; 
    std::string response; 
    bool chatting = true;

    CLIView::initChatView(params.simple_io, params.use_color);  

    CLIView::printChatLlaMaLogo(); 
    while (chatting){
        user_input = CLIView::getUserInput(params.multiline_input); 
        sendToInferenceEngine(user_input);
        streamResponse();         
    }


    CLIView::cleanChatView();


}

void ChatController::streamResponse(){
    std::string token_text = ""; 
    while(engine.has_next_token){
        token_text = engine.complete(); 
        CLIView::displayOutput(token_text); 
    }
    return; 
}; 


void ChatController::sendToInferenceEngine(std::string &user_input){
    engine.tokenizeInput(user_input.c_str());
    return; 
}

