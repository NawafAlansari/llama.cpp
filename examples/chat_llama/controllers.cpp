#include "controllers.h"
#include "CLIView.h"




ChatController::ChatController(gpt_params params): engine(params){
    this->params = params;
    session = SessionManager();
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
        UserInputHandler(user_input, chatting);  
    }


    CLIView::cleanChatView();


}

void ChatController::UserInputHandler(std::string &user_input, bool &chatting){

    cleanUserInput(user_input);
    
    if (user_input == "exit"){
        chatting = false; 
    }
    else if (user_input == "save"){
        std::string file_path = "./chat_session.txt"; 
        saveSession(file_path); 
    }
    else if (user_input == "load"){
        std::string file_path = "./chat_session.txt"; 
        loadSession(file_path); 
    }

    else{
        user_input = "User: " + user_input; 
        sendToInferenceEngine(user_input); 
        streamResponse(); 
    }
    return; 
}


void ChatController::streamResponse(){
    std::string token_text = ""; 
    while(engine.has_next_token){
        token_text = engine.getNextTokenText(); 
        outputHandler(token_text); 
    }
    //TODO: This is a bit awkward, but works for now. 
    engine.has_next_token = true; 
    return; 
}; 


void ChatController::sendToInferenceEngine(std::string &user_input){
    engine.tokenizeInput(user_input.c_str());
    return; 
}

void ChatController::saveSession(const std::string filePath)
{
    session.saveSession(engine, filePath); 
    return;

}

void ChatController::loadSession(const std::string filePath)
{
    session.loadSession(engine, filePath); 
    return;
}

void ChatController::outputHandler(std::string &output)
{   
    formatOutput(output); 
    CLIView::displayOutput(output);
    return;
}


void ChatController::formatOutput(std::string &output)
{
    if (!engine.has_next_token){ 
        output = output + "\n";
    } 
    return;
}


void ChatController::cleanUserInput(std::string &user_input){
    cleanSpaces(user_input);
    cleanNewLines(user_input); 

    return;
}

void ChatController::cleanSpaces(std::string &text){
    text.erase(std::remove(text.begin(), text.end(), ' '), text.end());
    return; 
}

void ChatController::cleanNewLines(std::string &text){
    text.erase(std::remove(text.begin(), text.end(), '\n'), text.end());
    return; 
}