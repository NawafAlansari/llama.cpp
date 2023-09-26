#include "controllers.h"
#include "CLIView.h"
#include "common.h"

#include <iterator>



ChatController::ChatController(gpt_params params): engine(params){
    this->params = params;
    session = SessionManager();
    return;  
}

void ChatController::run(){
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

ChatListController::ChatListController()
{
    return;
}

void ChatListController::run()
{
    std::string commandBuffer; 

    while(true){
        if(inCommandMode){
            getCommand(commandBuffer);
            executeCommand(commandBuffer);
            inCommandMode = false; 
        
        } else {
            int selected_chat = getNavigationInput(); 
            if (selected_chat == ':'){
                inCommandMode = true; 
            } else {
                handleNavigationInput(selected_chat);
            }
        }
    }
}


void ChatListController::getCommand(std::string &commandBuffer)
{
    std::cout << "Command: ";
    std::getline(std::cin, commandBuffer);
}


void ChatListController::executeCommand(std::string &command) 
{
    //TODO: First decide what command u will need and how to handle them. 
    cleanCommand(command);
    std::vector<std::string> tokenized_command; 
    std::istringstream iss(command);

    std::copy(std::istream_iterator<std::string>(iss),
              std::istream_iterator<std::string>(),
              std::back_inserter(tokenized_command));


    if (tokenized_command.empty()){
        std::cout << "Invalid Command\n";
        return; 
    }


    if (tokenized_command[0] == "new"){
        tokenized_command.erase(tokenized_command.begin()); 

        std::vector<const char*> c_tokens; 
        for (const auto &token: tokenized_command){
            c_tokens.push_back(token.c_str());
        }

        startNewChat(c_tokens.size(), const_cast<char**>(&c_tokens[0])); 
        return; 
    }

}

char ChatListController::getNavigationInput()
{
    char ch; 
    std::cin >> ch; 
    return ch; 
}

void ChatListController::handleNavigationInput(int selected_chat)
{
    //TODO: decide what are the possible navigation inputs and how to handle them.
}

void ChatListController::cleanCommand(std::string &command)
{   

    //remove all spaces from beginning of command
    command.erase(command.begin(), std::find_if(command.begin(), command.end(), [](int ch) {
        return !std::isspace(ch);
    }));

    //remove all spaces from end of command 
    command.erase(std::remove(command.begin(), command.end(), ' '), command.end());
    return;

    // remove newlines from command
    command.erase(std::remove(command.begin(), command.end(), '\n'), command.end());
    return; 
}

void ChatListController::startNewChat(int argc, char **argv)
{
}




ApplicationController::ApplicationController()
{
    return; 
}


void ApplicationController::run()
{
    while(running){
        IController * currentController = new ChatListController();
        currentController->run(); 

        delete currentController;
    }
}

