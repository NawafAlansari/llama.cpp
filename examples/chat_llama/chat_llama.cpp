#include "controllers.h"
#include "CLIView.h"
#include "inference.h"

#include "common.h"

//TODO(DONE): There is a weird bug where sometimes no user input goes through to model, or at least so it seems! Needs to be investigated.(DONE Awkwardly). 
//TODO: Implement chat history, saving and loading chat data. 
//TODO: Add negative prompting and stuff like that so that I can have more control over the conversation and stop it whenever it is the turn of the user.



int main(int argc, char *argv[])
{

    gpt_params params; 
    if (!gpt_params_parse(argc, argv, params)){
        return 1; 
    }
    ChatController chat(params);
    chat.runChat();
    return 0;
}