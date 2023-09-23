#include "controllers.h"
#include "CLIView.h"
#include "inference.h"

#include "common.h"

//TODO: There is a weird bug where sometimes no user input goes through to model, or at least so it seems! Needs to be investigated.



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