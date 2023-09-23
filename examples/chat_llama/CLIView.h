#ifndef CLI_VIEW_H_
#define CLI_VIEW_H_

#include <string> 
#include "inference.h"

class CLIView {
public:
    static void printChatLlaMaLogo();
    static std::string getUserInput(bool multiline_input);
    static void displayOutput(std::string &piece);

    static void initChatView(bool simple_io, bool use_color);
    static void cleanChatView(); 

}; 


#endif // CLI_VIEW_H_