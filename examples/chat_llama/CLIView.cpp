#include "CLIView.h"
#include "console.h"


void CLIView::printChatLlaMaLogo() {
    std::cout << R"(
  C  H  A  T       L  L  a  M  A
 /__/__/__/__   /__/__/__/__/__/__/
     |    |         |         |  
    /_\  /_\       /_\       /_\  
    )" << std::endl;
}


std::string CLIView::getUserInput(bool multiline_input)
{   
    std::string user_input;
    console::set_display(console::user_input);
    std::cout << "User: "; 
    console::readline(user_input, multiline_input);
    console::set_display(console::reset);
    return user_input; 
}

void CLIView::displayOutput(std::string &token_text)
{
    fprintf(stdout, "%s", token_text.c_str());
    fflush(stdout);
    return;
}


void CLIView::initChatView(bool simple_io, bool use_color){
    console::init(simple_io, use_color);
    return; 
}

void CLIView::cleanChatView()
{
    console::cleanup(); 
    return; 
}
