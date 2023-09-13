#pragma once

#include "llama.h"  // Assuming you have a file named "llama.h" that contains the definition of gpt_params
#include "common.h"
#include "grammar-parser.h"
#include "build-info.h"
#include "console.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


struct SessionState {
    int n_ctx;
    std::vector<llama_token> last_tokens;

    bool is_interacting; 
    bool is_antiprompt;
    bool input_echo;
    bool need_to_save_session;

    int n_past;
    int n_remain;
    int n_consumed;
    int n_session_consumed;
    int n_past_guidance0;

    std::ostringstream output_ss;

};

struct SessionTokens {
    std::vector<llama_token> input_tokens;
    std::vector<llama_token> output_tokens;
    std::vector<llama_token> candidataes; 
    
}; 


