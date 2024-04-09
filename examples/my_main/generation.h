#include <string>
#include <vector>
#include <unordered_map>


#include "common.h"
#include "llama.h"
#include "sampling.h"

llama_batch getBatch(llama_context *ctx, llama_model *model, std::string prompt); 
int generate(llama_context *ctx, llama_model *model, std::string prompt, int n_len);



/*
kv cache management: 
    - 
*/



//cached prompts; 

struct cached_prompt{
    std::string prompt; 
    std::vector<llama_token> tokens; 
    llama_seq_id seq_id; 
    llama_pos p0; 
    llama_pos p1; 
}; 

std::pair<llama_pos, llama_pos> cached_prompt_find_space(llama_context *ctx, int n_tokens); 
llama_seq_id cached_prompt_new_seq_id(llama_context *ctx, llama_pos p0, llama_pos p1); 

cached_prompt cached_prompt_make(std::string prompt, std::vector<llama_token> tokens); 
void cached_prompt_dump(cached_prompt cp);
void cached_prompt_free(cached_prompt &cp);



//prompt cache table 

struct cache_table{
    std::unordered_map<std::string, cached_prompt> prompt_map; 
}; 


cache_table cache_table_init();
void cache_table_add(cache_table &ct, std::string prompt_name, cached_prompt dp);
void cache_table_remove(cache_table &ct, std::string prompt_name);
void cache_table_save(cache_table ct, std::string path); 
void cache_table_load(cache_table &ct, std::string path);
void cache_table_dump(cache_table ct, int n);
void cache_table_free(cache_table &ct);



//High level functions: 

/*
Decode a prompt and add it to the cache table. 
*/
void decode_and_cache(llama_context *ctx, llama_model *model, cache_table &ct, std::string prompt_name, std::string prompt);

/*
Genrerate output from a prompt, using a precomputed cache. 
*/
void genreate_with_cache(llama_context *ctx, llama_model *model, cache_table ct, std::string cached_prompt_name, std::string prompt, int n_len);

