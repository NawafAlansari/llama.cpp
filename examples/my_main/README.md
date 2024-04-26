# Notes:
1. The baby llama example is an example that basically shows a basic implementation of the llama.cpp library over the ggml library directly, gives lots of insights, but I think it might be a little buggy. 
2. One thing I am not 100% clear on is the context: Does the context hold the data of all ggml_tensors that were created in it? what happens to that data when we delete the context? how do we copy data between two ggml_tensors that are in different contexts? 
3. What is a ggml_context? it is actually a very simple structure: 
```c
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;

    struct ggml_scratch scratch;
    struct ggml_scratch scratch_save;
};
```
   So, we can see that the context holds a memory buffer, a list of objects, and a scratch buffer. The scratch buffer is used to store temporary data that is used in the computation of the ggml_tensors. 
4. How does the process of allocating memory for a ggml_tensor in a context work? This is a little more comolicated it seems: 
    - First, when we want to create a new ggml_tensor, we call something like 
    ```c
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0); 
    ```
    what does this do? Roughly speaking, 
