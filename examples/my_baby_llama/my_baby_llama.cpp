#include "ggml.h"
#include "train.h"

#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#ifdef LLAMA_DEFAULT_RMS_EPS
constexpr float rms_norm_eps = LLAMA_DEFAULT_RMS_EPS;
#else
constexpr float rms_norm_eps = 5e-6f;
#endif



void dump_ggml_tensor_f32_1d(const struct ggml_tensor * t, int per_row=8, int64_t n_rows=-1) {
    if (n_rows < 0) {
        n_rows = t->ne[0];
    } else {
        n_rows = std::min(n_rows, t->ne[0]);
    }

    for (int i=0; i<t->ne[0]; ++i) {
        printf("%f ", ggml_get_f32_1d(t, i));
        if ((i+1) % per_row == 0) {
            printf("\n");
            if ((i+1) / per_row >= n_rows) {
                break;
            }
        }
    }
}


void dump_ggml_tensor_i32_1d(const struct ggml_tensor * t, int per_row=8) {
    for (int i=0; i<t->ne[0]; ++i) {
        printf("%d ", ggml_get_i32_1d(t, i));
        if ((i+1) % per_row == 0) {
            printf("\n");
        }
    }
}


void dump_ggml_tensor_f32_2d(const struct ggml_tensor * t){
    int n_batch = t->nb[1];
    int n_token = t->nb[0];

    for (int i=0; i<n_batch; ++i){
        printf("Batch %d\n", i);
        for (int j=0; j<n_token; ++j){
            printf("%f ", ggml_get_f32_nd(t, j, i, 0, 0));
        }
        printf("\n");
    }
}

void dump_ggml_tensor_i32_2d(const struct ggml_tensor * t){
    int n_batch = t->nb[1];
    int n_token = t->nb[0];

    for (int i=0; i<n_batch; ++i){
        printf("Batch %d\n", i);
        for (int j=0; j<n_token; ++j){
            printf("%d ", ggml_get_i32_nd(t, j, i, 0, 0));
        }
        printf("\n");
    }
}


void dump_ggml_tensor_f32_3d_slice(const struct ggml_tensor * t, int slice){
    GGML_ASSERT(ggml_is_3d(t));
    GGML_ASSERT(slice < t->ne[2]);

    for (int i=0; i<t->ne[0]; ++i){
        for (int j=0; j<t->ne[1]; ++j){
            printf("%f ", ggml_get_f32_nd(t, i, j, slice, 0));
        }
        printf("\n");
    }
}





struct llama_hparams{ 
    uint32_t n_vocab = 8; 
    uint32_t n_ctx = 8; 
    uint32_t n_embd = 32;
    uint32_t n_mult = 2; 
    uint32_t n_head = 8; 
    uint32_t n_layer = 2; 
    uint32_t n_rot = 16; 
};

static uint32_t get_n_ff(const struct llama_hparams* hparams) {
    const uint32_t n_ff = ((2*(4*hparams->n_embd)/3 + hparams->n_mult - 1)/hparams->n_mult)*hparams->n_mult;
    return n_ff;
}


struct llama_layer{
    struct ggml_tensor * attnetion_norm; //(n_embd, 1) 

    struct ggml_tensor * wq; //(n_embd, n_embd)
    struct ggml_tensor * wk; //(n_embd, n_embd)
    struct ggml_tensor * wv; //(n_embd, n_embd)
    struct ggml_tensor * wo; //(n_embd, n_embd)

    struct ggml_tensor * ffn_norm; //(n_embd, 1)
    
    struct ggml_tensor * w1; //(n_embd, n_ff)
    struct ggml_tensor * w2; //(n_ff, n_embd)
    struct ggml_tensor * w3; //(n_embd, n_ff)
}; 



struct llama_kv_cache { 
    struct ggml_context * ctx = NULL; 

    struct ggml_tensor * k; 
    struct ggml_tensor * v;

    int n;  //Number of tokens in cach 
};


struct llama_model{
    struct ggml_context *ctx = NULL; 

    llama_hparams hparams; 

    struct ggml_tensor * tok_embeddings; //(n_embd, n_vocab)

    struct ggml_tensor * norm; //(n_embd, 1)
    struct ggml_tensor * output; //(n_embd, n_vocab)

    std::vector<llama_layer> layers; 
}; 



void init_model(struct llama_model *model){
    const auto & hparams = model->hparams;

    const uint32_t n_embd = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_vocab = hparams.n_vocab;

    const uint32_t n_ff = get_n_ff(&hparams);

    struct ggml_context * ctx = model->ctx; 

    model->tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); 
    model->norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); 
    model->output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); 

    model->layers.resize(n_layer); 
    for (uint32_t i=0; i < n_layer; ++i){ 
        auto & layer = model->layers[i]; 

        layer.attnetion_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); 
        
        layer.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd); 
        layer.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd); 
        layer.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd); 
        layer.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd); 

        layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); 

        layer.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff);
        layer.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_embd); 
        layer.w3 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff); 
    }
}



void set_param_model(struct llama_model * model){ 
    const auto &hparams = model->hparams; 

    const uint32_t n_layer = hparams.n_layer; 

    struct ggml_context *ctx = model->ctx; 

    ggml_set_param(ctx, model->tok_embeddings); 
    ggml_set_param(ctx, model->norm);
    ggml_set_param(ctx, model->output); 

    for (uint32_t i=0; i < n_layer; ++i){
        auto &layer = model->layers[i]; 

        ggml_set_param(ctx, layer.attnetion_norm); 
        ggml_set_param(ctx, layer.wq);
        ggml_set_param(ctx, layer.wk); 
        ggml_set_param(ctx, layer.wv); 
        ggml_set_param(ctx, layer.wo); 
        ggml_set_param(ctx, layer.ffn_norm); 
        ggml_set_param(ctx, layer.w1); 
        ggml_set_param(ctx, layer.w2);
        ggml_set_param(ctx, layer.w3); 
    }
}


static void randomize_model(struct llama_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;

    const uint32_t n_layer = hparams.n_layer;

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    randomize_tensor_normal(model->tok_embeddings , rnd);
    randomize_tensor_normal(model->norm           , rnd);
    randomize_tensor_normal(model->output         , rnd);

    for (uint32_t i = 0; i < n_layer; ++i) {
        auto & layer = model->layers[i];
        randomize_tensor_normal(layer.attnetion_norm, rnd);

        randomize_tensor_normal(layer.wq, rnd);
        randomize_tensor_normal(layer.wk, rnd);
        randomize_tensor_normal(layer.wv, rnd);
        randomize_tensor_normal(layer.wo, rnd);

        randomize_tensor_normal(layer.ffn_norm, rnd);

        randomize_tensor_normal(layer.w1, rnd);
        randomize_tensor_normal(layer.w2, rnd);
        randomize_tensor_normal(layer.w3, rnd);
    }

    free_random_normal_distribution(rnd);
}


void init_kv_cache(llama_kv_cache * cache, struct llama_model *model, int n_batch){
    const auto &hparams = model->hparams; 
    
    const uint32_t n_ctx = hparams.n_ctx; 
    const uint32_t n_layer = hparams.n_layer; 
    const uint32_t n_embd = hparams.n_embd; 
    
    
    const int64_t n_mem = n_layer*n_ctx*n_batch; 
    const uint64_t n_elements = n_embd*n_mem; 
    
    
    if (!cache->ctx){
        struct ggml_init_params params; 
        params.mem_size   = 2u*n_elements*ggml_type_size(GGML_TYPE_F32) + 2u*1024*1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        cache->ctx = ggml_init(params); 

        if (!cache->ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            exit(1);
        }
    }
    
    cache->k = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements); 
    cache->v = ggml_new_tensor_1d(cache->ctx, GGML_TYPE_F32, n_elements); 
    
}

static void get_example_targets(int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = targets->ne[0];
    float randomness = 0.0f;
    // ggml_set_zero(targets);
    ggml_set_f32(targets, -1.0f);
    ggml_set_i32_1d(tokens_input, 0, 0);
    for (int i=1; i<n_tokens+1; ++i) {
        float x = example_id + i * 3.14159f * 2.0f * 1.0f * 0.5f / n_tokens;
        float y = sinf(x);//*cosf(x*1.1f+1.0f);
        float z = (y+1.0f)*0.5f; // scale to [0..1]
        z += (frand()-0.5f)*(randomness/n_vocab);
        z = (z < 0.0f) ? 0.0f : (z > 1.0f) ? 1.0f : z; // clamp to [0..1]
        int token = std::max(1,std::min(1+(int)(z*(float)(n_vocab-1)), n_vocab-1));
        ggml_set_f32_1d(targets, (i-1)*n_vocab + token, +1.0f);
        if (i<n_tokens) {
            ggml_set_i32_1d(tokens_input, i, token);
        }
    }
}



static void get_examples_targets_batch(
    struct ggml_context * ctx, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets
) {

    GGML_ASSERT(ggml_is_matrix(tokens_input)); 
    GGML_ASSERT(ggml_is_3d(targets)); 
    int n_tokens = tokens_input->ne[0]; 
    int n_batch = tokens_input->ne[1]; 
    GGML_ASSERT(n_tokens == targets->ne[1]); 
    GGML_ASSERT(n_batch == targets->ne[2]); 
    

    for(int k=0; k < n_batch; ++k){
        struct ggml_tensor * tokens_input_k = ggml_view_1d(ctx, tokens_input, tokens_input->ne[0], k*tokens_input->nb[1]);
        struct ggml_tensor * targets_k = ggml_view_2d(ctx, targets, targets->ne[0], targets->ne[1], targets->nb[1], k*targets->nb[2]);
        get_example_targets(example_id*n_batch + k, tokens_input_k, targets_k); 

    }
}


void build_kv_store(ggml_context * ctx, llama_model * model, llama_kv_cache & cache, ggml_cgraph * graph, ggml_tensor * k_cur, ggml_tensor * v_cur, int n_ctx, int n_tokens, int n_batch, int n_layer, int n_embd, int n_past, int il){

    //k_cur shape: [n_embd, n_tokens*n_batch]
    //v_cur shape: [n_embd, n_tokens*n_batch]
    {
        
        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, cache.k, n_tokens*n_embd*n_batch, (ggml_element_size(cache.k)*n_embd)*(il*n_ctx*n_batch + n_past));
        struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, cache.v, n_tokens*n_embd, n_batch, (ggml_element_size(cache.v)*n_ctx), (il*n_ctx*n_batch*n_embd*ggml_element_size(cache.v) + n_past*n_embd*ggml_element_size(cache.v)));

        
        ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view)); 
        ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view)); 


    }
}


//Batch Processing and Generation Functions 
static struct ggml_tensor * forward_batch(
    struct llama_model    * model, 
    struct llama_kv_cache * cache, 
    struct ggml_context   * ctx_compute,
    struct ggml_cgraph    * gf, 
    struct ggml_tensor    * tokens_input, 
    const  int              n_tokens, 
    const  int              n_past, 
    const  id_t             n_batch 
) {
    //Step0: Prepare kv cache and all needed constants.  
    const int N = n_tokens;
    
    struct llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;
    const int n_ff    = get_n_ff(&hparams);

    //Step1: prepare tensors: 
    //  1. copy the input tokens of the batch into a one dimensional tensor
    //  2. get the k and v cache. 
    //  3. Set the KQ positions of the current batch. This is a 1d tensor, where each position corresponds to all 
    //      the tokens in that position in batch (1d column of size n_batch). This is computed easily by 
    //      (kq_pos_i = i + n_past (n_past = the number of tokens processed so far in the sequence))
    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, N*n_batch); 
    memcpy(tokens->data, tokens_input->data, ggml_element_size(tokens)*N*n_batch); 

    struct ggml_tensor * kc = kv_self.k; 
    struct ggml_tensor * vc = kv_self.v; 

    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, N); 
    {
        int * data = (int *) KQ_pos->data; 
        for (int i=0; i<N; ++i){
            data[i] = i + n_past; 
        }
    }   
    
    //Step2: Get the embedding Vectors (inpL) for tokens. This is a tensor of size (n_embed, n_tokens*n_batch, 1)
    struct ggml_tensor * inpL = ggml_get_rows(ctx_compute, model->tok_embeddings, tokens); 
    assert_shape_2d(inpL, n_embd, N*n_batch); 
    
    //Step3: Loop through the layers, and apply them one by one 
    for (int il=0; il < n_layer; ++il){
        //1. Prepare some tensors: 
            //a. Copy embedding vector, this is residual stream we will add later
            //b. Initializne a cur tensor, that will hold the data of computation (?) 
            struct ggml_tensor *inpSA = inpL; 
            struct ggml_tensor * cur; 
            
        //2. Normalize the embedding, and apply the attention norm. 
        {
            cur = ggml_rms_norm(ctx_compute,inpL, rms_norm_eps); 
            assert_shape_2d(cur, n_embd, N*n_batch); 

            ggml_mul(ctx_compute, 
                    ggml_repeat(ctx_compute, model->layers[il].attnetion_norm, cur), 
                    cur);
            assert_shape_2d(cur, n_embd, N*n_batch);  
        }
        
        //3. Apply self attention (This is a big block):
        {
            //a. Compute Q K and RoPe them with the KQ positions. 
            struct ggml_tensor * Qcur = ggml_rope(ctx_compute, ggml_reshape_4d(ctx_compute, ggml_mul_mat(ctx_compute, model->layers[il].wq, cur), n_embd/n_head, n_head, N, n_batch), KQ_pos, n_rot, 0, 0); 
            struct ggml_tensor * Kcur = ggml_rope(ctx_compute, ggml_reshape_4d(ctx_compute, ggml_mul_mat(ctx_compute, model->layers[il].wk, cur), n_embd/n_head, n_head, N, n_batch), KQ_pos, n_rot, 0, 0); 
            assert_shape_4d(Qcur, n_embd/n_head, n_head, N, n_batch);
            assert_shape_4d(Kcur, n_embd/n_head, n_head, N, n_batch); 
            
            //b. Compute V vectors
            struct ggml_tensor * Vcur = ggml_cont(ctx_compute, ggml_permute(ctx_compute, ggml_reshape_3d(ctx_compute, ggml_mul_mat(ctx_compute, model->layers[il].wv, cur), n_embd, N, n_batch), 1, 0, 2, 3)); 
            assert_shape_3d(Vcur, N, n_embd, n_batch);

            //c. update the kv cache with the new keys and values. 
            build_kv_store(ctx_compute, model, kv_self, gf, Kcur, Vcur, n_ctx, n_tokens, n_batch, n_layer, n_embd, n_past, il);
            
            //d. Compute attention scores                
            //Get Current input Query tensor. 
            struct ggml_tensor * Q = ggml_permute(ctx_compute, Qcur, 0, 2, 1, 3); 
            assert_shape_4d(Q, n_embd/n_head, N, n_head, n_batch); 
            
            //Get cached Key tensor. 
            struct ggml_tensor * K  = ggml_view_3d(ctx_compute, kc, n_embd, (n_past+N), n_batch, n_embd*ggml_element_size(kc), n_ctx*n_embd*ggml_element_size(kc), il*n_batch*n_ctx*n_embd*ggml_element_size(kc)); 
            K = ggml_reshape_4d(ctx_compute, K, n_embd/n_head, n_head, n_past+N, n_batch);
            K = ggml_permute(ctx_compute, K, 0, 2, 1, 3); 

            assert_shape_4d(K, n_embd/n_head, n_past+N, n_head, n_batch);  

            //Compute unnormalized scores matrix and scale them. 
            struct ggml_tensor * KQ = ggml_mul_mat(ctx_compute, K, Q); 
            assert_shape_4d(KQ, n_past+N, N, n_head, n_batch); 

            struct ggml_tensor * KQ_scaled = ggml_scale(ctx_compute, KQ, 1.0f/sqrtf(float(n_embd)/n_head));
            assert_shape_4d(KQ_scaled, n_past+N, N, n_head, n_batch); 


            //Mask unnormalized attention scores. 
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx_compute, KQ_scaled, n_past);
            assert_shape_4d(KQ_masked, n_past+N, N, n_head, n_batch);  

            
            //Apply softmax on KQ_scaled, taking masking into account. 
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx_compute,KQ_masked); 
            assert_shape_4d(KQ_soft_max, n_past+N, N, n_head, n_batch); 
        
            //e.Compute KQV tensor 
            //Get and reshape V tensor. 
            struct ggml_tensor * V = ggml_view_4d(ctx_compute, vc, 
                        n_past + N, n_embd/n_head, n_head, n_batch, 
                        ggml_element_size(vc) * n_ctx, 
                        ggml_element_size(vc) * n_ctx * n_embd/n_head, 
                        ggml_element_size(vc)*n_ctx*n_embd, 
                        il*n_batch*n_ctx*n_embd*ggml_element_size(vc)); 
            assert_shape_4d(V, n_past+N, n_embd/n_head, n_head, n_batch); 

            //Compute KQV 
            struct ggml_tensor * KQV = ggml_mul_mat(ctx_compute, V, KQ_soft_max); 
            assert_shape_4d(KQV, n_embd/n_head, N, n_head, n_batch); 

            //Merge heads and assign back to cur tensor. 
            struct ggml_tensor * KQV_merged = ggml_permute(ctx_compute, KQV, 0, 2, 1, 3);
            assert_shape_4d(KQV_merged, n_embd/n_head, n_head, N, n_batch);  

            cur = ggml_reshape_2d(ctx_compute, ggml_cont(ctx_compute, KQV_merged), n_embd, N*n_batch); 
            assert_shape_2d(cur, n_embd, N*n_batch); 
            

            //Apply the projection. 
            cur = ggml_mul_mat(ctx_compute, model->layers[il].wo, cur); 
            assert_shape_2d(cur, n_embd, N*n_batch); 
        }   
        
        //4. add back cur and residual stream after attention computations. 
        struct ggml_tensor * inpFF = ggml_add(ctx_compute, cur, inpSA); 
        assert_shape_2d(inpFF, n_embd, N*n_batch); 
        
        //5. Compute Feed-Foeward network:
        {
            //a.Normalize
            {
                cur = ggml_rms_norm(ctx_compute, inpFF, rms_norm_eps); 
                assert_shape_2d(cur, n_embd, N*n_batch); 

                cur = ggml_mul(ctx_compute, 
                        ggml_repeat(ctx_compute, model->layers[il].ffn_norm, cur), 
                        cur); 
                assert_shape_2d(cur, n_embd, N*n_batch); 
                
            }
            //b.Apply feed forwards 
            struct ggml_tensor * tmp = ggml_mul_mat(ctx_compute, model->layers[il].w3, cur); 
            assert_shape_2d(tmp, n_ff, N*n_batch); 

            cur = ggml_mul_mat(ctx_compute, model->layers[il].w1, cur); 
            assert_shape_2d(cur, n_ff, N*n_batch); 

            //Multi;py element wise (?) 
            cur = ggml_mul(ctx_compute, cur, tmp); 
            assert_shape_2d(cur, n_ff, N*n_batch);

            cur = ggml_mul_mat(ctx_compute, model->layers[il].w2, cur); 
            assert_shape_2d(cur, n_embd, N*n_batch); 
        }   //cur shape [n_embd, N*n_batch, 1, 1]
            
        //6/Add back curr and result of Feed-Forward. 
        cur = ggml_add(ctx_compute, cur, inpFF); 
        assert_shape_2d(cur, n_embd, N*n_batch); 

        inpL = cur; 
        assert_shape_2d(inpL, n_embd, N*n_batch); 
        
    }
            
    //Step4: Normalzie the output of the loop. 
    {
        inpL = ggml_rms_norm(ctx_compute, inpL, rms_norm_eps); 
        assert_shape_2d(inpL, n_embd, N*n_batch); 

        inpL = ggml_mul(ctx_compute, ggml_repeat(ctx_compute, model->norm, inpL), inpL); 
        assert_shape_2d(inpL, n_embd, N*n_batch); 
        
    }

    //Step5: Apply LM head.
    inpL = ggml_mul_mat(ctx_compute, model->output, inpL); 
    assert_shape_2d(inpL, n_vocab, N*n_batch);

    {
        inpL = ggml_reshape_3d(ctx_compute, inpL, n_vocab, N, n_batch);
        assert_shape_3d(inpL, n_vocab, N, n_batch); 
    } 
    
    //Step6: Build and run the computational graph. Return result tensor. 
    ggml_build_forward_expand(gf, inpL); 
    
    return inpL; //Shape: (n_vocab, N, n_batch), N = n_tokens = the size of each batch. 
}



static struct ggml_tensor * forward(
    struct llama_model    * model,
    struct llama_kv_cache * cache,
    struct ggml_context   * ctx_compute,
    struct ggml_cgraph    * gf,
    struct ggml_tensor    * tokens_input,
    const  int              n_tokens,
    const  int              n_past
) {
    const int N = n_tokens;
    
    struct llama_kv_cache& kv_self = *cache;
    const auto & hparams = model->hparams;
    const int n_ctx   = hparams.n_ctx;
    const int n_vocab = hparams.n_vocab;
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_head  = hparams.n_head;
    const int n_rot   = hparams.n_rot;
    const int n_ff    = get_n_ff(&hparams);

    struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, N);
    memcpy(tokens->data, tokens_input->data, N*ggml_element_size(tokens));

    struct ggml_tensor * kc = kv_self.k; 
    struct ggml_tensor * vc = kv_self.v;

    struct ggml_tensor * KQ_pos = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, N); 
    {
        int * data = (int *) KQ_pos->data; 
        for (int i=0; i<N; ++i){
            data[i] = i + n_past; 
        }
    }

    struct ggml_tensor *  inpL = ggml_get_rows(ctx_compute, model->tok_embeddings, tokens);
    assert_shape_2d(inpL, n_embd, N);
    
    for (int il=0; il < n_layer; ++il){
        struct ggml_tensor * inpSA = inpL; 
        assert_shape_2d(inpSA, n_embd, N);
        
        struct ggml_tensor * cur; 

        //Normalize
        {
            cur = ggml_rms_norm(ctx_compute, inpL, rms_norm_eps);
            assert_shape_2d(cur, n_embd, N);

            cur = ggml_mul(ctx_compute, ggml_repeat(ctx_compute, model->layers[il].attnetion_norm, cur), cur);
            assert_shape_2d(cur, n_embd, N);
        }

        //self attention 
        {
            struct ggml_tensor * Qcur = ggml_rope(ctx_compute, ggml_reshape_3d(ctx_compute, ggml_mul_mat(ctx_compute, model->layers[il].wq, cur), n_embd/n_head, n_head, N), KQ_pos, n_rot, 0, 0);
            struct ggml_tensor * Kcur = ggml_rope(ctx_compute, ggml_reshape_3d(ctx_compute, ggml_mul_mat(ctx_compute, model->layers[il].wk, cur), n_embd/n_head, n_head, N), KQ_pos, n_rot, 0, 0);
            assert_shape_3d(Qcur, n_embd/n_head, n_head, N);
            assert_shape_3d(Kcur, n_embd/n_head, n_head, N);

            //Store new KV 
            {
                struct ggml_tensor *Vcur = ggml_mul_mat(ctx_compute, model->layers[il].wv, cur); 
                assert_shape_2d(Vcur, n_embd, N);

                kc = ggml_set_1d(ctx_compute, kc, ggml_reshape_1d(ctx_compute, Kcur, n_embd*N), (ggml_element_size(kv_self.k)*n_embd) * (il*n_ctx+n_past));
            }
        }
    }
    return ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, n_vocab, N);
}




static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}


// Debugging Functions 
void dump_model_hparams(llama_model model){
    printf("model.hparams.n_vocab: %u\n", model.hparams.n_vocab); 
    printf("model.hparams.n_ctx: %u\n", model.hparams.n_ctx);
    printf("model.hparams.n_embd: %u\n", model.hparams.n_embd);
    printf("model.hparams.mult: %u\n", model.hparams.n_mult);
    printf("model.hparams.n_head: %u\n", model.hparams.n_head);
    printf("model.hparams.n_layer: %u\n", model.hparams.n_layer);
    printf("model.hparams.n_rot: %u\n", model.hparams.n_rot);
}


void print_memory_usage(ggml_context *ctx){
    size_t mem_used = ggml_used_mem(ctx); 
    printf("Memory allocated: %zu\n", mem_used);
}


int main(int argc, char ** argv){
    //Initializing Model 
    struct ggml_init_params lcparams; 
    lcparams.mem_size = 1024ll * 1024ll * 1024ll; 
    lcparams.mem_buffer = NULL; 
    lcparams.no_alloc = false; 

    struct llama_model model; 
    model.hparams.n_rot = std::min(16u, model.hparams.n_embd / model.hparams.n_head);
    dump_model_hparams(model);

    model.ctx = ggml_init(lcparams); 

    printf("Before model init\n" );
    print_memory_usage(model.ctx);
    
    init_model(&model);

    printf("After model init\n" );
    print_memory_usage(model.ctx);
    
    set_param_model(&model); 
    randomize_model(&model, 1337, 0.0f, 1.0f, -1.0f, +1.0f);

    /**********Batch Processing and Generation**************/ 

    //Batch Processing 
    int n_batch = 8; 

    struct llama_kv_cache kv_self; 
    printf("kv cache init\n");
    kv_self.ctx = model.ctx;
    init_kv_cache(&kv_self, &model, n_batch); 
    int n_tokens = model.hparams.n_ctx; 
    int n_vocab = model.hparams.n_vocab; 
    
    
    size_t compute_size = 1024ll*1024ll*1024ll; 
    uint8_t * compute_addr = new uint8_t[compute_size]; 
    std::vector<uint8_t> work_buffer;

    //Batch Processing 
    {
        struct ggml_init_params params_compute = {
            compute_size, 
            compute_addr,
            false, 
        }; 

        struct ggml_context * ctx_compute = ggml_init(params_compute); 

        struct ggml_tensor * tokens_input = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * targets = ggml_new_tensor_3d(ctx_compute, GGML_TYPE_F32, n_vocab, n_tokens, n_batch); 

        struct ggml_cgraph *gf = NULL; 
        gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true); 

        get_examples_targets_batch(ctx_compute, 64, tokens_input, targets); 


        int n_past = 0; 
        struct ggml_tensor * logits = forward_batch(&model, &kv_self, ctx_compute, gf, tokens_input, n_tokens, n_past, n_batch); 

        ggml_graph_compute_helper(work_buffer, gf, /*n_threads*/ 1);
        ggml_free(ctx_compute); 
    }
        dump_ggml_tensor_f32_1d(kv_self.k, 8, -1);                

    
    //Generation 
    int n_gen = 128; 
    int sample_ctx = n_tokens - n_tokens/8; 
    
    struct ggml_tensor * tokens_input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * targets = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab, n_tokens);

    get_example_targets(137, tokens_input, targets); 
    assert_shape_1d(tokens_input, n_tokens);
    assert_shape_2d(targets, n_vocab, n_tokens);
    
    for(int i=sample_ctx; i<n_tokens; ++i){ 
        ggml_set_i32_1d(tokens_input, i, n_vocab/2); 
    }

    //start generation 
    for (int i=0; i<sample_ctx-1; ++i){
        struct ggml_init_params params = {
            compute_size, 
            compute_addr, 
            false
        }; 

        struct ggml_context * ctx_compute = ggml_init(params);
        
        struct ggml_cgraph * gf = NULL; 
        gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);

        int n_past = 0; 
        struct ggml_tensor * logits = forward(&model, &kv_self, ctx_compute, gf, tokens_input, n_tokens, n_past);

        //ggml_build_forward_expand(gf, logits);
        //ggml_graph_compute_helper(work_buffer, gf, 1);

        struct ggml_tensor * best_samples = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, sample_ctx);
        struct ggml_tensor * probs = ggml_new_tensor_2d(ctx_compute, GGML_TYPE_F32, n_vocab, sample_ctx);

        //sample_softmax(logits, probs, best_samples); 
        
        int token = ggml_get_i32_1d(best_samples, sample_ctx-1);

        //lshift_examples(tokens_input, targets, 1); 
        //ggml_set_i32_1d(tokens_input, n_tokens-1, token);
        //ggml_set_i32_1d(tokens_input, sample_ctx-1, token);
        
        ggml_free(ctx_compute); 
    }
    
    

    ggml_free(model.ctx); 
    printf("End of my_baby_llama.cpp\n"); 
    return 0;
}