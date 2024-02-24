#include "ggml/ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

struct ptmodel_hparams {
    int32_t n_input   = 2;
    int32_t n_hidden  = 16;
    int32_t n_output = 1;
};

struct ptmodel {
    ptmodel_hparams hparams;
    struct ggml_tensor * fc1_weight;
    struct ggml_tensor * fc1_bias;
    struct ggml_tensor * fc2_weight;
    struct ggml_tensor * fc2_bias;
    struct ggml_context * ctx;
};

const char * file_name = "../assets/model.bin";

bool load(const std::string & fname, ptmodel & model) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_input   = hparams.n_input;
        const int n_hidden  = hparams.n_hidden;
        const int n_output = hparams.n_output;

        ctx_size += n_input * n_hidden * ggml_type_size(GGML_TYPE_F32);
        ctx_size +=           n_hidden * ggml_type_size(GGML_TYPE_F32);

        ctx_size += n_hidden * n_output * ggml_type_size(GGML_TYPE_F32);
        ctx_size +=            n_output * ggml_type_size(GGML_TYPE_F32);

        printf("%s: ggml ctx size = %.2ld bytes\n", __func__, ctx_size);
    }

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size + 1024*1024,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

        {
            int32_t ne_weight[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
            }

            model.hparams.n_input  = ne_weight[0];
            model.hparams.n_hidden = ne_weight[1];

            model.fc1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_input, model.hparams.n_hidden);
            fin.read(reinterpret_cast<char *>(model.fc1_weight->data), ggml_nbytes(model.fc1_weight));
            ggml_set_name(model.fc1_weight, "fc1_weight");
        }

        {
            int32_t ne_bias[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
            }

            model.fc1_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_hidden);
            fin.read(reinterpret_cast<char *>(model.fc1_bias->data), ggml_nbytes(model.fc1_bias));
            ggml_set_name(model.fc1_bias, "fc1_bias");

            model.fc1_bias->op_params[0] = 0xdeadbeef;
        }
    }

    {
        int32_t n_dims;
        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));

        {
            int32_t ne_weight[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_weight[i]), sizeof(ne_weight[i]));
            }

            model.hparams.n_output = ne_weight[1];

            model.fc2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.hparams.n_hidden, model.hparams.n_output);
            fin.read(reinterpret_cast<char *>(model.fc2_weight->data), ggml_nbytes(model.fc2_weight));
            ggml_set_name(model.fc2_weight, "fc2_weight");
        }

        {
            int32_t ne_bias[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne_bias[i]), sizeof(ne_bias[i]));
            }

            model.fc2_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_output);
            fin.read(reinterpret_cast<char *>(model.fc2_bias->data), ggml_nbytes(model.fc2_bias));
            ggml_set_name(model.fc2_bias, "fc2_bias");
        }
    }

    fin.close();

    return true;
}

float predict(const ptmodel & model, std::vector<float> table_input) {
    const auto & hparams = model.hparams;

    static size_t buf_size = hparams.n_input * sizeof(float) * 1024 * 1024;
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.n_input);
    memcpy(input->data, table_input.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");

    ggml_tensor * fc1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc1_weight, input),                model.fc1_bias);
    ggml_tensor * fc2 = ggml_add(ctx0, ggml_mul_mat(ctx0, model.fc2_weight, ggml_relu(ctx0, fc1)), model.fc2_bias);

    ggml_tensor * probs = ggml_hardsigmoid(ctx0, fc2);
    ggml_set_name(probs, "probs");

    ggml_build_forward_expand(gf, probs);
    ggml_graph_compute_with_ctx(ctx0, gf, 1);

    ggml_graph_print(gf);
    // ggml_graph_dump_dot(gf, NULL, "model.dot");
    // ggml_graph_export(gf, "model.ggml");

    const float * probs_data = ggml_get_data_f32(probs);

    ggml_free(ctx0);

    return probs_data[0];
}

int main() {
    srand(time(NULL));
    ggml_time_init();

    ptmodel model;

    {
        const int64_t t_start_us = ggml_time_us();

        if (!load(file_name, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, file_name);
            return 1;
        }

        const int64_t t_load_us = ggml_time_us() - t_start_us;

        fprintf(stdout, "%s: loaded model in %.2f ms\n", __func__, t_load_us / 1000.0f);
    }

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            std::vector<float> input = {float(i), float(j)};
            const float prediction = predict(model, input);
            fprintf(stdout, "%s: predicted value for [%d %d]: %f\n", __func__, i, j, prediction);
        }
    }

    ggml_free(model.ctx);

    return 0;
}