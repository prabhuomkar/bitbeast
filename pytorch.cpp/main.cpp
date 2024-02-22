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
    int32_t n_classes = 1;
};

struct ptmodel {
    ptmodel_hparams hparams;
    struct ggml_tensor * fc1_weight;
    struct ggml_tensor * fc1_bias;
    struct ggml_tensor * fc2_weight;
    struct ggml_tensor * fc2_bias;
    struct ggml_context * ctx;
};

int main(int argc, char ** argv) {
    return 0;
}