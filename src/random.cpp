#include "random.h"
#include <iostream>

std::random_device rd;
std::mt19937 gen(rd());

std::vector<int> select_indices(int range, int many) {
    std::vector<int> indices(range);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    std::vector<int> selected_cols(indices.begin(), indices.begin() + many);
    // std::cout << indices[0] << std::endl;
    return selected_cols;
}
