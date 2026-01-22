#ifndef ELO_H
#define ELO_H

#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "consts.h"

class EloCalculator {
public:
    EloCalculator(const std::string& model_path, const std::vector<std::string>& model_names, bool load_from_file=true, bool adjust_K=false);

    void UpdateRatings(int player_a, int player_b, const std::vector<bool>& result);

    std::vector<float> GetRatings(bool adjust=false); // adjust = true 면 최소 레이팅 0으로 고정

    void saveRating(const std::string& model_path, const std::vector<std::string>& model_names);

private:
    std::vector<float> ratings;
    std::vector<int> game_counts;
    std::vector<std::string> model_names; 
    bool adjust_K;
};

#endif