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
    EloCalculator(const std::string& modelPath, const std::vector<std::string>& modelNames, bool loadFromFile=true, bool adjustK=false);

    void update_ratings(int playerA, int playerB, const std::vector<bool>& result);

    std::vector<float> get_ratings(bool adjust=false); // adjust = true 면 최소 레이팅 0으로 고정

    void save_rating(const std::string& modelPath, const std::vector<std::string>& modelNames);

    std::vector<float> calculate_rating(const std::vector<std::vector<float>> winrate);

private:
    std::vector<float> ratings;
    std::vector<int> gameCounts;
    std::vector<std::string> modelNames; 
    bool adjustK;
};

#endif