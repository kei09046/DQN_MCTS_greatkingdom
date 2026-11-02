#include "elo.h"

EloCalculator::EloCalculator(const std::string& modelPath, const std::vector<std::string>& modelNames, bool loadFromFile, bool adjustK): 
modelNames(modelNames), adjustK(adjustK)
{
    if(!loadFromFile){
        for(auto& name : modelNames){
            ratings.push_back(globalConfig.initRating);
            gameCounts.push_back(0);
        }
        return;
    }

    std::unordered_map<std::string, std::pair<float, int>> loadedModels;
    for(auto& name : modelNames){
        loadedModels[name] = {globalConfig.initRating, 0};
    }

    std::ifstream file(modelPath);
    if(file){
        std::string name;
        float rating;
        int games;
        while(file >> name >> rating >> games){
            auto it = loadedModels.find(name);
            if(it != loadedModels.end()){
                it->second = {rating, games};
            }
        }
    }

    for(auto& name : modelNames){
        ratings.push_back(loadedModels[name].first);
        gameCounts.push_back(loadedModels[name].second);
    }
}

void EloCalculator::update_ratings(int playerA, int playerB, const std::vector<bool>& result) { // result : true -> player_a win, false -> player_b win
    float ratingA = ratings[playerA];
    float ratingB = ratings[playerB];

    // tried to decrease rating variance as model plays more games, but this seems to be inappropriate.
    // float a_K = globalConfig.initK / (1.0f + game_counts[player_a] / 30.0f);
    // float b_K = globalConfig.initK / (1.0f + game_counts[player_b] / 30.0f);

    float aK = globalConfig.initK;
    float bK = globalConfig.initK;

    for (bool win : result) {
        float ea = 1.0f / (1.0f + pow(10.0f, (ratingB - ratingA) / 400.0f));
        float sa = (win ? 1.0f : 0.0f);
        float sb = 1.0f - sa;
        ratings[playerA] += aK * (sa - ea);
        ratings[playerB] += bK * (sb - (1.0 - ea));
    }
    gameCounts[playerA] += result.size();
    gameCounts[playerB] += result.size();
}

std::vector<float> EloCalculator::get_ratings(bool adjust){
    if (!adjust) {
        return ratings;
    }

    float minRating = *std::min_element(ratings.begin(), ratings.end());
    for(int i = 0; i < ratings.size(); ++i){
        ratings[i] -= minRating;
    }

    return ratings;
}

void EloCalculator::save_rating(const std::string& modelPath, const std::vector<std::string>& modelNames) {
    // TODO : update the file
    std::unordered_map<std::string, std::pair<float, int>> updatedRatings;
    std::unordered_map<std::string, bool> updated;
    std::ifstream file(modelPath);
    std::vector<std::string> lines;
    

    for(int i=0; i<modelNames.size(); ++i){
        updatedRatings[modelNames[i]] = {ratings[i], gameCounts[i]};
    }

    if(file){
        std::string line;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string name;
            if(!(iss >> name)) continue;

            if(updatedRatings.count(name) != 0){
                std::ostringstream oss;
                oss << name << " " << updatedRatings[name].first << " " << updatedRatings[name].second;
                lines.push_back(oss.str());
                updated[name] = true;
            }
            else{
                lines.push_back(line);
            }
        }
    }

    for(const auto& [name, rating_pair] : updatedRatings){
        if(updated.count(name) == 0){
            std::ostringstream oss;
            oss << name << " " << rating_pair.first << " " << rating_pair.second;
            lines.push_back(oss.str());
        }
    }

    std::ofstream fout(modelPath, std::ios::trunc);
    for(const auto& line : lines){
        fout << line << "\n";
    }
}

std::vector<float> EloCalculator::calculate_rating(const std::vector<std::vector<float>> winrate){
    int n = winrate.size();
    std::vector<float> R(n, 0.0f); // initial ratings

    const double K = 0.01;    // learning rate
    const int ITER = 50000;   // gradient iterations

    for (int it = 0; it < ITER; ++it) {
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double p = winrate[i][j];
                if (p <= 0.0 || p >= 1.0) continue; // avoid log singularity

                double expected = 1.0 / (1.0 + pow(10.0, (R[j] - R[i]) / 400.0));
                double error = p - expected;

                R[i] += K * error * 400.0;
                R[j] -= K * error * 400.0;
            }
        }
    }

    return R;
}