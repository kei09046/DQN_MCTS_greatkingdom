#include "elo.h"

EloCalculator::EloCalculator(const std::string& model_path, const std::vector<std::string>& model_names, bool load_from_file, bool adjust_K): 
model_names(model_names), adjust_K(adjust_K)
{
    if(!load_from_file){
        for(auto& name : model_names){
            ratings.push_back(globalConfig.initRating);
            game_counts.push_back(0);
        }
        return;
    }

    std::unordered_map<std::string, std::pair<float, int>> loaded_models;
    for(auto& name : model_names){
        loaded_models[name] = {globalConfig.initRating, 0};
    }

    std::ifstream file(model_path);
    if(file){
        std::string name;
        float rating;
        int games;
        while(file >> name >> rating >> games){
            auto it = loaded_models.find(name);
            if(it != loaded_models.end()){
                it->second = {rating, games};
            }
        }
    }

    for(auto& name : model_names){
        ratings.push_back(loaded_models[name].first);
        game_counts.push_back(loaded_models[name].second);
    }
}

void EloCalculator::UpdateRatings(int player_a, int player_b, const std::vector<bool>& result) { // result : true -> player_a win, false -> player_b win
    float rating_a = ratings[player_a];
    float rating_b = ratings[player_b];
    float a_K = globalConfig.initK / (1.0f + game_counts[player_a] / 30.0f);
    float b_K = globalConfig.initK / (1.0f + game_counts[player_b] / 30.0f);

    for (bool win : result) {
        float ea = 1.0f / (1.0f + pow(10.0f, (rating_b - rating_a) / 400.0f));
        float sa = (win ? 1.0f : 0.0f);
        float sb = 1.0f - sa;
        ratings[player_a] += a_K * (sa - ea);
        ratings[player_b] += b_K * (sb - (1.0 - ea));
    }
    game_counts[player_a] += result.size();
    game_counts[player_b] += result.size();
}

std::vector<float> EloCalculator::GetRatings(bool adjust){
    if (!adjust) {
        return ratings;
    }

    float min_rating = *std::min_element(ratings.begin(), ratings.end());
    for(int i = 0; i < ratings.size(); ++i){
        ratings[i] -= min_rating;
    }

    return ratings;
}

void EloCalculator::saveRating(const std::string& model_path, const std::vector<std::string>& model_names) {
    // TODO : update the file
    std::unordered_map<std::string, std::pair<float, int>> updated_ratings;
    std::unordered_map<std::string, bool> updated;
    std::ifstream file(model_path);
    std::vector<std::string> lines;
    

    for(int i=0; i<model_names.size(); ++i){
        updated_ratings[model_names[i]] = {ratings[i], game_counts[i]};
    }

    if(file){
        std::string line;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            std::string name;
            if(!(iss >> name)) continue;

            if(updated_ratings.count(name) != 0){
                std::ostringstream oss;
                oss << name << " " << updated_ratings[name].first << " " << updated_ratings[name].second;
                lines.push_back(oss.str());
                updated[name] = true;
            }
            else{
                lines.push_back(line);
            }
        }
    }

    for(const auto& [name, rating_pair] : updated_ratings){
        if(updated.count(name) == 0){
            std::ostringstream oss;
            oss << name << " " << rating_pair.first << " " << rating_pair.second;
            lines.push_back(oss.str());
        }
    }

    std::ofstream fout(model_path, std::ios::trunc);
    for(const auto& line : lines){
        fout << line << "\n";
    }
}