#include "elo.h"
#include <cctype>

namespace {
    // "natural" order : digit runs compare by numeric value, everything else character by character.
    // e.g. M_12b_15360.pt < M_12b_60480B.pt < M_12b_111360B.pt
    bool natural_less(const std::string& a, const std::string& b){
        size_t i = 0, j = 0;
        while(i < a.size() && j < b.size()){
            if(std::isdigit(static_cast<unsigned char>(a[i])) && std::isdigit(static_cast<unsigned char>(b[j]))){
                size_t ie = i, je = j;
                while(ie < a.size() && std::isdigit(static_cast<unsigned char>(a[ie]))) ++ie;
                while(je < b.size() && std::isdigit(static_cast<unsigned char>(b[je]))) ++je;

                // compare numerically without overflow : skip leading zeros, then longer run is bigger.
                size_t ia = i, jb = j;
                while(ia + 1 < ie && a[ia] == '0') ++ia;
                while(jb + 1 < je && b[jb] == '0') ++jb;
                if(ie - ia != je - jb)
                    return ie - ia < je - jb;
                const int c = a.compare(ia, ie - ia, b, jb, je - jb);
                if(c != 0)
                    return c < 0;
                i = ie;
                j = je;
            }
            else{
                if(a[i] != b[j])
                    return a[i] < b[j];
                ++i;
                ++j;
            }
        }
        return (a.size() - i) < (b.size() - j);
    }

    std::string first_token(const std::string& line){
        std::istringstream iss(line);
        std::string name;
        iss >> name;
        return name;
    }
}

EloCalculator::EloCalculator(const std::optional<std::string>& ratingPath, const std::vector<std::string>& modelNames, bool adjustK): 
modelNames(modelNames), adjustK(adjustK)
{
    if(!ratingPath.has_value()){
        for(int i=0; i<modelNames.size(); ++i){
            ratings.push_back(globalConfig.initRating);
            gameCounts.push_back(0);
        }
        return;
    }

    std::unordered_map<std::string, std::pair<float, int>> loadedModels;
    for(auto& name : modelNames){
        loadedModels[name] = {globalConfig.initRating, 0};
    }

    std::ifstream file(ratingPath.value());
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

// game-by-game elo update.
// result : true -> player_a win, false -> player_b win
void EloCalculator::update_ratings(int playerA, int playerB, const std::vector<bool>& result) {
    float ratingA = ratings[playerA];
    float ratingB = ratings[playerB];

    const float aK = globalConfig.initK;
    const float bK = globalConfig.initK;

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

// batched elo update.
void EloCalculator::update_ratings(int playerA, int playerB, int nGames, float aWins) {
    float ratingA = ratings[playerA];
    float ratingB = ratings[playerB];

    const float aK = globalConfig.initK;
    const float bK = globalConfig.initK;
    const float ea = 1.0f / (1.0f + pow(10.0f, (ratingB - ratingA) / 400.0f));

    ratings[playerA] += aK * (aWins - nGames * ea);
    ratings[playerB] += bK * ((nGames - aWins) - nGames * (1.0f - ea));

    gameCounts[playerA] += nGames;
    gameCounts[playerB] += nGames;
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

void EloCalculator::save_rating(const std::optional<std::string>& ratingPath, const std::vector<std::string>& modelNames) {
    if(!ratingPath.has_value())
        return;

    std::unordered_map<std::string, std::pair<float, int>> updatedRatings;
    std::unordered_map<std::string, bool> updated;
    std::ifstream file(ratingPath.value());
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

    // models are listed in natural order of their names (smaller numbers first).
    std::stable_sort(lines.begin(), lines.end(), [](const std::string& x, const std::string& y){
        return natural_less(first_token(x), first_token(y));
    });

    std::ofstream fout(ratingPath.value(), std::ios::trunc);
    for(const auto& line : lines){
        fout << line << "\n";
    }
    fout.close();
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