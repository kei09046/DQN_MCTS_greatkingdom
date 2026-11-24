#ifndef ELO_H
#define ELO_H

#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <optional>
#include <map>
#include "consts.h"
#include "config.h"

// "natural" order : digit runs compare by numeric value, everything else character by character.
// e.g. M_12b_15360.pt < M_12b_60480B.pt < M_12b_111360B.pt
bool natural_less(const std::string& a, const std::string& b);

class EloCalculator {
public:
    EloCalculator(const RatingConfig& cfg, const std::optional<std::string>& ratingPath, const std::vector<std::string>& modelNames,
        bool adjustK=false);

    void update_ratings(int playerA, int playerB, const std::vector<bool>& result);

    void update_ratings(int playerA, int playerB, int nGames, float aWins);

    std::vector<float> get_ratings(bool adjust=false); // adjust = true 면 최소 레이팅 0으로 고정

    void save_rating(const std::optional<std::string>& ratingPath, const std::vector<std::string>& modelNames);

    std::vector<float> calculate_rating(const std::vector<std::vector<float>> winrate);

    void set_rating(int idx, float rating, int games);

private:
    std::vector<float> ratings;
    std::vector<int> gameCounts;
    RatingConfig cfg;
    std::vector<std::string> modelNames; 
    bool adjustK;
};

// accumulated results of every model pair ever compared, stored one line per pair : "modelA modelB winsA games"
// (modelA before modelB in natural order, lines sorted the same way). Ratings are fitted from all of them at once.
class PairResults {
public:
    // path == nullopt : nothing is loaded or saved.
    explicit PairResults(const std::optional<std::string>& path);

    void add(const std::string& a, const std::string& b, int winsA, int games);

    void save() const;

    // Bradley-Terry maximum likelihood over every stored pair (each pair gets `prior` virtual wins per side,
    // so 0% / 100% results stay finite). The first model in natural order is fixed at initRating.
    // Returns name -> (rating, games played), for every model in the stored results.
    std::map<std::string, std::pair<float, int>> fit(float initRating, double prior = 1.0) const;

private:
    struct NaturalLess {
        bool operator()(const std::string& a, const std::string& b) const { return natural_less(a, b); }
    };
    using Key = std::pair<std::string, std::string>;
    struct KeyLess {
        bool operator()(const Key& x, const Key& y) const {
            if(x.first != y.first) return natural_less(x.first, y.first);
            return natural_less(x.second, y.second);
        }
    };

    std::optional<std::string> path;
    std::map<Key, std::pair<int, int>, KeyLess> pairs; // (a, b) -> (wins of a, games)
};

#endif