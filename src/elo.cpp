#include "elo.h"
#include <cctype>

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

namespace {
    std::string first_token(const std::string& line){
        std::istringstream iss(line);
        std::string name;
        iss >> name;
        return name;
    }
}

EloCalculator::EloCalculator(const RatingConfig& cfg, const std::optional<std::string>& ratingPath,
    const std::vector<std::string>& modelNames, bool adjustK):
cfg(cfg), modelNames(modelNames), adjustK(adjustK)
{
    if(!ratingPath.has_value()){
        for(int i=0; i<modelNames.size(); ++i){
            ratings.push_back(cfg.initRating);
            gameCounts.push_back(0);
        }
        return;
    }

    std::unordered_map<std::string, std::pair<float, int>> loadedModels;
    for(auto& name : modelNames){
        loadedModels[name] = {cfg.initRating, 0};
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

    const float aK = cfg.initK;
    const float bK = cfg.initK;

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

    const float aK = cfg.initK;
    const float bK = cfg.initK;
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

void EloCalculator::set_rating(int idx, float rating, int games){
    ratings[idx] = rating;
    gameCounts[idx] = games;
}

PairResults::PairResults(const std::optional<std::string>& path) : path(path) {
    if(!path.has_value())
        return;
    std::ifstream file(path.value());
    std::string a, b;
    int wins, games;
    while(file >> a >> b >> wins >> games)
        add(a, b, wins, games);
}

void PairResults::add(const std::string& a, const std::string& b, int winsA, int games){
    if(natural_less(b, a)){
        add(b, a, games - winsA, games);
        return;
    }
    auto& rec = pairs[{a, b}];
    rec.first += winsA;
    rec.second += games;
}

void PairResults::save() const {
    if(!path.has_value())
        return;
    std::ofstream fout(path.value(), std::ios::trunc);
    for(const auto& [key, rec] : pairs)
        fout << key.first << " " << key.second << " " << rec.first << " " << rec.second << "\n";
}

namespace {
    // solves A x = b in place (A : n x n, row-major) by Gaussian elimination with partial pivoting.
    std::vector<double> solve_linear(std::vector<double> A, std::vector<double> b){
        const int n = b.size();
        for(int c = 0; c < n; ++c){
            int piv = c;
            for(int r = c + 1; r < n; ++r)
                if(std::abs(A[r * n + c]) > std::abs(A[piv * n + c])) piv = r;
            if(piv != c){
                for(int k = 0; k < n; ++k) std::swap(A[c * n + k], A[piv * n + k]);
                std::swap(b[c], b[piv]);
            }
            for(int r = c + 1; r < n; ++r){
                const double f = A[r * n + c] / A[c * n + c];
                if(f == 0.0) continue;
                for(int k = c; k < n; ++k) A[r * n + k] -= f * A[c * n + k];
                b[r] -= f * b[c];
            }
        }
        std::vector<double> x(n);
        for(int r = n - 1; r >= 0; --r){
            double v = b[r];
            for(int k = r + 1; k < n; ++k) v -= A[r * n + k] * x[k];
            x[r] = v / A[r * n + r];
        }
        return x;
    }
}

std::map<std::string, std::pair<float, int>> PairResults::fit(float initRating, double prior) const {
    std::map<std::string, int, NaturalLess> index;
    for(const auto& [key, rec] : pairs){
        index.emplace(key.first, 0);
        index.emplace(key.second, 0);
    }
    const int n = index.size();
    std::vector<std::string> names;
    for(auto& [name, i] : index){
        i = names.size();
        names.push_back(name);
    }

    std::vector<double> wins(n * n, 0.0), played(n * n, 0.0);
    std::vector<int> games(n, 0);
    for(const auto& [key, rec] : pairs){
        const int a = index[key.first], b = index[key.second];
        wins[a * n + b] += rec.first + prior;
        wins[b * n + a] += rec.second - rec.first + prior;
        games[a] += rec.second;
        games[b] += rec.second;
    }
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            played[i * n + j] = wins[i * n + j] + wins[j * n + i];

    // natural-log strengths; s[0] fixed at 0. A very weak pull toward 0 (sd ~ 17000 Elo) keeps the system solvable
    // when the pairs do not connect every model; it has no visible effect otherwise.
    const double ridge = 1e-4;
    std::vector<double> s(n, 0.0);
    const int m = n - 1;
    for(int it = 0; it < 100 && m > 0; ++it){
        std::vector<double> hess(m * m, 0.0), grad(m, 0.0);
        for(int i = 1; i < n; ++i){
            double g = -ridge * s[i], h = ridge;
            for(int j = 0; j < n; ++j){
                const double nij = played[i * n + j];
                if(nij == 0.0) continue;
                const double p = 1.0 / (1.0 + std::exp(s[j] - s[i]));
                g += wins[i * n + j] - nij * p;
                const double w = nij * p * (1.0 - p);
                h += w;
                if(j > 0) hess[(i - 1) * m + (j - 1)] -= w;
            }
            hess[(i - 1) * m + (i - 1)] += h;
            grad[i - 1] = g;
        }
        const std::vector<double> step = solve_linear(hess, grad);
        double maxStep = 0.0;
        for(int i = 1; i < n; ++i){
            s[i] += step[i - 1];
            maxStep = std::max(maxStep, std::abs(step[i - 1]));
        }
        if(maxStep < 1e-10)
            break;
    }

    const double elo = 400.0 / std::log(10.0);
    std::map<std::string, std::pair<float, int>> result;
    for(int i = 0; i < n; ++i)
        result[names[i]] = {static_cast<float>(initRating + s[i] * elo), games[i]};
    return result;
}
