#include "config.h"
#include <fstream>
#include <stdexcept>
#include "nlohmann/json.hpp"


Config loadConfig(const std::string& path) {
    using json = nlohmann::json;

    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config file");

    json j; 
    f >> j;

    Config c{};

    // features
    auto& F = j.at("features");
    c.transTable    = F.value("transTable", false);
    c.dirichletNoise= F.value("dirichletNoise", false);
    c.googleDrive   = F.value("googleDrive", false);
    c.fpu = F.value("FPU", -1.0f);
    c.detailedStat = F.value("detailedStat", false);

    // paths
    auto& P = j.at("path");
    c.modelPath   = P.at("model_path");
    c.modelPrefix = P.at("model_prefix");
    c.drivePath   = P.at("drive_path");
    c.dataPath    = P.value("data_path", std::string("../data/"));

    // rating
    auto& R = j.at("rating");
    c.initK      = R.at("init_K");
    c.initRating = R.at("init_rating");

    // board
    auto& B = j.at("board");
    c.komi    = B.at("komi");

    for (auto& n : B.at("neutrals"))
        c.neutrals.emplace_back(n[0], n[1]);

    // mcts
    auto& M = j.at("mcts");
    c.mode = M.at("mode");
    c.time = M.at("time");
    c.nPlayout = M.at("nPlayout");
    c.cPuct     = M.at("cPuct");
    c.temp = M.at("temp");
    c.minVisitRatio = M.value("minVisitRatio", 0.005f);

    // per-engine overrides (optional). Missing fields fall back to the global values above.
    if (j.contains("engines")) {
        for (auto& E : j.at("engines"))
            c.engines.push_back({E.value("FPU", c.fpu), E.value("minVisitRatio", c.minVisitRatio)});
    }

    // nn
    auto& N = j.at("nn");
    c.batchSize    = N.at("batchSize");
    c.inputChannel = N.at("inputChannel");

    // cache (log2 sizes)
    auto& C = j.at("cache");
    c.tableSize      = 1u << C.at("log_tableSize").get<unsigned>();
    c.mutexPoolSize  = 1u << C.at("log_mutexPoolSize").get<unsigned>();

    // train
    auto& T = j.at("train");
    c.epochs             = T.at("epochs");
    c.check_freq         = T.at("check_freq");
    c.compare_game_cnt   = T.at("compare_game_cnt");
    c.compare_thread_num = T.at("compare_thread_num");
    c.search_thread_num  = T.at("search_thread_num");
    c.train_wait_time    = T.at("train_wait_time");
    c.save_freq          = T.at("save_freq");
    c.capacity           = T.at("capacity");
    c.trainStartPoint    = T.at("trainStartPoint");
    if(c.trainStartPoint < c.batchSize)
        throw std::runtime_error("trainStartPoint must be at least batchSize");
    c.windowFraction     = T.at("windowFraction");
    if(c.windowFraction <= 0.0f || c.windowFraction > 1.0f)
        throw std::runtime_error("windowFraction must be in (0, 1]");

    //dirichlet noise
    auto& D = j.at("dirichlet_noise");
    c.alpha = D.at("alpha");
    c.eps = D.at("epsilon");

    return c;
}