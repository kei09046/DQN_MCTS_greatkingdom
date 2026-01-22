#include "config.h"
#include <fstream>
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
    c.measureTime   = F.value("measureTime", false);
    c.transTable    = F.value("transTable", false);
    c.dirichletNoise= F.value("dirichletNoise", false);
    c.apvMCTS       = F.value("apvMCTS", false);
    c.googleDrive   = F.value("googleDrive", false);

    // paths
    auto& P = j.at("path");
    c.modelPath   = P.at("model_path");
    c.modelPrefix = P.at("model_prefix");
    c.drivePath   = P.at("drive_path");

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
    c.nPlayout = M.at("n_playout");
    c.cPuct     = M.at("cPuct");

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

    //dirichlet noise
    auto& D = j.at("dirichlet_noise");
    c.alpha = D.at("alpha");
    c.eps = D.at("epsilon");

    return c;
}