#include "config.h"
#include <fstream>
#include <stdexcept>
#include "nlohmann/json.hpp"


Config load_config(const std::string& path) {
    using json = nlohmann::json;

    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open config file " + path);

    json j;
    f >> j;

    Config c{};

    auto& F = j.at("features");
    auto& P = j.at("path");
    auto& R = j.at("rating");
    auto& B = j.at("board");
    auto& M = j.at("mcts");
    auto& N = j.at("nn");
    auto& C = j.at("cache");
    auto& T = j.at("train");
    auto& D = j.at("dirichlet_noise");

    // paths
    c.paths.modelPath   = P.at("model_path");
    c.paths.drivePath   = P.at("drive_path");
    c.paths.dataPath    = P.value("data_path", std::string("../data/"));
    c.paths.googleDrive = F.value("googleDrive", false);

    // rating
    c.rating.initK      = R.at("init_K");
    c.rating.initRating = R.at("init_rating");

    // board
    c.board.komi = B.at("komi");
    for (auto& n : B.at("neutrals"))
        c.board.neutrals.emplace_back(n[0], n[1]);

    // network of a model file without type / size in its name
    c.net.modelPrefix    = P.at("model_prefix");
    c.net.numHiddenBlock = N.at("numHiddenBlock");
    if(c.net.numHiddenBlock < 1)
        throw std::runtime_error("numHiddenBlock must be at least 1");

    // search
    SearchConfig& s = c.search;
    s.mode = M.at("mode");
    s.time = M.at("time");
    s.nPlayout = M.at("nPlayout");
    s.fastPlayout = M.at("fastPlayout");
    if(s.fastPlayout < 1)
        throw std::runtime_error("fastPlayout must be at least 1");
    s.cPuct = M.at("cPuct");
    s.temp = M.at("temp");
    s.fpu = F.value("FPU", -1.0f);
    s.minVisitRatio = M.value("minVisitRatio", 0.005f);
    s.transTable = F.value("transTable", false);
    s.dirichletNoise = F.value("dirichletNoise", false);
    s.alpha = D.at("alpha");
    s.eps = D.at("epsilon");
    s.searchThreadNum = T.at("search_thread_num");

    // per-engine overrides (optional). Missing fields keep the values above.
    if (j.contains("engines")) {
        for (auto& E : j.at("engines")) {
            SearchConfig e = s;
            e.fpu = E.value("FPU", s.fpu);
            e.minVisitRatio = E.value("minVisitRatio", s.minVisitRatio);
            e.rules.prune = E.value("prune", s.rules.prune);
            e.rules.transfer = E.value("transfer", s.rules.transfer);
            c.engines.push_back(e);
        }
    }

    // cache (log2 sizes)
    c.cache.tableSize     = 1u << C.at("log_tableSize").get<unsigned>();
    c.cache.mutexPoolSize = 1u << C.at("log_mutexPoolSize").get<unsigned>();

    // train
    TrainConfig& t = c.train;
    t.batchSize = N.at("batchSize");
    t.step.epochs = T.at("epochs");
    t.checkFreq = T.at("check_freq");
    t.compareGameCnt = T.at("compare_game_cnt");
    t.compareThreadNum = T.at("compare_thread_num");
    t.trainWaitTime = T.at("train_wait_time");
    t.synchronizeFreq = T.at("synchronize_freq");
    t.capacity = T.at("capacity");
    t.trainStartPoint = T.at("trainStartPoint");
    if(t.trainStartPoint < t.batchSize)
        throw std::runtime_error("trainStartPoint must be at least batchSize");
    t.valueQMix = T.at("valueQMix");
    if(t.valueQMix < 0.0f || t.valueQMix > 1.0f)
        throw std::runtime_error("valueQMix must be in [0, 1]");
    t.windowFraction = T.at("windowFraction");
    if(t.windowFraction <= 0.0f || t.windowFraction > 1.0f)
        throw std::runtime_error("windowFraction must be in (0, 1]");
    t.fallbackWinrate = T.at("fallbackWinrate");
    if(t.fallbackWinrate < 0.0f || t.fallbackWinrate > 0.5f)
        throw std::runtime_error("fallbackWinrate must be in [0, 0.5]");
    t.lrPatience = T.at("lrPatience");
    if(t.lrPatience < 1)
        throw std::runtime_error("lrPatience must be at least 1");
    t.lrImprovementWinrate = T.at("lrImprovementWinrate");
    if(t.lrImprovementWinrate < 0.5f || t.lrImprovementWinrate > 1.0f)
        throw std::runtime_error("lrImprovementWinrate must be in [0.5, 1]");
    t.initialLearningRate = T.at("initialLearningRate");
    if(!(t.initialLearningRate > 0.0f))
        throw std::runtime_error("initialLearningRate must be positive");
    t.fullSearchProb = M.at("fullSearchProb");
    if(!(t.fullSearchProb > 0.0f && t.fullSearchProb <= 1.0f))
        throw std::runtime_error("fullSearchProb must be in (0, 1]");
    t.fairKomiNoise = T.at("fairKomiNoise");
    if(t.fairKomiNoise < 0.0f)
        throw std::runtime_error("fairKomiNoise must be non-negative");
    t.wideKomiProb = T.at("wideKomiProb");
    if(t.wideKomiProb < 0.0f || t.wideKomiProb > 1.0f)
        throw std::runtime_error("wideKomiProb must be in [0, 1]");
    t.wideKomiNoise = T.at("wideKomiNoise");
    if(t.wideKomiNoise < 0.0f)
        throw std::runtime_error("wideKomiNoise must be non-negative");

    return c;
}
