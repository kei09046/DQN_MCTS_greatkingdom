#include "train.h"
#include "modelcompare.h"
#include "analysis.hpp"
#include "config.h"
#include "hash.h"
#include <iostream>
#include <tuple>
#include <string>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <fstream>

// a file name that doesn't exist as given is looked up in dataPath.
static std::string resolve_data_file(const std::string& name, const std::string& dataPath){
    return std::filesystem::exists(name) ? name : dataPath + name;
}


int main(int argc, char** argv) {
    std::cout << "starting ..." << std::endl;
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::string mod = argv[1];
    
    if(mod == "train"){
        Config cfg = load_config("../configs/train_config.json");
        // ./play train <model> <data file> <games> <threads> <is_shown> [learning rate] [-Nprune] [-Ntransfer] [-NcaptureMap]
        //             [-NscoringChannels]
        // if learning rate is not given, config file's default learning rate or optimizer's learning rate would be used.
        // ablation flags (any position after the mode) switch off one component of the method :
        //   -Nprune      search every legal move instead of only maximal scoring moves (implies -Ntransfer)
        //   -Ntransfer   keep each move's own prior logit instead of the max over the squares it claims
        //   -NcaptureMap train without the capture map loss
        //   -NscoringChannels  input channels 23 / 24 (my / opponent's territory building moves) are left all 0, in
        //                      self-play and in the checkpoint comparisons. Evaluate such a model with
        //                      "scoringChannels": false in "engines".
        std::vector<std::string> args;
        for(int i=2; i<argc; ++i){
            const std::string a = argv[i];
            if(a == "-Nprune")
                cfg.search.rules.prune = false;
            else if(a == "-Ntransfer")
                cfg.search.rules.transfer = false;
            else if(a == "-NcaptureMap")
                cfg.train.step.captureMapLoss = false;
            else if(a == "-NscoringChannels")
                cfg.search.rules.scoringChannels = false;
            else if(!a.empty() && a[0] == '-')
                throw std::runtime_error("unknown train option " + a + " (expected -Nprune, -Ntransfer, -NcaptureMap or -NscoringChannels)");
            else
                args.push_back(a);
        }
        if(args.size() < 5 || args.size() > 6){
            std::cerr << "usage : ./play train <model> <data file> <games> <threads> <is_shown> [learning rate] "
                         "[-Nprune] [-Ntransfer] [-NcaptureMap] [-NscoringChannels]" << std::endl;
            return 1;
        }
        RuleConfig& rules = cfg.search.rules;
        if(!rules.prune && rules.transfer){
            std::cout << "-Nprune : logit transfer is defined over the pruned move list only, so it is off too." << std::endl;
            rules.transfer = false;
        }
        std::cout << "ablation : prune " << rules.prune << " transfer " << rules.transfer
                  << " captureMap " << cfg.train.step.captureMapLoss << " scoringChannels " << rules.scoringChannels << std::endl;

        std::string modelFile = args[0];
        std::string dataFile = args[1];
        int gameNum = std::stoi(args[2]);
        int numThread = std::stoi(args[3]);
        bool isShown = static_cast<bool>(std::stoi(args[4]));

        std::optional<float> lr;
        if(args.size() > 5){
            lr = std::stof(args[5]);
            if(!(*lr > 0.0f))
                throw std::runtime_error("learning rate must be positive");
        }
        TrainPipeline line(modelFile, modelFile, cfg, "../configs/on_train_compare_config.json", true, lr); // use gpu
        if(dataFile != "none")
            line.load_replay_buffer(resolve_data_file(dataFile, cfg.paths.dataPath));
        line.run(gameNum, numThread, isShown, cfg.search.temp); // game_batch_num, train_thread_num, is_shown, 1/temp
    }
    else if(mod == "human_play"){
        const Config cfg = load_config("../configs/play_config.json");
        ModelCompare::play_human(cfg.board);
    }
    else if(mod == "play"){
        const Config cfg = load_config("../configs/play_config.json");
        std::string modelFile = argv[2];
        int co = std::stoi(argv[3]); // human Color
        ModelCompare::play(modelFile, cfg, (Color)co, 10.0f, true, true);
    }
    else if(mod == "analyze"){
        const Config cfg = load_config("../configs/play_config.json");
        std::string modelFile = argv[2];
        // optional overrides : ./play analyze <model> [FPU] [minVisitRatio]
        SearchConfig search = cfg.search;
        if(argc > 3)
            search.fpu = std::stof(argv[3]);
        if(argc > 4)
            search.minVisitRatio = std::stof(argv[4]);
        ModelCompare::analyze(modelFile, cfg, search, true);
    }
    else if(mod == "nninput"){
        // ./play nninput -- no model; see Analysis::nn_input_server.
        const Config cfg = load_config("../configs/play_config.json");
        Analysis::nn_input_server(cfg.board);
    }
    else if(mod == "rules"){
        // ./play rules -- no model; see Analysis::rules_server.
        const Config cfg = load_config("../configs/play_config.json");
        Analysis::rules_server(cfg.board);
    }
    else if(mod == "web"){
        const Config cfg = load_config("../configs/play_config.json");
        std::string modelFile = argv[2];
        std::string humanColor = argv[3];
        int co = (humanColor == "black") ? BLACK : ((humanColor == "white") ? WHITE : NEUTRAL); // human Color
        int playout = std::stoi(argv[4]);
        float temp = std::stof(argv[5]);
        std::cout << argv[3] << " human Color : " << co << std::endl;
        ModelCompare::play_web(modelFile, cfg, (Color)co, playout, temp, true);
    }
    else if(mod == "gtp"){
        std::cout << argv[2] << " " << argv[3] << std::endl;
        const Config cfg = load_config(argv[2]);
        std::string modelFile = argv[3];
        ModelCompare::play_gtp(modelFile, cfg, 10.0f, true);
    }
    else if(mod == "evaluate_two"){
        const Config cfg = load_config("../configs/off_train_compare_config.json");
        std::string target = argv[2];
        std::string compare = argv[3];
        float temp = std::stof(argv[4]); // < 1.0f
        int nGames = std::stoi(argv[5]);
        int nThreads = std::stoi(argv[6]);
        float winRate = ModelCompare::policy_evaluate(target, compare, cfg, {cfg.engine(0), cfg.engine(1)},
            std::cout, std::cout, false, true, temp, nGames, nThreads);
        std::cout << winRate << std::endl;
    }
    else if(mod == "evaluate_multi"){
        const Config cfg = load_config("../configs/off_train_compare_config.json");
        // ./play evaluate_multi <model1> <model2> [model3 ...] <temp> <nGames> <nThreads> <skips> <ratingPath | none> <resultPath>
        // models are sorted into training order; model i plays model i + d for each d in skips ("1,2,4"), or every
        // other model ("all"). Results are added to <ratingPath>.pairs and ratings, fitted from every stored pair,
        // go to ratingPath. ratingPath "none" : nothing is loaded or saved.
        const char* usage = "usage : ./play evaluate_multi <model1> <model2> [model3 ...] <temp> <nGames> <nThreads> "
                            "<skips : e.g. 1,2,4 | all> <ratingPath | none> <resultPath>";
        int nModels = argc - 8;
        if(nModels < 2){
            std::cerr << usage << std::endl;
            return 1;
        }
        std::vector<std::string> modelList(nModels);
        for(int i=0; i<nModels; ++i)
            modelList[i] = argv[2 + i];

        float temp = std::stof(argv[2 + nModels]); // < 1.0f
        int nGames = std::stoi(argv[3 + nModels]);
        int nThreads = std::stoi(argv[4 + nModels]);
        const std::string skipArg = argv[5 + nModels];
        const std::string ratingArg = argv[6 + nModels];
        const std::optional<std::string> ratingPath = (ratingArg == "none") ? std::nullopt : std::optional<std::string>(ratingArg);
        const std::string resultPath = argv[7 + nModels];

        std::vector<int> skips; // empty : round-robin
        if(skipArg != "all"){
            std::stringstream ss(skipArg);
            std::string tok;
            while(std::getline(ss, tok, ','))
                skips.push_back(std::stoi(tok));
            if(skips.empty() || std::any_of(skips.begin(), skips.end(), [](int d){ return d < 1; })){
                std::cerr << "skips must be positive integers separated by ',' or \"all\"\n" << usage << std::endl;
                return 1;
            }
        }

        std::cout << "models : " << std::endl;
        for(const auto& modelName : modelList)
            std::cout << modelName << " ";
        std::cout << "\n";

        std::cout << "nGames(per color) " << nGames << " nThreads " << nThreads << " skips " << skipArg << std::endl;

        std::vector<SearchConfig> engines;
        for(int i=0; i<nModels; ++i)
            engines.push_back(cfg.engine(i));

        std::ofstream outFile(resultPath);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }
        ModelCompare::policy_evaluate(modelList, cfg, engines, outFile, ratingPath, skips, false, true, temp, nGames, nThreads);
        outFile.close();
    }
    else if(mod == "test"){
        // ./play test <data file> [steps] [learning rate] : train an untrained network on a saved replay buffer.
        // a bare file name is looked up in data_path.
        const Config cfg = load_config("../configs/train_config.json");
        std::string dataFile = resolve_data_file(argv[2], cfg.paths.dataPath);
        int steps = (argc > 3) ? std::stoi(argv[3]) : 20000;
        float lr = (argc > 4) ? std::stof(argv[4]) : 0.0002f;
        train_on_static_data(cfg, dataFile, steps, lr);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
}
