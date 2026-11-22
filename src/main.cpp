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

Config globalConfig;

// a file name that doesn't exist as given is looked up in data_path.
static std::string resolve_data_file(const std::string& name){
    return std::filesystem::exists(name) ? name : globalConfig.dataPath + name;
}


int main(int argc, char** argv) {
    std::cout << "starting ..." << std::endl;
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::string mod = argv[1];
    
    if(mod == "train"){
        globalConfig = load_config("../configs/train_config.json");
        // ./play train <model> <data file> <games> <threads> <is_shown> [learning rate]
        // if learning rate is not given, config file's default learning rate or optimizer's learning rate would be used.
        std::string modelFile = argv[2];
        std::string dataFile = argv[3];
        int gameNum = std::stoi(argv[4]);
        int numThread = std::stoi(argv[5]);
        bool isShown = static_cast<bool>(std::stoi(argv[6]));

        std::optional<float> lr;
        if(argc > 7){
            lr = std::stof(argv[7]);
            if(!(*lr > 0.0f))
                throw std::runtime_error("learning rate must be positive");
        }
        TrainPipeline line(modelFile, modelFile, true, lr); // use gpu
        if(dataFile != "none")
            line.load_replay_buffer(resolve_data_file(dataFile));
        line.run(gameNum, numThread, isShown, globalConfig.temp); // game_batch_num, train_thread_num, is_shown, 1/temp
    }
    else if(mod == "human_play"){
        globalConfig = load_config("../configs/play_config.json");
        ModelCompare::play_human();
    }
    else if(mod == "play"){
        globalConfig = load_config("../configs/play_config.json");
        std::string modelFile = argv[2];
        int co = std::stoi(argv[3]); // human Color
        ModelCompare::play(modelFile, (Color)co, 10.0f, true, true);
    }
    else if(mod == "analyze"){
        globalConfig = load_config("../configs/play_config.json");
        std::string modelFile = argv[2];
        // optional overrides : ./play analyze <model> [FPU] [minVisitRatio]
        SearchParams params = globalConfig.search_params();
        if(argc > 3)
            params.fpu = std::stof(argv[3]);
        if(argc > 4)
            params.minVisitRatio = std::stof(argv[4]);
        ModelCompare::analyze(modelFile, true, params);
    }
    else if(mod == "nninput"){
        // ./play nninput -- no model; see Analysis::nn_input_server.
        globalConfig = load_config("../configs/play_config.json");
        Analysis::nn_input_server();
    }
    else if(mod == "rules"){
        // ./play rules -- no model; see Analysis::rules_server.
        globalConfig = load_config("../configs/play_config.json");
        Analysis::rules_server();
    }
    else if(mod == "web"){
        globalConfig = load_config("../configs/play_config.json");
        std::string modelFile = argv[2];
        std::string humanColor = argv[3];
        int co = (humanColor == "black") ? BLACK : ((humanColor == "white") ? WHITE : NEUTRAL); // human Color
        int playout = std::stoi(argv[4]);
        float temp = std::stof(argv[5]);
        std::cout << argv[3] << " human Color : " << co << std::endl;
        ModelCompare::play_web(modelFile, (Color)co, playout, temp, true);
    }
    else if(mod == "gtp"){
        std::cout << argv[2] << " " << argv[3] << std::endl;
        globalConfig = load_config(argv[2]);
        std::string modelFile = argv[3];
        ModelCompare::play_gtp(modelFile, 10.0f, true);
    }
    else if(mod == "evaluate_two"){
        globalConfig = load_config("../configs/off_train_compare_config.json");
        std::string target = argv[2];
        std::string compare = argv[3];
        float temp = std::stof(argv[4]); // < 1.0f
        int nGames = std::stoi(argv[5]);
        int nThreads = std::stoi(argv[6]);
        float winRate = ModelCompare::policy_evaluate(target, compare, std::cout, std::cout, true, true, temp, nGames, nThreads,
            {globalConfig.engine_params(0), globalConfig.engine_params(1)});
        std::cout << winRate << std::endl;
    }
    else if(mod == "evaluate_multi"){
        globalConfig = load_config("../configs/off_train_compare_config.json");
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

        std::vector<SearchParams> params;
        for(int i=0; i<nModels; ++i)
            params.push_back(globalConfig.engine_params(i));

        std::ofstream outFile(resultPath);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }
        ModelCompare::policy_evaluate(modelList, outFile, ratingPath, skips, false, true, temp, nGames, nThreads, params);
        outFile.close();
    }
    else if(mod == "test"){
        // ./play test <data file> [steps] [learning rate] : train an untrained network on a saved replay buffer.
        // a bare file name is looked up in data_path.
        globalConfig = load_config("../configs/train_config.json");
        std::string dataFile = resolve_data_file(argv[2]);
        int steps = (argc > 3) ? std::stoi(argv[3]) : 20000;
        float lr = (argc > 4) ? std::stof(argv[4]) : 0.0002f;
        train_on_static_data(dataFile, steps, lr);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
}
