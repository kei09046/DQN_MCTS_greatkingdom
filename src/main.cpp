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
        // ./play train <model> <data file> <games> <threads> <is_shown>
        // model "none" starts from an untrained network. The optimizer state model_path/<model>.opt is loaded if it exists.
        // data file : replay buffer snapshot (e.g. J53760.bin) to start from; "none" starts with an empty buffer.
        std::string modelFile = argv[2];
        std::string dataFile = argv[3];
        int gameNum = std::stoi(argv[4]);
        int numThread = std::stoi(argv[5]);
        bool isShown = static_cast<bool>(std::stoi(argv[6]));

        TrainPipeline line(modelFile, modelFile, true); // use gpu
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
        globalConfig = load_config("../configs/compare_config.json");
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
        globalConfig = load_config("../configs/compare_config.json");
        int nModels = argc - 4;
        std::vector<std::string> modelList(nModels);
        for(int i=0; i<nModels; ++i)
            modelList[i] = argv[2 + i];

        int nGames = std::stoi(argv[2 + nModels]);
        float temp = std::stof(argv[3 + nModels]); // < 1.0f
        std::vector<SearchParams> params;
        for(int i=0; i<nModels; ++i)
            params.push_back(globalConfig.engine_params(i));
        ModelCompare::policy_evaluate(modelList, std::cout, false, true, temp, nGames, params);
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
    else if(mod == "test_hash"){
        Hash hash;
        HashValue h = hash.base_hash();
        Game g = Game();
        while(true){
            int r, c;
            std::cout << "Enter move (row col): ";
            std::cin >> r >> c;
            if(r == -1 && c == -1) break;
            if(!g.is_legal(r, c)){
                std::cout << "Illegal move. Try again." << std::endl;
                continue;
            }
            h = hash.compute_hash_after_move(g, {r, c}, h);
            g.make_move({r, c});
            std::cout << "New hash: " << h << std::endl;
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
}
