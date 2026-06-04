#include "train.h"
#include "modelcompare.h"
#include "config.h"
#include <iostream>
#include <tuple>
#include <string>
#include <algorithm>

Config globalConfig;


int main(int argc, char** argv) {
    std::cout << "starting ..." << std::endl;
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::string mod = argv[1];
    
    if(mod == "train"){
        globalConfig = loadConfig("../configs/train_config.json");
        std::string model_file = argv[2];
        int game_num = std::stoi(argv[3]);
        int num_thread = std::stoi(argv[4]);
        bool is_shown = static_cast<bool>(std::stoi(argv[5]));

        TrainPipeline line(model_file, model_file, true); // use gpu
        line.run(game_num, num_thread, is_shown, globalConfig.temp, globalConfig.modelPrefix); // game_batch_num, train_thread_num, is_shown, 1/temp, model_prefix
    }
    else if(mod == "human_play"){
        globalConfig = loadConfig("../configs/play_config.json");
        ModelCompare::playHuman();
    }
    else if(mod == "play"){
        globalConfig = loadConfig("../configs/play_config.json");
        std::string model_file = argv[2];
        int co = std::stoi(argv[3]); // human Color
        ModelCompare::play(model_file, (Color)co, 10.0f, true, true);
    }
    else if(mod == "web"){
        globalConfig = loadConfig("../configs/play_config.json");
        std::string model_file = argv[2];
        std::string human_color = argv[3];
        int co = (human_color == "black") ? BLACK : ((human_color == "white") ? WHITE : NEUTRAL); // human Color
        int playout = std::stoi(argv[4]);
        float temp = std::stof(argv[5]);
        std::cout << argv[3] << " human Color : " << co << std::endl;
        ModelCompare::playWeb(model_file, (Color)co, playout, temp, true);
    }
    else if(mod == "gtp"){
        std::cout << argv[2] << " " << argv[3] << std::endl;
        globalConfig = loadConfig(argv[2]);
        std::string model_file = argv[3];
        ModelCompare::playGTP(model_file, 10.0f, true);
    }
    else if(mod == "evaluate_two"){
        globalConfig = loadConfig("../configs/compare_config.json");
        std::string target = argv[2];
        std::string compare = argv[3];
        float temp = std::stof(argv[4]); // < 1.0f
        int n_games = std::stoi(argv[5]);
        int n_threads = std::stoi(argv[6]);
        float winRate = ModelCompare::policy_evaluate(target, compare, std::cout, std::cout, false, true, temp, n_games, n_threads);
        std::cout << winRate << std::endl;
    }
    else if(mod == "evaluate_multi"){
        globalConfig = loadConfig("../configs/compare_config.json");
        int n_models = argc - 4;
        std::vector<std::string> model_list(n_models);
        for(int i=0; i<n_models; ++i)
            model_list[i] = argv[2 + i];

        int n_games = std::stoi(argv[2 + n_models]);
        float temp = std::stof(argv[3 + n_models]); // < 1.0f
        ModelCompare::policy_evaluate(model_list, std::cout, false, true, temp, n_games);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
}
