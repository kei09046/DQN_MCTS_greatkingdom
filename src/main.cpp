#include "train.h"
#include "modelcompare.h"
#include "config.h"
#include <iostream>
#include <tuple>
#include <string>
#include <algorithm>

const Config globalConfig = loadConfig("../configs/train_local_config.json");


int main(int argc, char** argv) {
    std::cout << "starting ..." << std::endl;
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::string mod = argv[1];
    
    if(mod == "train"){
        std::string model_file = argv[2];
        int game_num = std::stoi(argv[3]);
        int num_thread = std::stoi(argv[4]);
        bool is_shown = static_cast<bool>(std::stoi(argv[5]));

        TrainPipeline line(model_file, model_file, true); // use gpu
        line.run(game_num, num_thread, is_shown, 0.5f, globalConfig.modelPrefix); // game_batch_num, train_thread_num, is_shown, temp, model_prefix
    }
    else if(mod == "human_play"){
        ModelCompare::playHuman();
    }
    else if(mod == "play"){
        std::string model_file = argv[2];
        int co = std::stoi(argv[3]); // human color
        int playout = std::stoi(argv[4]);
        ModelCompare::play(model_file, (color)co, playout, 10.0f, true, true);
    }
    else if(mod == "gtp"){
        std::string model_file = argv[2];
        int playout = std::stoi(argv[3]);
        ModelCompare::playGTP(model_file, playout, 10.0f, true);
    }
    else if(mod == "evaluate_two"){
        std::string target = argv[2];
        std::string compare = argv[3];
        float temp = std::stof(argv[4]); // < 1.0f
        int n_games = std::stoi(argv[5]);
        int n_threads = std::stoi(argv[6]);
        float winRate = ModelCompare::policy_evaluate(target, compare, std::cout, std::cout, false, true, temp, n_games, n_threads);
        std::cout << winRate << std::endl;
    }
    else if(mod == "evaluate_multi"){
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
