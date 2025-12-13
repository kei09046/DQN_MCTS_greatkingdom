#include "train.h"
#include <iostream>
#include <tuple>
#include <string>
#include <algorithm>

int main(){
    std::string mod;
    std::cin >> mod;

    if(mod == "train"){
        std::string model_file;
        int game_num, num_thread;
        bool is_shown;
        std::cin >> model_file >> game_num >> num_thread >> is_shown;
        TrainPipeline line(model_file, model_file, true); // use gpu
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        line.run(game_num, num_thread, is_shown, 0.5f, default_model_type); // game_batch_num, train_thread_num, is_shown, temp, model_prefix
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
    }
    else if(mod == "play"){
        std::string model_file;
        int co, playout;
        std::cin >> model_file >> co >> playout; // human color
        ModelCompare::play(model_file, (color)co, playout, 10.0f, true, true);
    }
    else if(mod == "evaluate_two"){
        std::string target, compare;
        int n_games;
        float temp;
        std::cin >> target >> compare;
        std::cin >> n_games;
        std::cin >> temp; // < 1.0f
        ModelCompare::policy_evaluate(target, compare, std::cout, std::cout, true, true, temp, n_games);
    }
    else if(mod == "evaluate_multi"){
        int n_models, n_games;
        float temp;
        std::cin >> n_models;
        std::vector<std::string> model_list(n_models);
        for(int i=0; i<n_models; ++i)
            std::cin >> model_list[i];
        std::cin >> n_games;
        std::cin >> temp; // < 1.0f
        ModelCompare::policy_evaluate(model_list, std::cout, false, true, temp, n_games);
    }
}
