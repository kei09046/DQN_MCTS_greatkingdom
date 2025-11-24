#include "train.h"
#include <iostream>
#include <tuple>

// void startMCTS(color playerColor){
// 	Game* g = new Game();
// 	MCTS mctsEngine = MCTS();

// 	int r, c;
// 	color clr;

// 	while(true){
// 		if(g->getTurn() == playerColor){
// 			std::cout << "enter your move : " << std::endl;

//         	while(true){
// 				std::cin >> r >> c;
// 				if(!g->isLegal(r, c))
// 					std::cout << "illegal move! enter again" << std::endl;
// 				else
// 					break;
//         	}
// 		}

// 		else{
// 			mctsEngine.runSimulation(20000);
// 			std::tie(r, c) = mctsEngine.getMove();
// 			std::cout << "opponent's move : " << r << " " << c << std::endl;
// 		}

// 		clr = g->makeMove(r, c);
// 		mctsEngine.jump({r, c});
// 		g->displayBoardGUI();

// 		if(clr != EMPTY){
//             g->onGameEnd(clr);
//             return;
//         }
// 	}
// }

// void start(){
// 	Game* g = new Game();
//     int r, c;
//     color clr;

//     while(true){
//         std::cout << "enter your move : " << std::endl;

//         while(true){
//             std::cin >> r >> c;
//             if(!g->isLegal(r, c))
//                 std::cout << "illegal move! enter again" << std::endl;
//             else
//                 break;
//         }

//         clr = g->makeMove(r, c);
// 		g->displayBoardGUI();
//         if(clr != EMPTY){
//             g->onGameEnd(clr);
//             return;
//         }
//     }
// }

int main(){
	//startMCTS(WHITE);
	//start();
    std::string mod;
    std::cin >> mod;

    if(mod == "train"){
        std::string model_file;
        int game_num, num_thread;
        bool is_shown;
        std::cin >> model_file >> game_num >> num_thread >> is_shown;
        TrainPipeline line(model_file, model_file, true); // use gpu
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        line.run(game_num, num_thread, is_shown, 0.5f, "modele"); // game_batch_num, train_thread_num, is_shown, temp, model_prefix
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
    }
    else if(mod == "play"){
        std::string model_file;
        int co, playout;
        std::cin >> model_file >> co >> playout; // human color
        TrainPipeline::play(model_file, (color)co, playout, 10.0f, true, true);
    }
    else if(mod == "evaluate_two"){
        std::string target, compare;
        int n_games;
        float temp;
        std::cin >> target >> compare;
        std::cin >> n_games;
        std::cin >> temp; // < 1.0f
        TrainPipeline::policy_evaluate(target, compare, std::cout, std::cout, true, true, temp, n_games);
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
        TrainPipeline::policy_evaluate(model_list, std::cout, false, true, temp, n_games);
    }
}
