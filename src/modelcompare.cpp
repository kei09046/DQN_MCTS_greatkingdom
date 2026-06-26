#include <thread>
#include <atomic>
#include "modelcompare.h"
#include "consts.h"
#include "elo.h"


float ModelCompare::start_play(std::array<MCTS*, 2> player_list, std::ostream& part_res, bool is_shown, float temp) { // black wins : 1.0f, white wins : 0.0f
	if(is_shown)
		std::cout << "start play" << std::endl;
	//std::cerr << "start play" << std::endl;
	Game game_manager = Game();
	int diff, idx = 0;
	Move move;
    Color res;
	std::vector<Move> seq;

	while (true) {
		move = player_list[idx]->getMove(temp);
		if(is_shown){
			std::cout << static_cast<int>(move.first) << static_cast<int>(move.second) << " " << std::flush;
		}
		seq.push_back(move);
        res = game_manager.makeMove(move).first;

		if (res == EMPTY) {
            player_list[0]->jump(move);
            player_list[1]->jump(move);
			idx = 1 - idx;
			continue;
		}
		
        player_list[0]->reset();
        player_list[1]->reset();
		//std::cerr << std::endl;

		// if (is_shown) {
		// 	for (auto& moves : seq)
		// 		part_res << static_cast<int>(moves.first) << static_cast<int>(moves.second) << " ";
		// }

        if(is_shown)
            part_res << static_cast<int>(res) << std::endl;
        return res == BLACK ? 1.0f : 0.0f;
	}
}

void ModelCompare::play(const std::string& model, Color side, float temp, bool gpu, bool shown) {
	Game game_manager = Game();
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator);

	Move cord;
	Color res;

	while (true) {
		if (side == game_manager.getTurn()) {
			int r, c;
			std::cin >> r >> c;
			cord = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
		}
		else {
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			cord = player.getMove(temp);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "move time : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}

		if(cord == PASSMOVE || game_manager.isLegal(cord)){
			res = game_manager.makeMove(cord).first;
			displayBoardGUI(true, game_manager);
			if (res != EMPTY) {
				game_manager.onGameEnd(res);
				break;
			}
			player.jump(cord);
		}
	}

	delete evaluator;
	return;
}

void ModelCompare::playWeb(const std::string& model, const Color humanColor, int playout, float temp, bool gpu){
	Game game_manager = Game();
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator);

	Move cord;
	Color res;

	std::cout << "model loaded" << std::endl;
	while (true) {
		if (game_manager.getTurn() == humanColor) { // human turn
			int r, c;
			std::cin >> r >> c;
			cord = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
		}
		else { // engine turn
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			cord = player.getMove(temp);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "move: " << (int)cord.first << " " << (int)cord.second << std::endl;
			std::cout << "move time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}

		if(game_manager.isLegal(cord)){
			res = game_manager.makeMove(cord).first;
			if (res != EMPTY) {
				game_manager.onGameEnd(res);
				break;
			}
			player.jump(cord);
		}
	}

	delete evaluator;
	return;
}

void ModelCompare::playGTP(const std::string& model, float temp, bool gpu) {
    Game game_manager = Game();
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator);
	Move cord;
	Color res = EMPTY;

	std::atomic<bool> evaluating = false;
	std::mutex evalMutex;	
	std::condition_variable evalcv;

	std::thread eval_thread([&](){
		std::unique_lock<std::mutex> lk(evalMutex);
        evalcv.wait(lk, [&]{ return evaluating == true; });

		lk.unlock();
		while (evaluating) {
			player.runSimulation(PLAYOUT, 400, 0);
			float eval = player.getEval();
			ok(std::to_string(eval));
		}
		lk.lock();
	});

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
		std::cerr << cmd << "\n";

		if(cmd != "evaluate"){
			evaluating = false;
		}
        if (cmd == "protocol_version") ok("2");
        else if (cmd == "name") ok(globalConfig.modelPath + model);
        else if (cmd == "version") ok("0.1");
        else if (cmd == "list_commands")
        ok("protocol_version\nname\nversion\nboardsize\nclear_board\nplay\ngenmove\nquit");
        else if (cmd == "boardsize") ok(std::to_string(boardSize));
        else if (cmd == "clear_board") {
            game_manager = Game();
            player.reset();
            ok();
        }
		else if(cmd == "evaluate"){
			evaluating = true;
			evalcv.notify_one();
		}
        else if (cmd == "play"){ 
			res = cmd_play(iss, game_manager, player);
			if(res != EMPTY)
				std::cerr << "Game Over! Winner is : " << ((res == BLACK) ? "BLACK" : "WHITE") << std::endl;
		}
        else if (cmd == "genmove") {
			if(res == EMPTY)
				cmd_genmove(iss, game_manager, player, temp);
			else
				ok("resign");
		}
        else if (cmd == "quit") { ok(); break; }
        else ok();
    }

    delete evaluator;
    return;
}

void ModelCompare::playHuman() {
	Game game_manager = Game();
	//Game experiment = Game();

	Move cord;
	Color res;
	while (true) {
		int r, c;
		std::cin >> r >> c;
		cord = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};

		res = game_manager.makeMove(cord).first;
		displayBoardGUI(true, game_manager);
		std::cout << std::endl;

		// auto [win, moves, games, transtable] = experiment.expand();

		// int idx;
		// for(idx = 0; idx < moves.size(); ++idx)
		// 	if(moves[idx] == Move{r, c})
		// 		break;

		// displayBoardGUI(true, games[idx]);
		// std::cout << std::endl;
		// experiment = games[idx];

		
		if (res != EMPTY) {
			game_manager.onGameEnd(res);
			break;
		}
	}

	return;
}

std::vector<bool> ModelCompare::play_match(MCTS* player_one, MCTS* player_two,
		std::ostream& total_res, bool is_shown, float temp, int n_games) {

	std::vector<bool> result(n_games << 1);
	for (int i = 0; i < n_games; ++i) {
		//player_one plays as black
		result[i] = static_cast<bool>(ModelCompare::start_play({ player_one, player_two }, total_res, is_shown, temp));
	}

	for (int i = n_games; i < (n_games << 1); ++i) {
		result[i] = !static_cast<bool>(ModelCompare::start_play({ player_two, player_one }, total_res, is_shown, temp));
	}
	return result;
}

float ModelCompare::policy_evaluate(const std::string& mod_one, const std::string& mod_two, std::ostream& total_res, std::ostream& part_res, bool is_shown,
	bool gpu, float temp, int n_games, int n_thread) {
	auto eo = new Evaluator(globalConfig.modelPath + mod_one, gpu);
	auto et = new Evaluator(globalConfig.modelPath + mod_two, gpu);
	std::vector<MCTS*> base_players, oppo_players;

	for(int i=0; i<n_thread; ++i){
		base_players.push_back(new MCTS(eo));
		oppo_players.push_back(new MCTS(et));
	}

	std::vector<std::thread> evaluate_threads;
	std::atomic<int> total_win = 0;
	for(int i=0; i<n_thread; ++i){
		evaluate_threads.emplace_back([&, i]{
			std::vector<bool> b = play_match(base_players[i], oppo_players[i], total_res, (i == 0) && is_shown, temp, n_games / n_thread);
			total_win += std::count(b.begin(), b.end(), true);
		}
		);
	}
	for(auto& th : evaluate_threads){
		th.join();
	}

	for(int i=0; i<n_thread; ++i){
		delete base_players[i];
		delete oppo_players[i];
	}
	delete eo;
	delete et;
	total_res << "win count : " << total_win << std::endl;
	return total_win / static_cast<float>(n_games << 1);
}

std::vector<float> ModelCompare::policy_evaluate(std::vector<std::string> model_list,
	std::ostream& total_res, bool is_shown, bool gpu, float temp, int n_games) {
	int N = model_list.size();
	std::vector<MCTS*> players(N);
	std::vector<Evaluator*> evaluators(N);

	for (int i = 0; i < N; ++i) {
		evaluators[i] = new Evaluator(globalConfig.modelPath + model_list[i], gpu);
		players[i] = new MCTS(evaluators[i]);
	}

	bool load_from_file = false;
	EloCalculator elo_calculator(globalConfig.modelPath + "ratings.txt", model_list, load_from_file);

	for(int i=1; i<N; ++i){
		for(int j=0; j<N-i; ++j){
			std::vector<bool> b = play_match(players[j], players[j+i], total_res, is_shown, temp, n_games);
			elo_calculator.UpdateRatings(j, j+i, b);
			
			int win_cnt = std::count(b.begin(), b.end(), true);
			total_res << "Model " << model_list[j] << " VS Model " << model_list[j+i] << " : " 
				<< win_cnt << "/" << (n_games << 1) << " (" << (win_cnt * 100.0f / (n_games << 1)) << "%)" << std::endl;
		}
	}

	if(load_from_file){
		elo_calculator.saveRating(globalConfig.modelPath + "ratings.txt", model_list);
	}
	std::vector<float> ratings = elo_calculator.GetRatings(/*adjust=*/false);
	for(int i=0; i<N; ++i){
		total_res << model_list[i] << " Elo Rating : " << ratings[i] << std::endl;
	}

	for(int i=0; i<N; ++i){
		delete evaluators[i];
		delete players[i];
	}
	return ratings;
}


void ModelCompare::displayBoardGUI(bool showScore, const Game& game){
    char display[rowSize][colSize];

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            switch(game.getBoard({i, j})){
                case BLACK:
                    display[i][j] = 'o';
                    break;
                case WHITE:
                    display[i][j] = 'x';
                    break;
                case NEUTRAL:
                    display[i][j] = '+';
                    break;
                default:
                    display[i][j] = '-';
                    break;
            }

            if(showScore){
                switch(game.getScoreBoard({i, j})){
                    case BSCORE:
                        display[i][j] = 'b';
                        break;
                    case WSCORE:
                        display[i][j] = 'w';
                        break;
                    default:
                        break;
                }
            }
        }
    }

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            std::cout << display[i][j] << " ";
        }
        std::cout << "\n";
    }

	// const auto nnInput = PolicyValueNet::getData(game);
	// int cnt = rowSize * colSize * 8;
	// for(int i=8; i<=17; ++i){
	// 	std::cout << "\n channel " << i << "\n";
		
	// 	for(int j=0; j<rowSize; ++j){
	// 		for(int k=0; k<colSize; ++k)
	// 			std::cout << nnInput[cnt++] << " ";
	// 		std::cout << "\n";
	// 	}
	// }
}


Color ModelCompare::cmd_play(std::istringstream& iss, Game game_manager, MCTS& player) {
    char c;
    std::string v;
    iss >> c >> v;

    Move m = parse_vertex(v);
    Color res = game_manager.makeMove(m).first;
	if(res == EMPTY){
    	player.jump(m);
    	ok();
	}
	return res;
}

void ModelCompare::cmd_genmove(std::istringstream& iss, Game game_manager, MCTS& player, float temp) {
    char c;
    iss >> c;

    auto begin = std::chrono::steady_clock::now();
    Move m = player.getMove(temp);
    auto end = std::chrono::steady_clock::now();

    std::cerr << "move time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()
                << " us\n";

	if(m != RESIGNMOVE){
		Color res = game_manager.makeMove(m).first;
		player.jump(m);
	}

	if(m == PASSMOVE){
		ok("pass");
	}
	else if(m == RESIGNMOVE){
		ok("resign");
	}
	else{
		// convert to GTP coord
		char col = static_cast<int>(m.second) + 'A';
		if(col >= 'I') col++; // there is no 'I' in sabaki cord.
		std::string v;
		v += col;
		v += std::to_string(m.first + 1);
		std::cerr << "move made : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << " " << v << std::endl;
		ok(v);
	}
}

Move ModelCompare::parse_vertex(const std::string& v) {
    // e.g. "D4"
	if(v == "pass"){
		return PASSMOVE;
	}
	if(v == "resign"){
		return RESIGNMOVE;
	}
    char col = v[0];
	if(col >= 'J') col--; // there is no 'I' in sabaki cord.
    int row = std::stoi(v.substr(1));

    return {static_cast<uint8_t>(row - 1), static_cast<uint8_t>(col - 'A')};
}