#include "modelcompare.h"

float ModelCompare::start_play(std::array<MCTS*, 2> player_list, std::ostream& part_res, bool is_shown, float temp) { // black wins : 1.0f, white wins : 0.0f
	Game game_manager = Game();
	int diff, idx = 0;
	Move move;
    color res;
	std::vector<Move> seq;

	while (true) {
		move = player_list[idx]->getMove(temp);
		seq.push_back(move);
        res = game_manager.makeMove(move);

		if (res == EMPTY) {
            player_list[0]->jump(move);
            player_list[1]->jump(move);
			idx = 1 - idx;
			continue;
		}
		
        player_list[0]->reset();
        player_list[1]->reset();
		

		if (is_shown) {
			for (auto& moves : seq)
				part_res << static_cast<int>(moves.first) << static_cast<int>(moves.second) << " ";
		}

        if(is_shown)
            part_res << static_cast<int>(res) << std::endl;
        return res == BLACK ? 1.0f : 0.0f;
	}
}

void ModelCompare::play(const std::string& model, color side, int playout, float temp, bool gpu, bool shown) {
	Game game_manager = Game();
	auto evaluator = new Evaluator(model_path + model, gpu);
	MCTS player = MCTS(playout, evaluator);

	Move cord;
	color res;

	while (true) {
		if (side == game_manager.getTurn()) {
			u_int r, c;
			std::cin >> r >> c;
			cord = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
		}
		else {
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			cord = player.getMove(temp);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "move time : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}

		res = game_manager.makeMove(cord);
		game_manager.displayBoardGUI();
		if (res != EMPTY) {
			game_manager.onGameEnd(res);
			break;
		}
        player.jump(cord);
	}

	delete evaluator;
	return;
}

void ModelCompare::playGTP(const std::string& model, int playout, float temp, bool gpu) {
    Game game_manager = Game();
	auto evaluator = new Evaluator(model_path + model, gpu);
	MCTS player = MCTS(playout, evaluator);

	Move cord;
	color res;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "protocol_version") ok("2");
        else if (cmd == "name") ok(model_path + model);
        else if (cmd == "version") ok("0.1");
        else if (cmd == "list_commands")
        ok("protocol_version\nname\nversion\nboardsize\nclear_board\nplay\ngenmove\nquit");
        else if (cmd == "boardsize") ok(std::to_string(boardSize));
        else if (cmd == "clear_board") {
            game_manager = Game();
            player.reset();
            ok();
        }
        else if (cmd == "play"){ 
			cmd_play(iss, game_manager, player);
		}
        else if (cmd == "genmove") {
			cmd_genmove(iss, game_manager, player, temp);
		}
        else if (cmd == "quit") { ok(); break; }
        else ok();
    }

    delete evaluator;
    return;
}

std::vector<bool> ModelCompare::play_match(MCTS* player_one, MCTS* player_two,
		std::ostream& total_res, bool is_shown, float temp, int n_games) {

	std::vector<bool> result(n_games << 1);
	for (int i = 0; i < n_games; ++i) {
		//player_one plays as black
		result[i] = static_cast<bool>(ModelCompare::start_play({ player_one, player_two }, total_res, is_shown, temp));
		// total_res << win_cnt << "/" << i + 1 << std::endl;
	}

	for (int i = n_games; i < (n_games << 1); ++i) {
		result[i] = !static_cast<bool>(ModelCompare::start_play({ player_two, player_one }, total_res, is_shown, temp));
		// total_res << win_cnt << "/" << i + 1 << std::std::endl;
	}
	return result;
}

float ModelCompare::policy_evaluate(const std::string& mod_one, const std::string& mod_two, std::ostream& total_res, std::ostream& part_res, bool is_shown,
	bool gpu, float temp, int n_games) {
	auto eo = new Evaluator(model_path + mod_one, gpu);
	auto et = new Evaluator(model_path + mod_two, gpu);
	MCTS* base_player = new MCTS(n_playout, eo);
	MCTS* oppo_player = new MCTS(n_playout, et);

	std::vector<bool> b = play_match(base_player, oppo_player, total_res, is_shown, temp, n_games);

	delete base_player;
	delete oppo_player;
	delete eo;
	delete et;
	total_res << "win count : " << std::count(b.begin(), b.end(), true) << std::endl;
	return std::count(b.begin(), b.end(), true) / static_cast<float>(n_games << 1);
}

std::vector<float> ModelCompare::policy_evaluate(std::vector<std::string> model_list,
	std::ostream& total_res, bool is_shown, bool gpu, float temp, int n_games) {
	int N = model_list.size();
	std::vector<MCTS*> players(N);
	std::vector<Evaluator*> evaluators(N);

	for (int i = 0; i < N; ++i) {
		evaluators[i] = new Evaluator(model_path + model_list[i], gpu);
		players[i] = new MCTS(n_playout, evaluators[i]);
	}

	bool load_from_file = false;
	EloCalculator elo_calculator(model_path + "ratings.txt", model_list, load_from_file);

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
		elo_calculator.saveRating(model_path + "ratings.txt", model_list);
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


Move ModelCompare::parse_vertex(const std::string& v) {
    // e.g. "D4"
	if(v == "pass"){
		return passMove;
	}
	if(v == "resign"){
		return resignMove;
	}
    char col = v[0];
    int row = std::stoi(v.substr(1));

    return {static_cast<uint8_t>(row - 1), static_cast<uint8_t>(col - 'A')};
}

void ModelCompare::cmd_play(std::istringstream& iss, Game game_manager, MCTS& player) {
    char c;
    std::string v;
    iss >> c >> v;

    Move m = parse_vertex(v);
    color res = game_manager.makeMove(m);
    player.jump(m);
    ok();
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

    color res = game_manager.makeMove(m);
    player.jump(m);

	if(m == passMove){
		ok("pass");
	}
	else if(m == resignMove){
		ok("resign");
	}
	else{
		// convert to GTP coord
		char col = m.second + 'A';
		std::string v;
		v += col;
		v += std::to_string(m.first + 1);
		ok(v);
	}
}