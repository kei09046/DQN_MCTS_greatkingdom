#include <thread>
#include <atomic>
#include <poll.h>
#include <cstdio>
#include "modelcompare.h"
#include "consts.h"
#include "elo.h"
#include "analysis.hpp"
#include "random.h"
#include <sstream>
#include <optional>
#include <cctype>

namespace{
	// search params of idx-th engine; global config values if not given.
	SearchParams params_of(const std::vector<SearchParams>& params, int idx){
		return (idx < static_cast<int>(params.size())) ? params[idx] : globalConfig.search_params();
	}

	bool blank(const std::string& line){
		return line.find_first_not_of(" \t\r") == std::string::npos;
	}

	// asks for the game setup on stdin until a valid one (or an empty line : config komi / neutrals) is given.
	Game prompt_setup(){
		std::string line;
		while(true){
			std::cout << "game setup : <komi> <nNeutrals> [r c]... (empty line : komi " << globalConfig.komi
				<< ", config neutrals)" << std::endl;
			if(!std::getline(std::cin, line) || blank(line))
				return Game();
			std::istringstream iss(line);
			float komi;
			std::vector<Move> neutrals;
			if(Game::read_setup(iss, komi, neutrals))
				return Game(komi, neutrals);
			std::cout << "invalid setup" << std::endl;
		}
	}

	void print_setup(std::ostream& out, const Game& game){
		out << "setup : komi " << game.get_komi() << " neutrals";
		bool any = false;
		for(int i=0; i<boardSize; ++i)
			if(game.get_board({i / colSize, i % colSize}) == NEUTRAL){
				out << " (" << i / colSize << "," << i % colSize << ")";
				any = true;
			}
		if(!any)
			out << " none";
		out << std::endl;
	}

	// "komi <k>" mid-game : changes komi and rebuilds the players' trees (their values were for the old komi).
	// Returns false (nothing changed) if k is missing or not a valid komi.
	bool change_komi(std::istringstream& iss, Game& game, std::initializer_list<MCTS*> players){
		float komi;
		if(!(iss >> komi) || !Game::valid_komi(komi))
			return false;
		game.set_komi(komi);
		for(MCTS* p : players)
			p->reset(game);
		return true;
	}

	// one line of human input : "<r> <c>" -> the move, or "komi <k>" -> handled here (nullopt).
	std::optional<Move> read_human_input(Game& game, std::initializer_list<MCTS*> players, bool& eof){
		std::string line;
		if(!std::getline(std::cin, line)){
			eof = true;
			return std::nullopt;
		}
		std::istringstream iss(line);
		std::string first;
		iss >> first;
		if(first == "komi"){
			if(change_komi(iss, game, players))
				std::cout << "komi : " << game.get_komi() << std::endl;
			else
				std::cout << "invalid komi (must be x.5)" << std::endl;
			return std::nullopt;
		}
		std::istringstream mv(line);
		int r, c;
		if(!(mv >> r >> c)){
			if(!blank(line))
				std::cout << "invalid input : \"<r> <c>\" or \"komi <k>\"" << std::endl;
			return std::nullopt;
		}
		return Move{static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
	}
}


float ModelCompare::start_play(std::array<MCTS*, 2> playerList, const Game& start, std::ostream& partRes, bool isShown, float temp) { // black wins : 1.0f, white wins : 0.0f
	Game gameManager = start;
	playerList[0]->reset(start);
	playerList[1]->reset(start);
	int idx = 0;
	Move move;
    Color res;
	//std::vector<Move> seq;

	while (true) {
		move = playerList[idx]->get_move(temp);
		if(isShown){
			std::cout << static_cast<int>(move.first) << static_cast<int>(move.second) << " ";
		}
		//seq.push_back(move);
        res = gameManager.make_move(move).first;

		if (res == EMPTY) {
            playerList[0]->jump(move);
            playerList[1]->jump(move);
			idx = 1 - idx;
			continue;
		}
		
		//std::cerr << std::endl;

		// if (is_shown) {
		// 	for (auto& moves : seq)
		// 		part_res << static_cast<int>(moves.first) << static_cast<int>(moves.second) << " ";
		// }

        if(isShown){
            partRes << static_cast<int>(res) << " ";
            print_setup(partRes, start);
        }
        return res == BLACK ? 1.0f : 0.0f;
	}
}

void ModelCompare::play(const std::string& model, Color side, float temp, bool gpu, bool shown) {
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator);
	Game gameManager = prompt_setup();
	player.reset(gameManager);
	print_setup(std::cout, gameManager);
	gameManager.display_board_gui(false);

	Move cord;
	Color res;

	while (true) {
		if (side == gameManager.get_turn()) {
			std::cout << "your move : <r> <c>, or komi <k>" << std::endl;
			bool eof = false;
			const auto in = read_human_input(gameManager, {&player}, eof);
			if(eof)
				break;
			if(!in)
				continue;
			cord = *in;
		}
		else {
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			cord = player.get_move(temp);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "move : " << static_cast<int>(cord.first) << " " << static_cast<int>(cord.second) << std::endl;
			std::cout << "move time : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}

		if(cord == PASSMOVE || gameManager.is_legal(cord)){
			res = gameManager.make_move(cord).first;
			//game_manager.displayBoardGUI(true);
			if (res != EMPTY) {
				gameManager.on_game_end(res);
				break;
			}
			player.jump(cord);
		}

		gameManager.display_board_gui(false);
	}

	delete evaluator;
	return;
}

void ModelCompare::analyze(const std::string& model, bool gpu, SearchParams params) {
	// The "analyze" loop below peeks the raw stdin fd via poll() to detect a queued
	// "pause" command without blocking. Buffered stdio can silently read ahead past the
	// currently-parsed line (e.g. if the first getline() below runs late because model
	// loading was slow, it can gulp up an already-written "pause" line together with the
	// "analyze" line into its internal buffer), which would hide those bytes from a raw-fd
	// poll() entirely. Unbuffering stdin makes every std::cin read match a real fd-level
	// read, so poll() always sees exactly what's actually still unconsumed.
	setvbuf(stdin, nullptr, _IONBF, 0);

	Game gameManager = Game();
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator, params);
	// Owns debug-mode state and the printAnalysis/printVariation implementations (see
	// analysis.hpp) -- attach it so player's own hot path (playout()/updateEval()) can reach it
	// for debug-line streaming too, per MCTS::attachAnalysis's own comment.
	Analysis analysis;
	player.attach_analysis(&analysis);

	std::cout << "search params : FPU " << params.fpu << " minVisitRatio " << params.minVisitRatio << std::endl;
	std::cout << "analysis ready" << std::endl;

	std::string line;
	while (std::getline(std::cin, line)) {
		std::istringstream iss(line);
		std::string cmd;
		iss >> cmd;

		if (cmd == "quit") {
			break;
		}
		else if (cmd == "reset") {
			// "reset" : empty board with config komi / neutrals. "reset <komi> <nNeutrals> [r c]..." : that setup.
			float komi;
			std::vector<Move> neutrals;
			std::string rest;
			std::getline(iss, rest);
			std::istringstream setup(rest);
			if(blank(rest))
				gameManager = Game();
			else if(Game::read_setup(setup, komi, neutrals))
				gameManager = Game(komi, neutrals);
			else{
				std::cout << "reset invalid" << std::endl;
				continue;
			}
			player.reset(gameManager);
			std::cout << "reset ok" << std::endl;
		}
		else if (cmd == "komi") {
			// "komi <k>" : changes komi of the current position (search tree is rebuilt).
			if(change_komi(iss, gameManager, {&player}))
				std::cout << "komi ok" << std::endl;
			else
				std::cout << "komi invalid" << std::endl;
		}
		else if (cmd == "play") {
			int r, c;
			iss >> r >> c;
			Move m = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};

			if (m == PASSMOVE || gameManager.is_legal(m)) {
				Color res = gameManager.make_move(m).first;
				player.jump(m);
				std::cout << "play ok" << std::endl;
				if (res != EMPTY) {
					std::cout << "game over : " << static_cast<int>(res) << std::endl;
				}
			}
			else {
				std::cout << "play illegal" << std::endl;
			}
		}
		else if (cmd == "analyze") {
			// target is a *cumulative* visit count on the root, not a one-shot playout budget:
			// re-requesting the same or a lower target is a no-op (just re-prints current state),
			// and raising it later resumes search on the existing tree instead of restarting.
			int targetVisits = globalConfig.nPlayout;
			int requested;
			if (iss >> requested)
				targetVisits = requested;

			// Stream a fresh snapshot roughly this often, for a live "mid-search" view -- also the
			// nPlayout runSimulation sees per call, so chunk * minVisitRatio has to be >= 1 or the
			// minSearchPerChild passed below truncates to 0 and the "every child gets at least
			// some playout" forced-first-child phase silently never runs when driven from here.
			constexpr int chunk = 400;

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			bool ranAny = false;
			while (player.root_visits() < targetVisits && player.root_forced_state() == 0) {
				int remaining = targetVisits - player.root_visits();
				int n = std::min(chunk, remaining);
				// minVisitRatio of this call's own playout budget, same ratio MCTS::getMove uses
				// -- e.g. n == chunk == 400 with minVisitRatio 0.005 gives 2 forced playouts
				// per child every chunk.
				player.run_simulation(PLAYOUT, n, 0, player.search_params().min_visits(n));
				analysis.print_analysis(player);
				ranAny = true;

				// Let a queued command (e.g. "pause") interrupt a long-running search: peek
				// stdin non-blockingly between chunks and stop early without consuming the
				// line, so the outer REPL loop picks it up as the next command.
				struct pollfd pfd = {0, POLLIN, 0};
				if (poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN))
					break;
			}
			if (!ranAny)
				analysis.print_analysis(player);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "analyze time : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}
		else if (cmd == "select") {
			// "select <temp>" -- pick a move from the current search result with the same rule
			// MCTS::getMove uses (Node::selectMove), without searching further. Does not play it;
			// the caller sends "play" afterwards. Pass is printed as "<rowSize> 0", resign as "255 255".
			float temp = globalConfig.temp;
			iss >> temp;
			Move m = player.select_move(temp);
			std::cout << "selected " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;
		}
		else if (cmd == "fairkomi") {
			analysis.print_fair_komi(player);
		}
		else if (cmd == "pause") {
			// The in-progress "analyze" loop (if any) already stopped itself to read this
			// line; nothing further to do here besides acknowledging it.
			std::cout << "paused" << std::endl;
		}
		else if (cmd == "debug") {
			// "debug on"/"debug off" -- toggles per-playout search-path + NN-eval logging (see
			// Analysis::setDebugMode/printPlayoutDebugLine in analysis.hpp/.cpp). While on,
			// each playout's line is printed the moment it finishes -- nothing is stored or
			// batched here; the GUI (Python) is what accumulates and manages the resulting
			// stream.
			std::string state;
			iss >> state;
			analysis.set_debug_mode(state == "on");
			std::cout << "debug " << (analysis.get_debug_mode() ? "on" : "off") << std::endl;
		}
		else {
			std::cout << "unknown command" << std::endl;
		}
	}

	delete evaluator;
	return;
}

void ModelCompare::play_web(const std::string& model, const Color humanColor, int playout, float temp, bool gpu){
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator);

	Move cord;
	Color res;

	std::cout << "model loaded" << std::endl;
	Game gameManager = prompt_setup();
	player.reset(gameManager);
	print_setup(std::cout, gameManager);
	while (true) {
		if (gameManager.get_turn() == humanColor) { // human turn : "<r> <c>", or "komi <k>"
			bool eof = false;
			const auto in = read_human_input(gameManager, {&player}, eof);
			if(eof)
				break;
			if(!in)
				continue;
			cord = *in;
		}
		else { // engine turn
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			cord = player.get_move(temp);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "move: " << (int)cord.first << " " << (int)cord.second << std::endl;
			std::cout << "move time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}

		if(gameManager.is_legal(cord)){
			res = gameManager.make_move(cord).first;
			if (res != EMPTY) {
				gameManager.on_game_end(res);
				break;
			}
			player.jump(cord);
		}
	}

	delete evaluator;
	return;
}

void ModelCompare::play_gtp(const std::string& model, float temp, bool gpu) {
    std::vector<Move> neutrals = globalConfig.neutrals;
    Game gameManager = Game();
	auto evaluator = new Evaluator(globalConfig.modelPath + model, gpu);
	MCTS player = MCTS(evaluator);
	Move cord;
	Color res = EMPTY;

	std::atomic<bool> evaluating = false;
	std::mutex evalMutex;	
	std::condition_variable evalcv;

	std::thread evalThread([&](){
		std::unique_lock<std::mutex> lk(evalMutex);
        evalcv.wait(lk, [&]{ return evaluating == true; });

		lk.unlock();
		while (evaluating) {
			player.run_simulation(PLAYOUT, 400, 0);
			float eval = player.get_eval().first;
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
        ok("protocol_version\nname\nversion\nboardsize\nclear_board\nkomi\nset_neutrals\nplay\ngenmove\nquit");
        else if (cmd == "boardsize") ok(std::to_string(boardSize));
        else if (cmd == "clear_board") {
            // keeps the current komi and neutral stones.
            gameManager = Game(gameManager.get_komi(), neutrals);
            player.reset(gameManager);
            res = EMPTY;
            ok();
        }
        else if (cmd == "komi") {
            // allowed at any time; mid-game it rebuilds the search tree.
            if(change_komi(iss, gameManager, {&player}))
                ok();
            else
                err("komi must be x.5");
        }
        else if (cmd == "set_neutrals") {
            // "set_neutrals [vertex]..." (none : no neutral stone). Only before the first move.
            if(gameManager.get_move_count() != 0){
                err("board not empty");
                continue;
            }
            std::vector<Move> ns;
            std::string v;
            bool valid = true;
            while(iss >> v){
                const Move m = parse_vertex(v);
                if(m.first >= rowSize || m.second >= colSize || std::find(ns.begin(), ns.end(), m) != ns.end()){
                    valid = false;
                    break;
                }
                ns.push_back(m);
            }
            if(!valid){
                err("invalid vertex");
                continue;
            }
            neutrals = ns;
            gameManager = Game(gameManager.get_komi(), neutrals);
            player.reset(gameManager);
            ok();
        }
		else if(cmd == "evaluate"){
			evaluating = true;
			evalcv.notify_one();
		}
        else if (cmd == "play"){ 
			res = cmd_play(iss, gameManager, player);
			if(res != EMPTY)
				std::cerr << "Game Over! Winner is : " << ((res == BLACK) ? "BLACK" : "WHITE") << std::endl;
		}
        else if (cmd == "genmove") {
			if(res == EMPTY)
				cmd_genmove(iss, gameManager, player, temp);
			else
				ok("resign");
		}
        else if (cmd == "quit") { ok(); break; }
        else ok();
    }

    delete evaluator;
    return;
}

void ModelCompare::play_human() {
	Game gameManager = prompt_setup();
	Game experiment = gameManager;
	print_setup(std::cout, gameManager);
	experiment.set_policy_mask();
	Move cord;
	Color res;
	while (true) {
		bool eof = false;
		const auto in = read_human_input(gameManager, {}, eof);
		if(eof)
			break;
		if(!in){
			experiment.set_komi(gameManager.get_komi());
			continue;
		}
		cord = *in;
		const int r = cord.first, c = cord.second;

		res = gameManager.make_move(cord).first;
		gameManager.display_board_gui(true);
		std::cout << std::endl;

		// auto [onlymove, forcedState] = experiment.tacticCheck();
		// if(onlymove != RESIGNMOVE){
		// 	std::cout << "threat detected! " << static_cast<int>(onlymove.first) 
		// 	<< " " << static_cast<int>(onlymove.second) << std::endl;
		// }
		const auto& moves = experiment.get_available_moves();
		// const auto& transTable = experiment.get_transfer_table();
		
		int idx;
		for(idx = 0; idx < moves.size(); ++idx)
			if(moves[idx] == r * colSize + c)
				break;

		std::cout << "move idx : " << idx << std::endl;
		// std::cout << "trans table : ";
		// for(auto m : transTable)
		// 	std::cout << m << " ";
		std::cout << std::endl;
		 
		experiment.make_move_given_score(cord);
		experiment.set_policy_mask();
		experiment.display_board_gui(true);
		std::cout << std::endl;

		if (res != EMPTY) {
			gameManager.on_game_end(res);
			break;
		}
	}

	return;
}

std::vector<bool> ModelCompare::play_match(MCTS* playerOne, MCTS* playerTwo, const FairKomiTable& fairKomi,
		std::ostream& totalRes, bool isShown, float temp, int nGames) {

	// each pair of games (one with each color) gets its own random setup, drawn like self-play's.
	std::vector<bool> result(nGames << 1);
	for (int i = 0; i < nGames; ++i) {
		const std::vector<Move> neutrals = rnd::sample_neutrals();
		const Game start(rnd::sample_fair_komi(fairKomi.at(rnd::neutral_setup_index(neutrals)), globalConfig.fairKomiNoise),
			neutrals);
		// result[i] : player_one plays as black, result[nGames + i] : player_two plays as black
		result[i] = static_cast<bool>(ModelCompare::start_play({ playerOne, playerTwo }, start, totalRes, isShown, temp));
		result[nGames + i] = !static_cast<bool>(ModelCompare::start_play({ playerTwo, playerOne }, start, totalRes, isShown, temp));
	}
	return result;
}

float ModelCompare::policy_evaluate(const std::string& modOne, const std::string& modTwo, std::ostream& totalRes, std::ostream& partRes, bool isShown,
	bool gpu, float temp, int nGames, int nThread, const std::vector<SearchParams>& params) {
	auto eo = new Evaluator(globalConfig.modelPath + modOne, gpu);
	auto et = new Evaluator(globalConfig.modelPath + modTwo, gpu);
	std::vector<MCTS*> basePlayers, oppoPlayers;

	for(int i=0; i<nThread; ++i){
		basePlayers.push_back(new MCTS(eo, params_of(params, 0)));
		oppoPlayers.push_back(new MCTS(et, params_of(params, 1)));
	}

	// fair komi : the mean of both models' estimates, so neither model alone decides how setups are balanced.
	const FairKomiTable fairKomi = combine_fair_komi({estimate_fair_komi(*eo->model()), estimate_fair_komi(*et->model())},
		globalConfig.komi);
	totalRes << "fair komi : ";
	print_fair_komi(totalRes, fairKomi);
	totalRes << std::endl;

	std::vector<std::thread> evaluateThreads;
	std::atomic<int> totalWin = 0;
	for(int i=0; i<nThread; ++i){
		evaluateThreads.emplace_back([&, i]{
			std::vector<bool> b = play_match(basePlayers[i], oppoPlayers[i], fairKomi, totalRes, (i == 0) && isShown, temp, nGames / nThread);
			totalWin += std::count(b.begin(), b.end(), true);
		}
		);
	}
	for(auto& th : evaluateThreads){
		th.join();
	}

	for(int i=0; i<nThread; ++i){
		delete basePlayers[i];
		delete oppoPlayers[i];
	}
	delete eo;
	delete et;
	totalRes << "win count : " << totalWin << std::endl;
	return totalWin / static_cast<float>(nGames << 1);
}

std::vector<float> ModelCompare::policy_evaluate(std::vector<std::string> modelList,
	std::ostream& totalRes, bool isShown, bool gpu, float temp, int nGames, const std::vector<SearchParams>& params) {
	int N = modelList.size();
	std::vector<MCTS*> players(N);
	std::vector<Evaluator*> evaluators(N);

	for (int i = 0; i < N; ++i) {
		evaluators[i] = new Evaluator(globalConfig.modelPath + modelList[i], gpu);
		players[i] = new MCTS(evaluators[i], params_of(params, i));
	}

	// one fair komi table for every pairing (the mean of all models' estimates), so all games share one setup distribution.
	std::vector<std::vector<std::optional<float>>> estimates;
	for(auto* e : evaluators)
		estimates.push_back(estimate_fair_komi(*e->model()));
	const FairKomiTable fairKomi = combine_fair_komi(estimates, globalConfig.komi);
	totalRes << "fair komi : ";
	print_fair_komi(totalRes, fairKomi);
	totalRes << std::endl;

	bool loadFromFile = false;
	EloCalculator eloCalculator(globalConfig.modelPath + "ratings.txt", modelList, loadFromFile);

	for(int i=1; i<N; ++i){
		for(int j=0; j<N-i; ++j){
			std::vector<bool> b = play_match(players[j], players[j+i], fairKomi, totalRes, isShown, temp, nGames);
			eloCalculator.update_ratings(j, j+i, b);
			
			int winCnt = std::count(b.begin(), b.end(), true);
			totalRes << "Model " << modelList[j] << " VS Model " << modelList[j+i] << " : " 
				<< winCnt << "/" << (nGames << 1) << " (" << (winCnt * 100.0f / (nGames << 1)) << "%)" << std::endl;
		}
	}

	if(loadFromFile){
		eloCalculator.save_rating(globalConfig.modelPath + "ratings.txt", modelList);
	}
	std::vector<float> ratings = eloCalculator.get_ratings(/*adjust=*/false);
	for(int i=0; i<N; ++i){
		totalRes << modelList[i] << " Elo Rating : " << ratings[i] << std::endl;
	}

	for(int i=0; i<N; ++i){
		delete evaluators[i];
		delete players[i];
	}
	return ratings;
}


Color ModelCompare::cmd_play(std::istringstream& iss, Game& gameManager, MCTS& player) {
    char c;
    std::string v;
    iss >> c >> v;

    Move m = parse_vertex(v);
    Color res = gameManager.make_move(m).first;
	if(res == EMPTY){
    	player.jump(m);
    	ok();
	}
	return res;
}

void ModelCompare::cmd_genmove(std::istringstream& iss, Game& gameManager, MCTS& player, float temp) {
    char c;
    iss >> c;

    auto begin = std::chrono::steady_clock::now();
    Move m = player.get_move(temp);
    auto end = std::chrono::steady_clock::now();

    std::cerr << "move time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()
                << " us\n";

	if(m != RESIGNMOVE){
		Color res = gameManager.make_move(m).first;
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
	if(v.size() < 2 || !std::isdigit(static_cast<unsigned char>(v[1])))
		return RESIGNMOVE; // not a vertex
	char col = std::toupper(static_cast<unsigned char>(v[0]));
	if(col >= 'J') col--; // there is no 'I' in sabaki cord.
    int row = std::stoi(v.substr(1));
	if(col < 'A' || row < 1)
		return RESIGNMOVE;

    return {static_cast<uint8_t>(row - 1), static_cast<uint8_t>(col - 'A')};
}