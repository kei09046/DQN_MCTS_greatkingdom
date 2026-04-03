#ifndef MODELCOMPARE_H
#define MODELCOMPARE_H

#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <atomic>
#include "PMCTS.h"
#include "consts.h"
#include "elo.h"

class ModelCompare{
public:
    static float start_play(std::array<MCTS*, 2> player_list, // two models play against each other
		std::ostream& part_res, bool is_shown = false, float temp = 0.1f);

	static void play(const std::string& model, Color side, float temp, bool gpu, bool shown); // play against human

	static void playHuman();

	static void playGTP(const std::string& model, float temp, bool gpu); // play against human via GTP

	static void playWeb(const std::string& model, const Color humanColor, int playout, float temp, bool gpu); // play against human/itself on web

	static std::vector<bool> play_match(MCTS* player_one, MCTS* player_two, // return result of the match 1 : win for player_one, 0 : win for player two
		std::ostream& total_res, bool is_shown = false, float temp = 1.0f, int n_games = 100);

	static float policy_evaluate(const std::string& mod_one, const std::string& mod_two, 
		std::ostream& total_res, std::ostream& part_res, bool is_shown = false, bool gpu = true, float temp = 1.0f, int n_games = 96, int n_thread=16);

	static std::vector<float> policy_evaluate(std::vector<std::string> model_list, // return list of elo
		std::ostream& total_res, bool is_shown = false, bool gpu = true, float temp = 1.0f, int n_games = 100);

	static void displayBoardGUI(bool showScore, const Game& game);
	
private:
	inline static void ok(const std::string& s=""){
		std::cout << "= " << s << "\n\n";
        std::cout.flush();
	}

	inline static void err(const std::string& s) {
		std::cout << "? " << s << "\n\n";
		std::cout.flush();
	}

	static Move parse_vertex(const std::string& v);

	static Color cmd_play(std::istringstream& iss, Game game_manager, MCTS& player);

	static void cmd_genmove(std::istringstream& iss, Game game_manager, MCTS& player, float temp);
};

#endif