#ifndef MODELCOMPARE_H
#define MODELCOMPARE_H

#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <optional>

#include "PMCTS.h"
#include "fairkomi.h"

class ModelCompare{
public:
    static float start_play(std::array<MCTS*, 2> playerList, const Game& start, // two models play against each other from start
		std::ostream& partRes, bool isShown = false, float temp = 0.1f);

	// play against human. Asks for the setup "<komi> <nNeutrals> [r c]..." first (empty line : config); on the human's turn,
	// "<r> <c>" plays a move and "komi <k>" changes komi. play_web and play_human read input the same way.
	// The entry points below take the whole Config of their mode (model path, the network / cache / search of the
	// engine, default board) and hand each object its part.
	static void play(const std::string& model, const Config& cfg, Color side, float temp, bool gpu, bool shown);

	static void play_human(const BoardConfig& board);

	// play against human via GTP. "komi <k>" works at any time, "set_neutrals [vertex]..." before the first move.
	static void play_gtp(const std::string& model, const Config& cfg, float temp, bool gpu);

	// Interactive analysis mode: reads "reset [<komi> <nNeutrals> [r c]...]" / "komi <k>" / "play <r> <c>" / "analyze [playouts]" / "quit"
	// commands from stdin, letting a caller build up an arbitrary position and request a
	// winrate / policy / visit-count breakdown for it without playing a full game.
	// search : the engine's search config (cfg.search with command-line overrides).
	static void analyze(const std::string& model, const Config& cfg, const SearchConfig& search, bool gpu);

	static void play_web(const std::string& model, const Config& cfg, const Color humanColor, int playout, float temp, bool gpu); // play against human/itself on web

	// return result of the match 1 : win for player_one, 0 : win for player two. Each pair of games (one per color
	// assignment) gets random neutral stones and a komi of that setup's fair komi (fairKomi) +- up to fairKomiNoise.
	static std::vector<bool> play_match(MCTS* playerOne, MCTS* playerTwo, const FairKomiTable& fairKomi, float fairKomiNoise,
		std::ostream& totalRes, bool isShown = false, float temp = 1.0f, int nGames = 100);

	// engines[i] : search config of the i-th model's engine. cfg : model path, network / cache config, komi.
	static float policy_evaluate(const std::string& modOne, const std::string& modTwo, const Config& cfg,
		const std::array<SearchConfig, 2>& engines, std::ostream& totalRes, std::ostream& partRes, bool isShown = false,
		bool gpu = true, float temp = 1.0f, int nGames = 96, int nThread = 8);

	// compares two already-built engines. returns playersOne's win rate.
	static float policy_evaluate(const std::vector<MCTS*>& playersOne, const std::vector<MCTS*>& playersTwo,
		const FairKomiTable& fairKomi, float fairKomiNoise, std::ostream& totalRes, bool isShown = false, float temp = 1.0f,
		int nGames = 96);

	// models are put in training order (natural order of names) and model i plays model i + d for every d in skips
	// (empty skips : full round-robin). Results are added to <eloFilePath>.pairs, ratings are fitted from every stored
	// pair (Bradley-Terry) and written to eloFilePath. Returns the ratings in training order.
	// engines[i] : search config of modelList[i]'s engine (one per model).
	static std::vector<float> policy_evaluate(std::vector<std::string> modelList, const Config& cfg,
		const std::vector<SearchConfig>& engines, std::ostream& totalRes, const std::optional<std::string>& eloFilePath,
		const std::vector<int>& skips = {}, bool isShown = false, bool gpu = true, float temp = 1.0f, int nGames = 100,
		int nThread = 8);

	
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

	static Color cmd_play(std::istringstream& iss, Game& gameManager, MCTS& player);

	static void cmd_genmove(std::istringstream& iss, Game& gameManager, MCTS& player, float temp);
};

#endif