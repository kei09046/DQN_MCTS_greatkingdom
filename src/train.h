#ifndef TRAIN_H
#define TRAIN_H

#include "PMCTS.h"
#include "memory.h"
#include "neuralNet.h"
#include "random.h"
#include "replaybuffer.h"
#include "consts.h"
#include "modelcompare.h"
#include <string>
#include <utility>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <mutex>
#include <regex>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <optional>
#include <tuple>


// running comparison of the search's root value q with the game result z over value-trained positions.
// A negative correlation means q is read from the wrong perspective.
class ValueTargetStats {
	std::mutex mtx;
	double n = 0, sq = 0, sz = 0, sqq = 0, szz = 0, sqz = 0, sabs = 0;
public:
	void add(float q, float z){
		std::lock_guard<std::mutex> lock(mtx);
		n += 1; sq += q; sz += z; sqq += q * q; szz += z * z; sqz += q * z; sabs += std::abs(q - z);
	}
	// (count, Pearson correlation, mean |q - z|) since the last call; resets the counters.
	std::tuple<long, double, double> takeSummary(){
		std::lock_guard<std::mutex> lock(mtx);
		const double cov = sqz / n - (sq / n) * (sz / n);
		const double var = (sqq / n - (sq / n) * (sq / n)) * (szz / n - (sz / n) * (sz / n));
		const std::tuple<long, double, double> out{static_cast<long>(n), (n > 1 && var > 0) ? cov / std::sqrt(var) : 0.0, n > 0 ? sabs / n : 0.0};
		n = sq = sz = sqq = szz = sqz = sabs = 0;
		return out;
	}
};

// Fills in the training targets of a finished self-play game, in place. buffer[i] holds position i's input and policy
// target, sequence[i] the move played from it, rootValue[i] the search's value there. map comes from the move that ended
// the game (Game::makeMoveWithStat) : the captured stones for a capture, the occupancy map for a score ending.
// blackLead : black's territory lead at the end, komi not applied (used for score endings). stats may be null.
void labelFinishedGame(std::vector<TrainData>& buffer, const std::vector<Move>& sequence,
	const std::vector<std::optional<float>>& rootValue, Color winner, Wintype wintype,
	const std::vector<float>& map, float blackLead, float valueQMix, ValueTargetStats* stats = nullptr);

class TrainPipeline {
private:
	float learning_rate;
	int comparisons_without_improvement = 0;
	float captureRatio;
	int save_cnt; // indicate how many games have been played; used for model naming
	ValueTargetStats valueTargetStats;
	int skipped_steps = 0; // training steps skipped for a non-finite loss since the last save
	int games_played = 0; // used to check how many games have been played by inference model. Used for multiple inference threads case. 
	PolicyValueNet inference_model, train_model;
	bool gpu;
	std::mutex buffer_mutex;
	std::atomic<int> wintype_counter[4];
	std::atomic<int> total_score_diff;
	std::atomic<int> total_game_length;
	std::vector<float> train_losses[9];
	
	std::string current_best_model_file;

	void pin_threads_to_core(std::thread& th, int core_id);


	void displayTrainData(const std::shared_ptr<const TrainData> data) const;

public:
	ReplayBuffer replayBuffer;
	std::vector<float>* state_batch, *nextmove_batch, *result_batch, *outcome_batch, *score_batch, *map_batch;
	std::vector<Trainhead>* type_batch;

	TrainPipeline(std::string init_model, std::string test_model, bool gpu = false);

	void start_self_play(MCTS* player, bool is_shown = false, float temp = 0.1f, int n_games = 1); // used during training

	void insertData(TrainData& data, const int& forced = 0, const bool& only = false);

	// fills the replay buffer from a snapshot written during an earlier run (data/<model>.bin), before run().
	void loadReplayBuffer(const std::string& data_file);

	void train();

	void run(const int game_batch_num=10000, const int inference_thread_num=4, const bool is_shown=false, float temp=0.5f);
};

// Trains an untrained network on positions saved by ReplayBuffer::save, reporting training and validation
// losses as it goes. The newest validation_fraction of the positions are held out. Saves <prefix>static.pt.
void trainOnStaticData(const std::string& data_file, int steps, float lr, float validation_fraction = 0.1f);

#endif