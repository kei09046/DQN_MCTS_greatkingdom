#ifndef TRAIN_H
#define TRAIN_H

#include "PMCTS.h"
#include "memory.h"
#include "neuralNet.h"
#include "random.h"
#include "replaybuffer.h"
#include "consts.h"
#include "modelcompare.h"
#include "fairkomi.h"
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
	std::tuple<long, double, double> take_summary(){
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
// valueQMix : weight of q in position 0's value target, falling linearly to 0 at the terminal position.
void label_finished_game(std::vector<TrainData>& buffer, const std::vector<Move>& sequence,
	const std::vector<std::optional<float>>& rootValue, Color winner, Wintype wintype,
	const std::vector<float>& map, float blackLead, float valueQMix, ValueTargetStats* stats = nullptr);

class TrainPipeline {
private:
	float learningRate;
	int comparisonsWithoutImprovement = 0;
	float captureRatio;
	int saveCnt; // indicate how many games have been played; used for model naming
	ValueTargetStats valueTargetStats;
	int skippedSteps = 0; // training steps skipped for a non-finite loss since the last save
	int gamesPlayed = 0; // used to check how many games have been played by inference model. Used for multiple inference threads case. 
	PolicyValueNet inferenceModel, trainModel;
	bool gpu;
	std::mutex bufferMutex;
	std::atomic<int> wintypeCounter[4];
	std::atomic<int> totalScoreDiff;
	std::atomic<int> totalGameLength;
	std::vector<float> trainLosses[9];
	
	std::string currentBestModelFile;

	// fair komi of each neutral setup (rnd::neutral_setup_index), estimated by the self-play model : the komi at which
	// its value of the empty board is even. Self-play draws komi around it (rnd::sample_fair_komi).
	FairKomiTable fairKomi;
	std::mutex fairKomiMutex;

	// recomputes fairKomi with inferenceModel. Call only while no self-play game is being evaluated.
	void update_fair_komi();

	void pin_threads_to_core(std::thread& th, int coreId);


	void display_train_data(const std::shared_ptr<const TrainData> data) const;

public:
	ReplayBuffer replayBuffer;
	std::vector<float>* stateBatch, *nextmoveBatch, *resultBatch, *outcomeBatch, *scoreBatch, *mapBatch;
	std::vector<Trainhead>* typeBatch;

	TrainPipeline(std::string initModel, std::string testModel, bool gpu = false);

	void start_self_play(MCTS* player, bool isShown = false, float temp = 0.1f); // used during training

	void insert_data(TrainData& data, const int& forced = 0, const bool& only = false);

	// fills the replay buffer from a snapshot written during an earlier run (data/<model>.bin), before run().
	void load_replay_buffer(const std::string& dataFile);

	void train();

	void run(const int gameBatchNum=10000, const int inferenceThreadNum=4, const bool isShown=false, float temp=0.5f);
};

// Trains an untrained network on positions saved by ReplayBuffer::save, reporting training and validation
// losses as it goes. The newest validation_fraction of the positions are held out. Saves <prefix>static.pt.
void train_on_static_data(const std::string& dataFile, int steps, float lr, float validationFraction = 0.1f);

#endif