#ifndef TRAIN_H
#define TRAIN_H

#include "PMCTS.h"
#include "memory.h"
#include "neuralNet.h"
#include "random.h"
#include "rotation.h"
#include "consts.h"
#include "modelcompare.h"
#include <string>
#include <deque>
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


class TrainPipeline {
private:
	float learning_rate = 0.001f;
	unsigned int save_cnt; // indicate how many games have been played; used for model naming
	unsigned int games_played = 0; // used to check how many games have been played by inference model. Used for multiple inference threads case. 
	PolicyValueNet prev_policy; // used for comparison
	PolicyValueNet inference_model, train_model;
	bool gpu;
	std::mutex buffer_mutex;
	std::atomic<int> wintype_counter[4];
	std::atomic<int> total_score_diff;
	std::atomic<int> total_game_length;
	
	std::string current_best_model_file;

	void pin_threads_to_core(std::thread& th, int core_id);

public:
    std::deque<TrainData*>* game_buffer;
	// std::array<float, inputChannel * batchSize * inputSize>* state_batch;
	// std::array<float, batchSize * outputSize>* nextmove_batch;
	// std::array<float, batchSize>* result_batch;
	std::vector<float>* state_batch, *nextmove_batch, *result_batch, *score_batch;

	TrainPipeline(std::string init_model, std::string test_model, bool gpu = false);

	void start_self_play(MCTS* player, bool is_shown = false, float temp = 0.1f, int n_games = 1); // used during training

	void insert_data(TrainData data);

	void train();

	void run(const int game_batch_num=10000, const int inference_thread_num=4, const bool is_shown=false, float temp=0.5f, const std::string& model_prefix="model");
};

#endif