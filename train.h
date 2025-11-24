#ifndef TRAIN_H
#define TRAIN_H

#include "PMCTS.h"
#include "memory.h"
#include "policyvalue.h"
#include "random.h"
#include "rotation.h"
#include "elo.h"
#include "consts.h"
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
	std::mutex buffer_mutex;

	static std::vector<bool> play_match(MCTS* player_one, MCTS* player_two, // return result of the match 1 : win for player_one, 0 : win for player two
		std::ostream& total_res, bool is_shown = false, float temp = 1.0f, int n_games = 100);
	void synchronize_model(PolicyValueNet& target, PolicyValueNet& source, std::vector<MCTS>& players);
	void pin_threads_to_core(std::thread& th, int core_id);

public:
    std::deque<TrainData*>* game_buffer;
	std::array<float, inputDepth * batchSize * inputSize>* state_batch;
	std::array<float, batchSize * outputSize>* nextmove_batch;
	std::array<float, batchSize>* winner_batch;


	static float start_play(std::array<MCTS*, 2> player_list, // 서로 다른 모델들끼리 경기(학습확인용)
		std::ostream& part_res, bool is_shown = false, float temp = 0.1f);

	static void play(const std::string& model, color side, int playout, float temp, bool gpu, bool shown); // 사람과 경기

	static float policy_evaluate(const std::string& mod_one, const std::string& mod_two, 
		std::ostream& total_res, std::ostream& part_res, bool is_shown = false, bool gpu = true, float temp = 1.0f, int n_games = 100);

	static std::vector<float> policy_evaluate(std::vector<std::string> model_list, // return list of elo
		std::ostream& total_res, bool is_shown = false, bool gpu = true, float temp = 1.0f, int n_games = 100);


	TrainPipeline(std::string init_model, std::string test_model, bool gpu = false);

	void start_self_play(MCTS* player, bool is_shown = false, float temp = 0.1f, int n_games = 1); // 학습 중 경기(학습용)

	void insert_data(TrainData data);

	void train();

	void run(const int game_batch_num=10000, const int inference_thread_num=4, const bool is_shown=false, float temp=0.5f, const std::string& model_prefix="model");
};

#endif