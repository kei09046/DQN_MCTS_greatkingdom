#include "train.h"

TrainPipeline::TrainPipeline(std::string init_model,
	std::string test_model, bool gpu) : train_model(globalConfig.modelPath + init_model, gpu), inference_model(globalConfig.modelPath + init_model, gpu),
	prev_policy(globalConfig.modelPath + test_model, gpu), current_best_model_file(test_model), gpu(gpu){
	state_batch = new std::vector<float>(globalConfig.inputChannel * globalConfig.batchSize * inputSize);
	nextmove_batch = new std::vector<float>(globalConfig.batchSize * outputSize);
	score_batch = new std::vector<float>(globalConfig.batchSize);
	result_batch = new std::vector<float>(globalConfig.batchSize);
	map_batch = new std::vector<float>(globalConfig.batchSize * boardSize);

	gameBuffer[0] = new std::deque<std::shared_ptr<TrainData>>();
	gameBuffer[1] = new std::deque<std::shared_ptr<TrainData>>();
	
	save_cnt = 0;
	std::smatch match;
    std::regex re("(\\d+)");
    if (std::regex_search(init_model, match, re)) {
        save_cnt = std::stoi(match[1]);
    }

	total_game_length = 0;
	total_score_diff = 0;
	for(int i=0; i<4; ++i)
		wintype_counter[i] = 0;
	for(int i=0; i<7; ++i)
		train_losses[i].reserve(globalConfig.compare_game_cnt);
}

void TrainPipeline::start_self_play(MCTS* player, bool is_shown, float temp, int n_games) {
	Game game_manager = Game();
	int moveCnt = 0;
	MoveData moveProb;
	InputMatrix state;

	std::vector<Move> sequence;
	std::vector<TrainData> buffer;

	#ifdef measureTime
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	#endif

	while (true) {
		state = PolicyValueNet::getData(game_manager);

		if(moveCnt < 4)
			moveProb = player->getMoveProb(temp); // temp : actually 1/temp high temp -> less random
		else
			moveProb = player->getMoveProb(temp * 5); // infinitesimal temp
		
		auto m = std::get<0>(moveProb);
		auto [winner, wintype, map] = game_manager.makeMoveWithStat(m);

		if(m != RESIGNMOVE){
			sequence.push_back(m);
		}
		else{ // Agent only resigns when capture is unavoidable.
			wintype = CAPTURE;
		}

		if (winner == EMPTY) {
			// add placeHolder value to buffer
			buffer.emplace_back(state, std::get<1>(moveProb), 0, 0.0f, std::vector<float>{});

			if(!player->jump(m)){ // very rare case
				std::cerr << "game manager's state : " << std::endl;
				ModelCompare::displayBoardGUI(false, game_manager);
				std::cout << std::endl;
				for(const auto& i : sequence){
					std::cerr << static_cast<int>(i.first) << "," << static_cast<int>(i.second) << " ";
				}
				std::cerr << "\n";
				player->reset();
				#ifdef measureTime
				player->resetTimeStats();
				#endif
				return;
			}
			moveCnt++;
		}

		else {
			// if position is black's turn to move, judge from white's perspective.
			float result = (winner == BLACK) ? -0.9f : 0.9f;
			float score_diff = game_manager.scoreDiff(WHITE); // komi not applied.

			// calculate train stats
			total_score_diff.fetch_add((int)score_diff);
			total_game_length.fetch_add(sequence.size());
			wintype_counter[((winner == BLACK) ? 2 : 0) + ((wintype == CAPTURE) ? 1 : 0)].fetch_add(1);

			if (is_shown) {
				std::cout << "\n";
				for(const auto& i : sequence){
					std::cout << static_cast<int>(i.first) << "," << static_cast<int>(i.second) << " ";
				}
				std::cout << "\n";
				std::cout << "episode length : " << sequence.size() << " winner : " << (int)winner << " wintype : " << (int)wintype << " score diff : " << score_diff << "\n\n";
				
				#ifdef measureTime
				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				std::vector<int> timeStats = player->getTimeStats();
				
				std::cout << "average expand time : " << timeStats[0] / sequence.size() << "[us]\n";
				std::cout << "average evaluate time : " << timeStats[1] / sequence.size() << "[us]\n";
				std::cout << "average makeMove time : " << timeStats[4] / sequence.size() << "[us]\n";
				std::cout << "average extra time : " << timeStats[5] / sequence.size() << "[us]\n";
				std::cout << "average cache insert time : " << timeStats[8] / sequence.size() << "[us]\n";
				std::cout << "average cache find time : " << timeStats[9] / sequence.size() << "[us]\n";
				std::cout << "eval cache hit rate : " << static_cast<float>(timeStats[6]) / (sequence.size() * nPlayout) << "\n";
				std::cout << "terminal hit rate : " << static_cast<float>(timeStats[7]) / (sequence.size() * nPlayout) << "\n";
				std::cout << "average move time : " << std::chrono::duration_cast<std::chrono::milliseconds>(middle - begin).count() / sequence.size() << "[ms]\n";
				std::cout << "move time : " << std::chrono::duration_cast<std::chrono::milliseconds>(middle - begin).count() << "[ms]\n";
				std::cout << "total time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]\n";
				#endif
			}

			#ifdef measureTime
			player->resetTimeStats();
			#endif

			if(wintype == CAPTURE){
				std::vector<float> maskedMap(boardSize, 0.0f);
				int idx = 0;
				for(const auto& move : sequence){
					if(move != PASSMOVE && map[move.first * colSize + move.second] != 0.0f){
						maskedMap[move.first * colSize + move.second] = 1.0f;
						break;
					}
					idx++;
				}

				if(idx % 2 != 0){
					result = -result;
					score_diff = -score_diff;
				}
				for(int i=idx; i<buffer.size(); ++i){
					std::get<2>(buffer[i]) = result;
					std::get<3>(buffer[i]) = score_diff;
					std::get<4>(buffer[i]) = maskedMap;
					insertData(buffer[i], CAPTURE);
					result = -result; // switch color
					score_diff = -score_diff;
					int m = sequence[i].first * colSize + sequence[i].second;
					maskedMap[m] = map[m];
				}
			}

			else{
				std::vector<float> maps[2];
				maps[0] = std::move(map);
				maps[1].reserve(boardSize);
				for(auto v : maps[0])
					maps[1].push_back(-v);
				std::vector<float> capMap(boardSize, 0.0f);

				int idx = 0;
				static long toWhere = 0;
				for(TrainData& data : buffer){
					std::get<2>(data) = result;
					std::get<3>(data) = score_diff;
					if(toWhere++ % 2 == 0){
						std::get<4>(data) = maps[idx];
						insertData(data, SCORE);
					}
					else{
						std::get<4>(data) = capMap;
						insertData(data, CAPTURE);
					}
					
					result = -result; // switch color
					score_diff = -score_diff;
					idx = 1 - idx;
				}
			}

			player->reset();
			return;
		}
	}
}

void TrainPipeline::insertData(const TrainData& data, uint8_t wintype) {
	std::vector<std::shared_ptr<TrainData>> rotatedData = generateDihedralTransformations(data);
	int idx = (wintype == CAPTURE) ? 0 : 1;

	buffer_mutex.lock();
	for(std::shared_ptr<TrainData> data : rotatedData){ // add data to the buffer
		gameBuffer[idx]->push_back(data);
	}

	while(gameBuffer[idx]->size() > globalConfig.capacity){ // if full, remove data from front
		gameBuffer[idx]->pop_front();
	}
	buffer_mutex.unlock();
}

void TrainPipeline::train(){
	for(int bufferIdx = 0; bufferIdx < 2; ++bufferIdx){
		int B = std::min((int)(gameBuffer[bufferIdx]->size()), globalConfig.batchSize);
		if(B == 0)
			continue;
			
		std::vector<int> indices = select_indices(std::min((int)(gameBuffer[bufferIdx]->size()), globalConfig.capacity), B); // randomly select samples from buffer
		std::vector<std::shared_ptr<TrainData>> batch_data;
		batch_data.reserve(B);

		buffer_mutex.lock(); 
		for(int i=0; i<B; ++i){
			std::shared_ptr<TrainData> data = (*(gameBuffer[bufferIdx]))[indices[i]];
			batch_data.push_back(data);
		}
		buffer_mutex.unlock();

		// copy data from gameBuffer to batch
		for (int i = 0; i < B; ++i) {
			const auto& data = *batch_data[i];

			const auto& state = std::get<0>(data);
			const auto& nextmove = std::get<1>(data);
			const auto& map = std::get<4>(data);

			int state_offset = i * globalConfig.inputChannel * inputSize;
			int move_offset  = i * outputSize;
			int map_offset = i * boardSize;

			std::copy(state.begin(), state.end(), state_batch->begin() + state_offset);
			std::copy(nextmove.begin(), nextmove.end(), nextmove_batch->begin() + move_offset);
			std::copy(map.begin(), map.end(), map_batch->begin() + map_offset);

			(*result_batch)[i] = std::get<2>(data);
			(*score_batch)[i] = std::get<3>(data);
		}
		// std::cout << "state batch : " << std::endl;
		// for(int i=0; i<inputChannel * inputSize; ++i)
		// 	std::cout << (*state_batch)[i] << " ";
		// std::cout << "\n nextmove batch : ";

		// for(int i=0; i<outputSize; ++i)
		// 	std::cout << (*nextmove_batch)[i] << " ";
		// std::cout << "\n evaluation batch : " << std::endl;
		
		// std::cout << (*result_batch)[0] << std::endl;
		// if game ended by capture, score data is not used.
		if(bufferIdx == 0){
			auto [pLoss, vLoss, cmLoss] = train_model.trainCap(*state_batch, *nextmove_batch, *result_batch, *map_batch, learning_rate);
			train_losses[0].push_back(pLoss);
			train_losses[1].push_back(vLoss);
			train_losses[2].push_back(cmLoss);
		}
		else{
			auto [pLoss, vLoss, sLoss, smLoss] = train_model.trainSc(*state_batch, *nextmove_batch, *result_batch, *score_batch, *map_batch, learning_rate);
			train_losses[3].push_back(pLoss);
			train_losses[4].push_back(vLoss);
			train_losses[5].push_back(sLoss);
			train_losses[6].push_back(smLoss);
		}
	}
}

void TrainPipeline::run(const int game_batch_num, const int inference_thread_num, const bool is_shown, float temp, const std::string& model_prefix)
{
	std::string model_file;

	std::atomic<bool> stop_flag = false;
	std::atomic<bool> start_flag = false; // flag to indicate if self-play has started
	std::atomic<bool> pause_flag = false;
	std::atomic<bool> train_paused = false;
	std::vector<std::atomic<bool>> self_play_paused(inference_thread_num);
	std::mutex pause_mutex, train_mutex, save_mutex;
	std::condition_variable pause_cv, train_cv;

	std::vector<std::thread> self_play_threads;
	std::vector<MCTS> mcts_players; // MCTS players of size train_thread_num
	mcts_players.reserve(inference_thread_num);

	auto evaluator = new Evaluator(&inference_model);
	for(int i=0; i<inference_thread_num; ++i){
		mcts_players.emplace_back(evaluator);
		self_play_paused[i] = false;
	}

	int train_iter = 0;
	// Self-play threads
	for(int j=0; j<inference_thread_num; ++j){
		self_play_threads.emplace_back([&, j] {
			for (int i = 0; i < game_batch_num / inference_thread_num && !stop_flag; ++i) {
				self_play_paused[j].store(false);
				start_self_play(&(mcts_players[j]), is_shown && (j == 0), temp, 1); // modifies state_buffer
				self_play_paused[j].store(true);
				pause_cv.notify_one();

				if(!start_flag && gameBuffer[0]->size() > globalConfig.batchSize){
					start_flag = true; // signal that self-play has started
					train_cv.notify_one(); // notify train thread
				}

				save_mutex.lock(); // critical part

				if (((++games_played + save_cnt) % globalConfig.save_freq) == 0) {
					std::cout << "save model" << std::endl;
					pause_flag = true; // asks other threads to pause
					train_cv.notify_one(); // notify train thread

					std::unique_lock<std::mutex> lock(pause_mutex);
					pause_cv.wait(lock, [&] { bool s = train_paused.load(); 
						for(int k=0; k<inference_thread_num; ++k) s = s & self_play_paused[k].load();
							return s; }); // wait until all train and self_play threads are paused

					model_file = model_prefix + std::to_string(games_played + save_cnt) + ".pt";
					const std::string save_path = globalConfig.modelPath + model_file;
					train_model.save_model(save_path); // save model to file

					if(globalConfig.googleDrive)
						train_model.save_model(globalConfig.drivePath + model_file); // save model to file

					std::cout << "model properly saved " << games_played << std::endl;
					std::cout << "train_iter : " << train_iter << std::endl; // check train/inference balance. 
					std::cout << "wintype count : " << wintype_counter[0] << " " << wintype_counter[1] << " " << wintype_counter[2] << " " << wintype_counter[3] << std::endl;
					int game_played = wintype_counter[0] + wintype_counter[1] + wintype_counter[2] + wintype_counter[3];
					std::cout << "games played " << game_played << std::endl;
					std::cout << "average score difference : " << (float)total_score_diff / game_played << std::endl;
					std::cout << "average game length : " << (float)total_game_length / game_played << std::endl;
					std::cout << "train losses : " << std::endl;
					for(int i=0; i<7; ++i){
						for(float l : train_losses[i])
							std::cout << l << " ";
						std::cout << std::endl;
					}

					total_game_length = 0;
					total_score_diff = 0;
					for(int i=0; i<4; ++i)
						wintype_counter[i] = 0;
					for(int i=0; i<7; ++i)
						train_losses[i].clear();
					
					if((games_played + save_cnt) % globalConfig.check_freq == 0){
						globalConfig = loadConfig("../configs/compare_config.json");

						float win_rate = ModelCompare::policy_evaluate(model_file, current_best_model_file, 
							std::cout, std::cout, false, true, 0.5f, globalConfig.compare_game_cnt / 2, globalConfig.compare_thread_num);
						std::cout << "model " << model_file << " vs " << current_best_model_file << 
						" winrate " << win_rate << std::endl;

						if(win_rate > 0.5f){
							std::cout << "Best model updated! " << current_best_model_file << " to " << model_file << std::endl;
							current_best_model_file = model_file;
							train_model.save_model(globalConfig.modelPath + model_prefix + std::to_string(games_played + save_cnt) + "B.pt"); // best models are saved
						}
						
						globalConfig = loadConfig("../configs/train_config.json");
					}

					// critical section
					evaluator->updateModel(&train_model); // synchronize train_model and inference_model
					
					pause_flag.store(false); // restart train thread
					train_cv.notify_one(); // notify train thread
				}
				
				save_mutex.unlock();
			}

			self_play_paused[j].store(true);
			pause_cv.notify_one();
		});
	}

    // Training thread
    std::thread train_thread([&] {
        while (true) {
            std::unique_lock<std::mutex> lock(train_mutex);
            train_cv.wait(lock, [&] { return stop_flag || start_flag || pause_flag ^ train_paused || gameBuffer[0]->size() > globalConfig.batchSize; });

			if(stop_flag){
				break;
			}
			else if(pause_flag ^ train_paused){
				if(train_paused)
					std::cerr << "train resumed" << std::endl;
				else
					std::cerr << "train paused" << std::endl;
				train_paused.store(pause_flag.load());
				pause_cv.notify_one();
			}
            else if (gameBuffer[0]->size() > globalConfig.batchSize && !pause_flag) {
				std::this_thread::sleep_for(std::chrono::milliseconds(globalConfig.train_wait_time / inference_thread_num));
				train_iter++;
                train(); 
            }
        }
    });

	for(auto& th : self_play_threads){
		th.join();
	}
	stop_flag = true;
    train_cv.notify_one();
    train_thread.join();
	delete evaluator;
}

void TrainPipeline::pin_threads_to_core(std::thread& th, int core_id){
#ifdef __linux__
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);
	int rc = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
	if (rc != 0) {
		std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
	}
#endif
}