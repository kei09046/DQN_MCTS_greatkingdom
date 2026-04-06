#include "train.h"

TrainPipeline::TrainPipeline(std::string init_model,
	std::string test_model, bool gpu) : train_model(globalConfig.modelPath + init_model, gpu), inference_model(globalConfig.modelPath + init_model, gpu),
	prev_policy(globalConfig.modelPath + test_model, gpu), current_best_model_file(test_model), gpu(gpu){
	state_batch = new std::vector<float>(globalConfig.inputChannel * globalConfig.batchSize * inputSize);
	nextmove_batch = new std::vector<float>(globalConfig.batchSize * outputSize);
	score_batch = new std::vector<float>(globalConfig.batchSize);
	result_batch = new std::vector<float>(globalConfig.batchSize);
	game_buffer = new std::deque<std::shared_ptr<TrainData>>();
	
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
}

void TrainPipeline::start_self_play(MCTS* player, bool is_shown, float temp, int n_games) {
	Game game_manager = Game();
	int moveCnt = 0;
	MoveData moveProb;
	InputMatrix state;

	std::vector<std::pair<float, float>> sequence;
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
		sequence.push_back(m);
		auto [winner, wintype] = game_manager.makeMove(m);

		if (winner == EMPTY) {
			buffer.emplace_back(state, std::get<1>(moveProb), 0, 0.0f);
			if(!player->jump(m)){ // very rare case
				std::cerr << "game manager's state : " << std::endl;
				ModelCompare::displayBoardGUI(false, game_manager);
				std::cout << std::endl;
				for(const auto& i : sequence){
					std::cerr << i.first << "," << i.second << " ";
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
			#ifdef measureTime
			std::chrono::steady_clock::time_point middle = std::chrono::steady_clock::now();
			#endif

			int result = (winner == BLACK) ? ((wintype == SCORE) ? 2 : 3) 
			: ((wintype == SCORE) ? 0 : 1); // if position is black's turn to move, judge from white's perspective.
			float score_diff = game_manager.scoreDiff(BLACK); // komi not applied.

			// calculate train stats
			total_score_diff.fetch_add((int)score_diff);
			total_game_length.fetch_add(sequence.size());
			wintype_counter[result].fetch_add(1);
			//std::cerr << score_diff << " " << sequence.size() << " " << result << std::endl;

			for(TrainData& data : buffer){
				std::get<2>(data) = result;
				std::get<3>(data) = score_diff;
				insert_data(data);
				result = (result + 2) % 4; // switch color
				score_diff *= -1.0f;
			}
			player->reset();

			if (is_shown) {
				std::cout << "\n";
				for(const auto& i : sequence){
					std::cout << i.first << "," << i.second << " ";
				}
				std::cout << "\n";
				std::cout << "episode length : " << sequence.size() << " winner : " << (int)winner << " wintype : " << (int)wintype << "\n\n";
				
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
			return;
		}
	}
}

void TrainPipeline::insert_data(const TrainData& data) {
	std::vector<std::shared_ptr<TrainData>> rotatedData = generateDihedralTransformations(data);

	buffer_mutex.lock();
	for(std::shared_ptr<TrainData> data : rotatedData){ // add data to the buffer
		game_buffer->push_back(data);
	}

	while(game_buffer->size() > globalConfig.capacity){ // if full, remove data from front
		game_buffer->pop_front();
	}
	buffer_mutex.unlock();
}

void TrainPipeline::train(){
	std::vector<int> indices = select_indices(std::min((int)game_buffer->size(), globalConfig.capacity), globalConfig.batchSize); // randomly select samples from buffer
	std::vector<std::shared_ptr<TrainData>> batch_data;
	batch_data.reserve(globalConfig.batchSize);

	buffer_mutex.lock(); 
	for(int i=0; i<globalConfig.batchSize; ++i){
		std::shared_ptr<TrainData> data = (*game_buffer)[indices[i]];
		batch_data.push_back(data);
	}
	buffer_mutex.unlock();

	// copy data from game_buffer to batch
	// for(int i=0; i < globalConfig.batchSize; ++i){ // copies data to batch
	// 	std::shared_ptr<TrainData> data = batch_data[i];
	// 	for(int j=0; j < globalConfig.inputChannel * inputSize; ++j){ 
	// 		(*state_batch)[i * globalConfig.inputChannel * inputSize + j] = std::get<0>(*data)[j];
	// 	}

	// 	for(int j=0; j < outputSize; ++j){
	// 		(*nextmove_batch)[i * outputSize + j] = std::get<1>(*data)[j];
	// 	}

	// 	(*result_batch)[i] = std::get<2>(*data);

	// 	(*score_batch)[i] = std::get<3>(*data);
	// }
	for (int i = 0; i < globalConfig.batchSize; ++i) {
		const auto& data = *batch_data[i];

		const auto& state = std::get<0>(data);
		const auto& nextmove = std::get<1>(data);

		int state_offset = i * globalConfig.inputChannel * inputSize;
		int move_offset  = i * outputSize;

		std::copy(state.begin(), state.end(), state_batch->begin() + state_offset);
		std::copy(nextmove.begin(), nextmove.end(), nextmove_batch->begin() + move_offset);

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

	try{
		train_model.train_step(*state_batch, *nextmove_batch, *result_batch, *score_batch, learning_rate);
	}catch(const c10::Error& e){
		std::cerr << "failed while training " << std::endl;
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

				if(!start_flag && game_buffer->size() > globalConfig.batchSize){
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

					total_game_length = 0;
					total_score_diff = 0;
					for(int i=0; i<4; ++i)
						wintype_counter[i] = 0;
					
					if((games_played + save_cnt) % globalConfig.check_freq == 0){
						float win_rate = ModelCompare::policy_evaluate(model_file, current_best_model_file, 
							std::cout, std::cout, false, true, 0.5f, globalConfig.compare_game_cnt / 2, globalConfig.compare_thread_num);
						std::cout << "model " << model_file << " vs " << current_best_model_file << 
						" winrate " << win_rate << std::endl;

						if(win_rate > 0.5f){
							std::cout << "Best model updated! " << current_best_model_file << " to " << model_file << std::endl;
							current_best_model_file = model_file;
							train_model.save_model(globalConfig.modelPath + model_prefix + std::to_string(games_played + save_cnt) + "B.pt"); // best models are saved
						}
						else if(win_rate < 0.4f){
							//std::cout << "model fallback!" << model_file << " to " << current_best_model_file << std::endl;
							//train_model.load_model(globalConfig.modelPath + current_best_model_file);
						}
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
            train_cv.wait(lock, [&] { return stop_flag || start_flag || pause_flag ^ train_paused || game_buffer->size() > globalConfig.batchSize; });

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
            else if (game_buffer->size() > globalConfig.batchSize && !pause_flag) {
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