#include "train.h"

TrainPipeline::TrainPipeline(std::string init_model,
	std::string test_model, bool gpu) : train_model(model_path + init_model, gpu), inference_model(model_path + init_model, gpu),
	prev_policy(model_path + test_model, gpu), current_best_model_file(test_model), gpu(gpu){
	state_batch = new std::array<float, inputChannel * batchSize * inputSize>();
	nextmove_batch = new std::array<float, batchSize* (outputSize)>();
	winner_batch = new std::array<float, batchSize>();
	game_buffer = new std::deque<TrainData*>();

	save_cnt = 0;
	std::smatch match;
    std::regex re("(\\d+)");
    if (std::regex_search(init_model, match, re)) {
        save_cnt = std::stoi(match[1]);
    }
}

void TrainPipeline::start_self_play(MCTS* player, bool is_shown, float temp, int n_games) {
	Game game_manager = Game();
	int moveCnt = 0;
	MoveData moveProb;
	InputMatrix state;
	color result;

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
		result = game_manager.makeMove(m);

		if (result == EMPTY) {
			buffer.emplace_back(state, std::get<1>(moveProb), 0.0f, 0);
			if(!player->jump(m)){ // very rare case
				std::cerr << "game manager's state : " << std::endl;
				game_manager.displayBoardGUI();
				std::cout << std::endl;
				for(auto& i : sequence){
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

			float value = (result == BLACK) ? -1.0f : 1.0f;
			for(TrainData& data : buffer){
				std::get<2>(data) = value;
				insert_data(data);
				value = -value;
			}
			player->reset();

			if (is_shown) {
				std::cout << "\n";
				for(auto& i : sequence){
					std::cout << i.first << "," << i.second << " ";
				}
				std::cout << "\n";
				std::cout << "episode length : " << sequence.size() << " winner : " << static_cast<int>(result) << "\n\n";
				
				#ifdef measureTime
				std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
				std::vector<int> timeStats = player->getTimeStats();
				
				std::cout << "average expand time : " << timeStats[0] / sequence.size() << "[us]\n";
				std::cout << "average evaluate time : " << timeStats[1] / sequence.size() << "[us]\n";
				std::cout << "average makeMove time : " << timeStats[4] / sequence.size() << "[us]\n";
				std::cout << "average extra time : " << timeStats[5] / sequence.size() << "[us]\n";
				std::cout << "average cache insert time : " << timeStats[8] / sequence.size() << "[us]\n";
				std::cout << "average cache find time : " << timeStats[9] / sequence.size() << "[us]\n";
				std::cout << "eval cache hit rate : " << static_cast<float>(timeStats[6]) / (sequence.size() * n_playout) << "\n";
				std::cout << "terminal hit rate : " << static_cast<float>(timeStats[7]) / (sequence.size() * n_playout) << "\n";
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

void TrainPipeline::insert_data(TrainData data) {
	std::vector<TrainData*> rotatedData = generateDihedralTransformations(data);

	buffer_mutex.lock();
	for(TrainData* data : rotatedData){ // add data to the buffer
		game_buffer->push_back(data);
	}

	while(game_buffer->size() > capacity){ // if full, remove data from front
		TrainData* data = game_buffer->front();
		if(std::get<3>(*data) == 0){
			delete data;
		}
		else{ // being used during training, mark for deletion later
			std::get<3>(*data) |= 1; 
		}
		game_buffer->pop_front();
	}
	buffer_mutex.unlock();
}

void TrainPipeline::train(){
	std::vector<int> indices = select_indices(std::min(game_buffer->size(), capacity), batchSize); // randomly select samples from buffer
	std::vector<TrainData*> batch_data(batchSize);

	buffer_mutex.lock(); 
	for(int i=0; i<batchSize; ++i){
		TrainData* data = (*game_buffer)[indices[i]];
		std::get<3>(*data) |= 2; // mark as being used during training
		batch_data[i] = data;
	}
	buffer_mutex.unlock();

	// copy data from game_buffer to batch
	for(int i=0; i < batchSize; ++i){ // copies data to batch
		TrainData* data = batch_data[i];
		for(int j=0; j < inputChannel * inputSize; ++j){ 
			(*state_batch)[i * inputChannel * inputSize + j] = std::get<0>(*data)[j];
		}

		for(int j=0; j < outputSize; ++j){
			(*nextmove_batch)[i * outputSize + j] = std::get<1>(*data)[j];
		}

		(*winner_batch)[i] = std::get<2>(*data);
	}
	// std::cout << "state batch : " << std::endl;
	// for(int i=0; i<inputChannel * inputSize; ++i)
	// 	std::cout << (*state_batch)[i] << " ";
	// std::cout << "\n nextmove batch : ";

	// for(int i=0; i<outputSize; ++i)
	// 	std::cout << (*nextmove_batch)[i] << " ";
	// std::cout << "\n evaluation batch : " << std::endl;
	
	// std::cout << (*winner_batch)[0] << std::endl;

	buffer_mutex.lock(); // remove data that were marked for deletion
	for(TrainData* data : batch_data){
		if(std::get<3>(*data) & 1){ // if marked for deletion, delete
			delete data;
		}
		else{
			std::get<3>(*data) = 0; // unmark
		}
	}
	buffer_mutex.unlock();

	for(int i=0; i<epochs; ++i)
		train_model.train_step(*state_batch, *nextmove_batch, *winner_batch, learning_rate);
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
		mcts_players.emplace_back(n_playout, evaluator);
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

				if(!start_flag && game_buffer->size() > batchSize){
					start_flag = true; // signal that self-play has started
					train_cv.notify_one(); // notify train thread
				}

				save_mutex.lock(); // critical part
				if (((++games_played + save_cnt) % save_freq) == 0) {
					std::cout << "save model" << std::endl;
					pause_flag = true; // asks other threads to pause
					train_cv.notify_one(); // notify train thread

					std::unique_lock<std::mutex> lock(pause_mutex);
					pause_cv.wait(lock, [&] { bool s = train_paused.load(); 
						for(int k=0; k<inference_thread_num; ++k) s = s & self_play_paused[k].load();
							return s; }); // wait until all train and self_play threads are paused

					model_file = model_prefix + std::to_string(games_played + save_cnt) + ".pt";
					const std::string save_path = model_path + model_file;
					train_model.save_model(save_path); // save model to file
					#ifdef googleDrive
					train_model.save_model(drive_path + model_file); // save model to file
					#endif
					std::cout << "model properly saved " << games_played << std::endl;
					std::cout << "train_iter : " << train_iter << std::endl; // check train/inference balance. 
					
					if((games_played + save_cnt) % check_freq == 0){
						float win_rate = ModelCompare::policy_evaluate(model_file, current_best_model_file, 
							std::cout, std::cout, false, true, 0.5f, compare_game_cnt / 2, compare_thread_num);
						std::cout << "model " << model_file << " vs " << current_best_model_file << 
						" winrate " << win_rate << std::endl;
						if(win_rate > 0.55f){
							std::cout << "Best model updated! " << current_best_model_file << " to " << model_file << std::endl;
							current_best_model_file = model_file;
						}
						else if(win_rate < 0.45f){
							std::cout << "model fallback!" << model_file << " to " << current_best_model_file << std::endl;
							train_model.load_model(model_path + current_best_model_file);
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
            train_cv.wait(lock, [&] { return stop_flag || start_flag || pause_flag ^ train_paused || game_buffer->size() > batchSize; });

			if(stop_flag){
				break;
			}
			else if(pause_flag ^ train_paused){
				train_paused.store(pause_flag.load());
				pause_cv.notify_one();
			}
            else if (game_buffer->size() > batchSize && !pause_flag) {
				std::this_thread::sleep_for(std::chrono::milliseconds(train_wait_time / inference_thread_num));
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