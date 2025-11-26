#include "train.h"

float TrainPipeline::start_play(std::array<MCTS*, 2> player_list, std::ostream& part_res, bool is_shown, float temp) { // black wins : 1.0f, white wins : 0.0f
	Game game_manager = Game();
	int diff, idx = 0;
	std::pair<int, int> move;
    color res;
	std::vector<std::pair<int, int>> seq;

	while (true) {
		move = player_list[idx]->getMove(temp);
		seq.push_back(move);
        res = game_manager.makeMove(move.first, move.second);

		if (res == EMPTY) {
            player_list[0]->jump(move);
            player_list[1]->jump(move);
			idx = 1 - idx;
			continue;
		}
		
        player_list[0]->reset();
        player_list[1]->reset();
		

		if (is_shown) {
			for (auto& moves : seq)
				part_res << moves.first << moves.second << " ";
		}

        if(is_shown)
            part_res << res << std::endl;
        return res == BLACK ? 1.0f : 0.0f;
	}
}

void TrainPipeline::play(const std::string& model, color side, int playout, float temp, bool gpu, bool shown) {
	Game game_manager = Game();
	PolicyValueNet pv(model_path + model, gpu);
	auto eval_cache = new EvalCache();
	auto trans_table = new std::unordered_map<HashValue, Node*>();

	MCTS player = MCTS(playout, &pv, eval_cache, trans_table);
	std::pair<int, int> cord;
	color res;

	while (true) {
		if (side == game_manager.getTurn()) {
			std::cin >> cord.first >> cord.second;
		}
		else {
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
			cord = player.getMove(temp);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "move time : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
		}

		res = game_manager.makeMove(cord.first, cord.second);
		game_manager.displayBoardGUI();
		if (res != EMPTY) {
			game_manager.onGameEnd(res);
			break;
		}
        player.jump(cord);
	}

	delete trans_table;
	delete eval_cache;
	return;
}

float TrainPipeline::policy_evaluate(const std::string& mod_one, const std::string& mod_two, std::ostream& total_res, std::ostream& part_res, bool is_shown,
	bool gpu, float temp, int n_games) {
	PolicyValueNet po(model_path + mod_one, gpu);
	PolicyValueNet pt(model_path + mod_two, gpu);
	auto eval_cache_o = new EvalCache();
	auto eval_cache_t = new EvalCache();
	auto trans_table_o = new std::unordered_map<HashValue, Node*>();
	auto trans_table_t = new std::unordered_map<HashValue, Node*>();
	MCTS* base_player = new MCTS(n_playout, &po, eval_cache_o, trans_table_o);
	MCTS* oppo_player = new MCTS(n_playout, &pt, eval_cache_t, trans_table_t);

	std::vector<bool> b = play_match(base_player, oppo_player, total_res, is_shown, temp, n_games);

	delete base_player;
	delete oppo_player;
	delete eval_cache_o;
	delete eval_cache_t;
	delete trans_table_o;
	delete trans_table_t;
	return std::count(b.begin(), b.end(), true) / static_cast<float>(n_games << 1);
}

std::vector<float> TrainPipeline::policy_evaluate(std::vector<std::string> model_list,
	std::ostream& total_res, bool is_shown, bool gpu, float temp, int n_games) {
	int N = model_list.size();
	
	auto trans_tables = std::vector<std::unordered_map<HashValue, Node*>*>();
	trans_tables.reserve(N);
	for(int i=0; i<N; ++i)
		trans_tables.push_back(new std::unordered_map<HashValue, Node*>());

	auto caches = std::vector<EvalCache*>();
	caches.reserve(N);
	for(int i=0; i<N; ++i)
		caches.push_back(new EvalCache());

	std::vector<MCTS*> players(N);
	for (int i = 0; i < N; ++i) {
		PolicyValueNet* pv = new PolicyValueNet(model_path + model_list[i], gpu);
		players[i] = new MCTS(n_playout, pv, caches[i], trans_tables[i]);
	}

	bool load_from_file = false;
	EloCalculator elo_calculator(model_path + "ratings.txt", model_list, load_from_file);

	for(int i=1; i<N; ++i){
		for(int j=0; j<N-i; ++j){
			std::vector<bool> b = play_match(players[j], players[j+i], total_res, is_shown, temp, n_games);
			elo_calculator.UpdateRatings(j, j+i, b);
			
			int win_cnt = std::count(b.begin(), b.end(), true);
			total_res << "Model " << model_list[j] << " VS Model " << model_list[j+i] << " : " 
				<< win_cnt << "/" << (n_games << 1) << " (" << (win_cnt * 100.0f / (n_games << 1)) << "%)" << std::endl;
		}
	}

	if(load_from_file){
		elo_calculator.saveRating(model_path + "ratings.txt", model_list);
	}
	std::vector<float> ratings = elo_calculator.GetRatings(/*adjust=*/false);
	for(int i=0; i<N; ++i){
		total_res << model_list[i] << " Elo Rating : " << ratings[i] << std::endl;
	}

	for(int i=0; i<N; ++i){
		delete players[i];
		delete caches[i];
		delete trans_tables[i];
	}
	return ratings;
}

TrainPipeline::TrainPipeline(std::string init_model,
	std::string test_model, bool gpu) : train_model(model_path + init_model, gpu), inference_model(model_path + init_model, gpu),
	prev_policy(model_path + test_model, gpu){
	state_batch = new std::array<float, inputDepth * batchSize * inputSize>();
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
	std::array<float, inputDepth * inputSize> state;
	color result;

	std::vector<std::pair<float, float>> sequence;
	std::vector<TrainData> buffer;

	#ifdef measureTime
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	#endif

	while (true) {
		// 처음 네 수 dirichlet factor 0.5. 
		state = PolicyValueNet::getData(game_manager);

		if(moveCnt < 4)
			moveProb = player->getMoveProb(temp); // temp : actually 1/temp high temp -> less random
		else
			moveProb = player->getMoveProb(temp * 5); // infinitesimal temp
		
		auto m = std::get<0>(moveProb);
		sequence.push_back(m);
		result = game_manager.makeMove(m.first, m.second);

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
				// for(auto& i : sequence){
				// 	std::cout << i.first << "," << i.second << " ";
				// }
				// std::cout << "\n";
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
				//std::cout << "eval norot cache hit rate : " << static_cast<float>(timeStats[7]) / (sequence.size() * n_playout) << "\n";
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
	std::vector<int> indices = select_indices(game_buffer->size(), batchSize); // randomly select samples from buffer
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
		TrainData* data = (*game_buffer)[indices[i]];
		for(int j=0; j < inputDepth * inputSize; ++j){ 
			(*state_batch)[i * inputDepth * inputSize + j] = std::get<0>(*data)[j];
		}

		for(int j=0; j < outputSize; ++j){
			(*nextmove_batch)[i * outputSize + j] = std::get<1>(*data)[j];
		}

		(*winner_batch)[i] = std::get<2>(*data);
	}
	// std::cout << "state batch : " << std::endl;
	// for(int i=0; i<inputDepth * inputSize; ++i)
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
	auto eval_cache = new EvalCache();

	auto trans_tables = std::vector<std::unordered_map<HashValue, Node*>*>();
	trans_tables.reserve(inference_thread_num);
	for(int i=0; i<inference_thread_num; ++i)
		trans_tables.push_back(new std::unordered_map<HashValue, Node*>());

	for(int i=0; i<inference_thread_num; ++i){
		mcts_players.emplace_back(n_playout, &inference_model, eval_cache, trans_tables[i]);
		self_play_paused[i] = false;
	}

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

					// critical section
					model_file = model_prefix + std::to_string(games_played + save_cnt);
					synchronize_model(inference_model, train_model, mcts_players); // synchronize train_model and inference_model
					inference_model.save_model(model_path + model_file + std::string(".pt")); // save model to file
					std::cout << "model properly saved" << std::endl;
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
				std::this_thread::sleep_for(std::chrono::milliseconds(3200 / inference_thread_num));
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
}

std::vector<bool> TrainPipeline::play_match(MCTS* player_one, MCTS* player_two,
		std::ostream& total_res, bool is_shown, float temp, int n_games) {

	std::vector<bool> result(n_games << 1);
	for (int i = 0; i < n_games; ++i) {
		//player_one plays as black
		result[i] = static_cast<bool>(TrainPipeline::start_play({ player_one, player_two }, total_res, is_shown, temp));
		// total_res << win_cnt << "/" << i + 1 << std::endl;
	}

	for (int i = n_games; i < (n_games << 1); ++i) {
		result[i] = !static_cast<bool>(TrainPipeline::start_play({ player_two, player_one }, total_res, is_shown, temp));
		// total_res << win_cnt << "/" << i + 1 << std::std::endl;
	}
	return result;
}

void TrainPipeline::synchronize_model(PolicyValueNet& target, PolicyValueNet& source, std::vector<MCTS>& players) {
	torch::NoGradGuard no_grad;
	auto src_state = source.policy_value_net->named_parameters();
	auto dst_state = target.policy_value_net->named_parameters();

	for (auto& item : src_state) {
		dst_state[item.key()].copy_(item.value());
	}
	for (auto& item : dst_state) {
		target.policy_value_net->named_parameters()[item.key()].copy_(item.value());
	}

	players[0].updateModel(); // only one player needs to clear cache.
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