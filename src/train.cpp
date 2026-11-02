#include "train.h"

TrainPipeline::TrainPipeline(std::string initModel,
	std::string testModel, bool gpu) : trainModel(globalConfig.modelPath + initModel, gpu), inferenceModel(globalConfig.modelPath + initModel, gpu),
	currentBestModelFile(testModel), gpu(gpu), captureRatio(0.5f),
	replayBuffer(globalConfig.capacity, globalConfig.trainStartPoint, globalConfig.windowFraction){
	stateBatch = new std::vector<float>(globalConfig.inputChannel * globalConfig.batchSize * inputSize);
	nextmoveBatch = new std::vector<float>(globalConfig.batchSize * outputSize);
	scoreBatch = new std::vector<float>(globalConfig.batchSize);
	resultBatch = new std::vector<float>(globalConfig.batchSize);
	outcomeBatch = new std::vector<float>(globalConfig.batchSize);
	mapBatch = new std::vector<float>(globalConfig.batchSize * boardSize);
	typeBatch = new std::vector<Trainhead>(globalConfig.batchSize);
	
	saveCnt = PolicyValueNet::parse_model_name(initModel).games.value_or(0);
	if(trainModel.model_type() != globalConfig.modelPrefix || trainModel.num_blocks() != globalConfig.numHiddenBlock)
		std::cout << "note : " << initModel << " is a " << trainModel.model_type() << " model with " << trainModel.num_blocks()
			<< " blocks; config model_prefix / numHiddenBlock only apply to \"none\"" << std::endl;
	learningRate = globalConfig.initialLearningRate;

	// Load optimizer state if possible.
	const std::string optFile = globalConfig.modelPath + std::filesystem::path(initModel).stem().string() + ".opt";
	if(std::filesystem::exists(optFile)){
		try{
			trainModel.load_optimizer(optFile);
			learningRate = trainModel.get_learning_rate(); // continues any halving from the earlier run
		}
		catch(const std::exception& e){
			std::cerr << "could not load optimizer state " << optFile << " (" << e.what() << "), starting with a fresh optimizer" << std::endl;
		}
	}
	else
		std::cout << "no optimizer state " << optFile << ", starting with a fresh optimizer" << std::endl;
	std::cout << "learning rate : " << learningRate << std::endl;

	totalGameLength = 0;
	totalScoreDiff = 0;
	for(int i=0; i<4; ++i)
		wintypeCounter[i] = 0;
	for(int i=0; i<9; ++i)
		trainLosses[i].reserve(globalConfig.compareGameCnt);
}

void label_finished_game(std::vector<TrainData>& buffer, const std::vector<Move>& sequence,
	const std::vector<std::optional<float>>& rootValue, Color winner, Wintype wintype,
	const std::vector<float>& map, float blackLead, float valueQMix, ValueTargetStats* stats){
	const bool byCapture = (wintype == CAPTURE);

	// value target of position 0, from the perspective of the player who just moved (white). Flips every move.
	float result = (winner == BLACK) ? -0.9f : 0.9f;

	// score ending : occupancy map from black's (= position 0's) perspective, negated on odd positions.
	// score target of position 0 is white's lead (komi not applied), flipped every move like the value.
	std::vector<float> occupancy[2];
	float scoreDiff = 0.0f;
	if(!byCapture){
		occupancy[0] = map;
		for(auto v : occupancy[0])
			occupancy[1].push_back(-v);
		scoreDiff = -blackLead;
	}

	// capture ending : map marks the captured stones. A position's target marks only those already on the board there,
	// so it never asks for stones that haven't been played yet. Unsigned : the input shows whose stone each one is.
	std::vector<int> placedAt(boardSize, -1); // index into sequence of the move that placed the stone on each point
	if(byCapture){
		for(int k = 0; k < static_cast<int>(sequence.size()); ++k)
			if(sequence[k].first < rowSize)
				placedAt[sequence[k].first * colSize + sequence[k].second] = k;
	}

	for(int i=0; i<sequence.size(); ++i){
		TrainData& data = buffer.at(i);
		std::get<2>(data) = result;
		std::get<6>(data) = result; // game result z, before any mixing

		if(byCapture){
			// score / occupancy only mean something at a score ending. Capture map on every position (zeros if
			// none of the captured stones is on the board yet: none of the current stones gets captured).
			std::get<5>(data) = POLICYHEAD | VALUEHEAD | CMAPHEAD;
			std::vector<float> capture(boardSize, 0.0f);
			for(int p = 0; p < boardSize; ++p)
				if(map[p] != 0.0f && placedAt[p] >= 0 && placedAt[p] < i) // position i holds the stones of moves 0..i-1
					capture[p] = 1.0f;
			std::get<4>(data) = std::move(capture);
		}
		else{
			// nothing was captured : the capture map target is all zero (derived from OCCUPYHEAD in the loss).
			std::get<5>(data) = POLICYHEAD | VALUEHEAD | SCOREHEAD | OCCUPYHEAD | CMAPHEAD;
			std::get<3>(data) = scoreDiff;
			std::get<4>(data) = occupancy[i % 2];
		}

		// game result z mixed with the search's value q: z alone is mostly decided by mistakes made much later, so q gets
		// more weight early on. The weight falls linearly from valueQMix at position 0 to 0 at the terminal position
		// (position sequence.size(), after the last move), where z is exact.
		if(rootValue.at(i)){
			const float z = std::get<2>(data);
			const float qv = *rootValue.at(i);
			if(stats)
				stats->add(qv, z);
			const float lambda = valueQMix * static_cast<float>(sequence.size() - i) / sequence.size();
			std::get<2>(data) = (1.0f - lambda) * z + lambda * qv;
		}

		result = -result; // switch color
		scoreDiff = -scoreDiff;
	}
}

void TrainPipeline::start_self_play(MCTS* player, bool isShown, float temp, int nGames) {
	Game gameManager = Game();
	NNInput state;

	std::vector<Move> sequence;
	std::vector<TrainData> buffer;
	std::vector<int> forced;
	std::vector<bool> only; 
	std::vector<bool> recorded; // full search : inserted into the replay buffer. Fast search positions are only labelled.
	std::vector<std::optional<float>> rootValue; // search's value of each position, in the value target's perspective

	#ifdef measureTime
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	#endif

	while (true) {
		gameManager.set_policy_mask();
		state = PolicyValueNet::get_data(gameManager);

		// playout cap randomization : a fast search position keeps its place in the sequence (labels alternate per move)
		// but isn't recorded.
		const bool full = (globalConfig.fullSearchProb >= 1.0f) || (rnd::pick0or1(globalConfig.fullSearchProb) == 0);
		const MoveData md = full ? player->get_move_with_stat(temp) : MoveData{player->get_fast_move(temp), {}, 0, false, std::nullopt};
		const auto& [m, prob, forcedState, isOnlyMove, q] = md;
		//printMove(m);

		auto [winner, wintype, map] = gameManager.make_move_with_stat(m);

		assert(m != RESIGNMOVE);
		if(m != RESIGNMOVE){
			sequence.push_back(m);
			// add placeHolder value to buffer
			buffer.emplace_back(state, prob, 0.0f, 0.0f, std::vector<float>(boardSize, 0.0f), ALL, 0.0f);
			forced.push_back(forcedState);
			only.push_back(isOnlyMove);
			recorded.push_back(full);
			rootValue.push_back(full ? q : std::nullopt); // fast search values stay out of the q / z statistics
		}

		if (winner == EMPTY) {
			if(!player->jump(m)){ // very rare case
				std::cerr << "game manager's state : " << std::endl; 
				gameManager.display_board_gui(true);
				std::cout << std::endl;
				for(const auto& i : sequence){
					std::cerr << static_cast<int>(i.first) << "," << static_cast<int>(i.second) << " ";
				}
				std::cerr << "\n";
				player->reset(Game());
				#ifdef measureTime
				player->resetTimeStats();
				#endif
				return;
			}
			continue;
		}

		// game over : by capture, or by score (no legal moves / move limit). The game is never continued.
		wintypeCounter[((winner == BLACK) ? 2 : 0) + ((wintype == CAPTURE) ? 1 : 0)].fetch_add(1);
		totalGameLength.fetch_add(sequence.size());

		if (isShown) {
			std::cout << "\n";
			for(const auto& i : sequence){
				std::cout << static_cast<int>(i.first) << "," << static_cast<int>(i.second) << " ";
			}
			std::cout << "\n";
			// game_manager.displayBoardGUI(true);
			std::cout << "episode length : " << sequence.size() << " winner : " << (int)winner << " wintype : " << (int)wintype << "\n\n";
		}

		if(wintype != CAPTURE)
			totalScoreDiff.fetch_add(static_cast<int>(-gameManager.score_diff(BLACK)));

		label_finished_game(buffer, sequence, rootValue, winner, wintype, map, gameManager.score_diff(BLACK),
			globalConfig.valueQMix, &valueTargetStats);
		for(int i=0; i<sequence.size(); ++i)
			if(recorded.at(i))
				insert_data(buffer.at(i), forced.at(i), only.at(i));

		player->reset(Game());
		return;
	}
}

void TrainPipeline::load_replay_buffer(const std::string& dataFile){
	replayBuffer.load(dataFile);
	std::cout << "replay buffer loaded : " << dataFile << " (" << replayBuffer.size() << " positions, sampling window "
		<< replayBuffer.window() << ", " << replayBuffer.total_added() << " positions added in total)" << std::endl;
}

void TrainPipeline::insert_data(TrainData& data, const int& forced, const bool& only) {
	// if forced < 0 -> don't train policy head.
	// if forced = 0 and only = true -> don't train policy head.
	// if forced > 0 -> maybe add another copy?
	if(forced < 0 || (forced == 0 && only))
		std::get<5>(data) &= ~POLICYHEAD;

	replayBuffer.add(data); // stored once; a random symmetry is applied when it is sampled
}

void TrainPipeline::train(){
	const int B = globalConfig.batchSize;
	for(int iter = 0; iter < 2; ++iter){
		replayBuffer.sample_batch(B, *stateBatch, *nextmoveBatch, *resultBatch, *outcomeBatch, *scoreBatch, *mapBatch, *typeBatch);

		auto [pLoss, vLoss, sLoss, cmLoss, smLoss, pEntropy, vzLoss] = trainModel.train(*stateBatch,
			*nextmoveBatch, *resultBatch, *outcomeBatch, *scoreBatch, *mapBatch, *typeBatch, learningRate);
		if(!std::isfinite(pLoss) || !std::isfinite(vLoss) || !std::isfinite(sLoss) || !std::isfinite(smLoss)){
			skippedSteps++; // train_model.train skipped the optimizer step
			continue;
		}
		trainLosses[0].push_back(pLoss);
		trainLosses[1].push_back(vLoss);
		trainLosses[2].push_back(sLoss);
		trainLosses[3].push_back(cmLoss);
		trainLosses[4].push_back(smLoss); 
		trainLosses[5].push_back(pEntropy);
		trainLosses[6].push_back(vzLoss);
	}
}

void TrainPipeline::run(const int gameBatchNum, const int inferenceThreadNum, const bool isShown, float temp)
{
	// name of the model.
	std::string modelName;
	// copied once: globalConfig is swapped to the compare config while the train thread may be waiting.
	const size_t trainStartPoint = globalConfig.trainStartPoint;
	const float fallbackWinrate = globalConfig.fallbackWinrate;
	const int lrPatience = globalConfig.lrPatience;
	// above 0.5, so that a checkpoint of equal strength (which beats 0.5 in ~46% of 96-game comparisons by chance)
	// doesn't keep resetting the learning-rate counter.
	const float lrImprovementWinrate = globalConfig.lrImprovementWinrate;

	std::atomic<bool> stopFlag = false;
	std::atomic<bool> startFlag = false; // flag to indicate if self-play has started
	std::atomic<bool> pauseFlag = false;
	std::atomic<bool> trainPaused = false;
	std::vector<std::atomic<bool>> selfPlayPaused(inferenceThreadNum);
	std::mutex pauseMutex, trainMutex, saveMutex;
	std::condition_variable pauseCv, trainCv;

	std::vector<std::thread> selfPlayThreads;
	std::vector<MCTS> mctsAgent; // MCTS players of size train_thread_num
	mctsAgent.reserve(inferenceThreadNum);

	auto evaluator = new Evaluator(&inferenceModel);
	for(int i=0; i<inferenceThreadNum; ++i){
		mctsAgent.emplace_back(evaluator);
		selfPlayPaused[i] = false;
	}

	int trainIter = 0;	
	// Self-play threads
	for(int j=0; j<inferenceThreadNum; ++j){
		selfPlayThreads.emplace_back([&, j] {
			for (int i = 0; i < gameBatchNum / inferenceThreadNum && !stopFlag; ++i) {
				selfPlayPaused[j].store(false);
				start_self_play(&(mctsAgent[j]), isShown && (j == 0), temp, 1); // modifies state_buffer
				selfPlayPaused[j].store(true);
				pauseCv.notify_one();

				if(!startFlag && replayBuffer.size() >= trainStartPoint){
					startFlag = true; // signal that self-play has started
					trainCv.notify_one(); // notify train thread
				}

				saveMutex.lock(); // critical part

				if (((++gamesPlayed + saveCnt) % globalConfig.synchronizeFreq) == 0) {
					std::cout << "synchronize model" << std::endl;
					pauseFlag = true; // asks other threads to pause
					trainCv.notify_one(); // notify train thread

					// wait until all train and self_play threads are paused
					std::unique_lock<std::mutex> lock(pauseMutex);
					pauseCv.wait(lock, [&] { bool s = trainPaused.load(); 
						for(int k=0; k<inferenceThreadNum; ++k) s = s & selfPlayPaused[k].load();
							return s; });

					// print / setup train statistics.
					{
						std::cout << "model properly saved " << gamesPlayed << std::endl;
						std::cout << "train_iter : " << trainIter << std::endl; // check train/inference balance. 
						std::cout << "wintype count : " << wintypeCounter[0] << " " << wintypeCounter[1] << " " << wintypeCounter[2] << " " << wintypeCounter[3] << std::endl;
						int gamePlayed = wintypeCounter[0] + wintypeCounter[1] + wintypeCounter[2] + wintypeCounter[3];
						captureRatio = static_cast<float>((wintypeCounter[1] + wintypeCounter[3])) / gamePlayed;
						std::cout << "games played : " << gamePlayed << std::endl;
						std::cout << "capture ratio : " << captureRatio << std::endl;
						std::cout << "replay buffer : " << replayBuffer.size() << " stored, sampling window " << replayBuffer.window()
							<< ", " << replayBuffer.total_added() << " positions added in total" << std::endl;
						if(skippedSteps > 0 || replayBuffer.dropped() > 0)
							std::cout << "non-finite data : " << skippedSteps << " training steps skipped, "
								<< replayBuffer.dropped() << " positions dropped in total" << std::endl;
						skippedSteps = 0;
						{
							const auto [n, corr, mae] = valueTargetStats.take_summary();
							std::cout << "value targets : search value vs game result correlation " << corr
								<< ", mean |q - z| " << mae << " over " << n << " positions (valueQMix " << globalConfig.valueQMix << ")" << std::endl;
						}
						std::cout << "average score difference : " << (float)totalScoreDiff / gamePlayed << std::endl;
						std::cout << "average game length : " << (float)totalGameLength / gamePlayed << std::endl;
						if(!trainLosses[0].empty()){
							const auto mean = [](const std::vector<float>& v){ return std::accumulate(v.begin(), v.end(), 0.0f) / v.size(); };
							std::cout << "mean policy loss : " << mean(trainLosses[0]) << " policy target entropy : " << mean(trainLosses[5])
								<< " policy KL : " << mean(trainLosses[0]) - mean(trainLosses[5]) << " value loss : " << mean(trainLosses[1])
								<< " value MSE vs z : " << mean(trainLosses[6]) << std::endl;
						}
						std::cout << "train losses : " << std::endl;
						for(int i=0; i<9; ++i){
							for(float l : trainLosses[i])
								std::cout << l << " ";
							std::cout << std::endl;
						}

						totalGameLength = 0;
						totalScoreDiff = 0;
						for(int i=0; i<4; ++i)
							wintypeCounter[i] = 0;
						for(int i=0; i<9; ++i)
							trainLosses[i].clear();
					}
					
					if((gamesPlayed + saveCnt) % globalConfig.checkFreq == 0){
						globalConfig = load_config("../configs/compare_config.json");

						float winrate = ModelCompare::policy_evaluate(modelName, currentBestModelFile, 
							std::cout, std::cout, true, true, 0.5f, globalConfig.compareGameCnt / 2, globalConfig.compareThreadNum);

						std::cout << "model " << modelName << " vs " << currentBestModelFile << 
						" winrate " << winrate << std::endl;

						if(winrate <= 0)
							std::cout << "rating differential : -INF\n";
						else if(winrate >= 1)
							std::cout << "rating differential : +INF\n";
						else
							std::cout << "rating differential : " << 400.0 * std::log10(winrate / (1.0 - winrate)) << "\n";

						if(winrate > 0.5f){
							std::cout << "Best model updated! " << currentBestModelFile << " to " << modelName << std::endl;
							currentBestModelFile = modelName;
							trainModel.save_model(globalConfig.modelPath + trainModel.model_file_name(std::to_string(gamesPlayed + saveCnt) + "B")); // best models are saved
						}
						else if(winrate < fallbackWinrate){
							std::cout << "model fallback!" << std::endl;
							trainModel.load_weights(globalConfig.modelPath + currentBestModelFile); // keeps the optimizer state
						}

						// no clear improvement for lrPatience comparisons in a row : halve the learning rate.
						if(winrate >= lrImprovementWinrate)
							comparisonsWithoutImprovement = 0;
						else if(++comparisonsWithoutImprovement >= lrPatience){
							learningRate *= 0.5f;
							comparisonsWithoutImprovement = 0;
							std::cout << "no improvement in " << lrPatience << " comparisons : learning rate halved to " << learningRate << std::endl;
						}
						std::cout << "comparisons without improvement : " << comparisonsWithoutImprovement << " / " << lrPatience
							<< ", learning rate " << learningRate << std::endl;

						trainModel.set_learning_rate(learningRate);

						// save model
						{
							modelName = trainModel.model_file_name(std::to_string(gamesPlayed + saveCnt));
							const std::string modelPath = globalConfig.modelPath + modelName;
							trainModel.save_model(modelPath);

							// save to google drive.
							if(globalConfig.googleDrive)
								trainModel.save_model(globalConfig.drivePath + modelName);

							// save training data & optimizer state.
							const std::string dataFile = globalConfig.dataPath + modelName;
							replayBuffer.save(dataFile);
							std::cout << "training data saved : " << dataFile << " (" << replayBuffer.size() << " positions)" << std::endl;

							// optimizer state filename.
							const std::string optPath = modelPath;
							trainModel.save_optimizer(optPath);
							std::cout << "optimizer state saved : " << optPath << std::endl;
						}
						
						globalConfig = load_config("../configs/train_config.json");
					}

					// critical section
					evaluator->update_model(&trainModel); // synchronize train_model and inference_model
					
					pauseFlag.store(false); // restart train thread
					trainCv.notify_one(); // notify train thread
				}
				
				saveMutex.unlock();
			}

			selfPlayPaused[j].store(true);
			pauseCv.notify_one();
		});
	}

    // Training thread
    std::thread trainThread([&] {
        while (true) {
            std::unique_lock<std::mutex> lock(trainMutex);
            trainCv.wait(lock, [&] { return stopFlag || startFlag || pauseFlag ^ trainPaused || replayBuffer.size() >= trainStartPoint; });

			if(stopFlag){
				break;
			}
			else if(pauseFlag ^ trainPaused){
				if(trainPaused)
					std::cerr << "train resumed" << std::endl;
				else
					std::cerr << "train paused" << std::endl;
				trainPaused.store(pauseFlag.load());
				pauseCv.notify_one();
			}
            else if (replayBuffer.size() >= trainStartPoint && !pauseFlag) {
				std::this_thread::sleep_for(std::chrono::milliseconds(globalConfig.trainWaitTime / inferenceThreadNum));
				trainIter++;
                train(); 
            }
        }
    });

	for(auto& th : selfPlayThreads){
		th.join();
	}
	stopFlag = true;
    trainCv.notify_one();
    trainThread.join();
	delete evaluator;
}

void TrainPipeline::pin_threads_to_core(std::thread& th, int coreId){
#ifdef __linux__
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(coreId, &cpuset);
	int rc = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
	if (rc != 0) {
		std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
	}
#endif
}

void TrainPipeline::display_train_data(const std::shared_ptr<const TrainData> data) const{
	// TrainData = std::tuple<NNInput, std::vector<float>, float, float, std::vector<float>, Trainhead>
	const auto& [state, policy, value, score, map, wintype, outcome] = *data;
	const auto& boardState = state;

	char display[rowSize][colSize];
    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
			int idx = i * colSize + j;
			if(boardState[idx] == 1.0f){
				display[i][j] = 'o';
			}
			else if(boardState[boardSize + idx] == 1.0f){
				display[i][j] = 'x';
			}
			else if(boardState[2 * boardSize + idx] == 1.0f){
				display[i][j] = '+';
			}
			else if(boardState[3 * boardSize + idx] == 1.0f){
				display[i][j] = 'm';
			}
			else if(boardState[4 * boardSize + idx] == 1.0f){
				display[i][j] = 'e';
			}
			else{
				display[i][j] = '-';
			}            
        }
    }

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            std::cout << display[i][j] << " ";
        }
        std::cout << "\n";
    }

	std::cout << "policy : " << std::endl;
	for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
			std::printf("%8.4f ", policy[i * colSize + j]);
        }
        std::cout << "\n";
    }

	std::cout << "value : " << value << std::endl;

	std::cout << "score : " << score << std::endl;

	switch (wintype) {
		case CAPTURE: std::cout << "type : CAPTURE" << std::endl; break;
		case SCORE:   std::cout << "type : SCORE" << std::endl; break;
		case NONE:    std::cout << "type : NONE" << std::endl; break;
		case RESIGN:  std::cout << "type : RESIGN" << std::endl; break;
		default:      std::cout << "type : ERROR" << std::endl; break;
	}

	if(wintype == CAPTURE || wintype == SCORE){
		std::cout << "map : " << std::endl;
		for(int i=0; i<rowSize; ++i){
			for(int j=0; j<colSize; ++j){
				std::printf("%8.4f ", map[i * colSize + j]);
			}
			std::cout << "\n";
		}
	}
}
void train_on_static_data(const std::string& dataFile, int steps, float lr, float validationFraction){
	size_t added = 0;
	std::vector<PackedSample> samples = ReplayBuffer::read_file(dataFile, added);
	const int B = globalConfig.batchSize;

	// positions are stored oldest first and a game's positions are consecutive, so taking the newest ones as
	// validation keeps (all but one) games entirely on one side. A random split would put positions of the same
	// game on both sides, and the value head could pass validation by remembering game outcomes.
	const size_t nVal = static_cast<size_t>(samples.size() * validationFraction);
	const size_t nTrain = samples.size() - nVal;
	if(nVal < static_cast<size_t>(B) || nTrain < static_cast<size_t>(B))
		throw std::runtime_error("not enough positions in " + dataFile + " for a batch of " + std::to_string(B) + " on each side");

	ReplayBuffer trainBuffer(nTrain, nTrain, 1.0f), valBuffer(nVal, nVal, 1.0f);
	for(size_t i = 0; i < samples.size(); ++i)
		(i < nTrain ? trainBuffer : valBuffer).add_packed(std::move(samples[i]));
	samples.clear();
	std::cout << dataFile << " : " << nTrain << " training / " << nVal << " validation positions, "
		<< steps << " steps, learning rate " << lr << std::endl;

	PolicyValueNet net("none", torch::cuda::is_available()); // untrained
	std::vector<float> state(B * globalConfig.inputChannel * inputSize), move(B * outputSize), result(B), outcome(B), score(B), map(B * boardSize);
	std::vector<Trainhead> type(B);

	using Losses = std::array<float, 7>; // policy, value, score, capture map, score map, target entropy, value MSE vs z
	const auto print = [](const char* name, const Losses& l){
		std::cout << " | " << name << " policy " << l[0] << " (entropy " << l[5] << ", KL " << l[0] - l[5] << ") value " << l[1]
			<< " (vs z " << l[6] << ") score " << l[2] << " score map " << l[4] << " capture map " << l[3];
	};

	const int reportEvery = 200, valBatches = 8;
	Losses trainSum{};
	for(int step = 1; step <= steps; ++step){
		trainBuffer.sample_batch(B, state, move, result, outcome, score, map, type);
		const auto [p, v, s, cm, sm, e, vz] = net.train(state, move, result, outcome, score, map, type, lr);
		const Losses l{p, v, s, cm, sm, e, vz};
		for(int k = 0; k < 7; ++k)
			trainSum[k] += l[k];

		if(step % reportEvery == 0 || step == steps){
			const int n = (step % reportEvery == 0) ? reportEvery : step % reportEvery;
			Losses trainMean, valMean{};
			for(int k = 0; k < 7; ++k)
				trainMean[k] = trainSum[k] / n;
			for(int b = 0; b < valBatches; ++b){
				valBuffer.sample_batch(B, state, move, result, outcome, score, map, type);
				const auto [vp, vv, vs, vcm, vsm, ve, vvz] = net.train(state, move, result, outcome, score, map, type, lr, /*update=*/false);
				const Losses vl{vp, vv, vs, vcm, vsm, ve, vvz};
				for(int k = 0; k < 7; ++k)
					valMean[k] += vl[k] / valBatches;
			}
			std::cout << "step " << step;
			print("train", trainMean);
			print("validation", valMean);
			std::cout << std::endl;
			trainSum = {};
		}
	}

	const std::string modelName = globalConfig.modelPath + net.model_file_name("static");
	net.save_model(modelName);
	std::cout << "model saved : " << modelName << std::endl;
}
