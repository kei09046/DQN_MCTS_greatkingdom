#include "neuralNet.h"
#include "consts.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

// For 9*9 board.
Net::Net(int channelSize, int blockSize): channelSize(channelSize), cv1(torch::nn::Conv2dOptions(channelSize, 128, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(128)),

// Policy head
at_cv3(torch::nn::Conv2dOptions(128, 32, 1).bias(false)),
at_bn3(torch::nn::BatchNorm2d(32)),
at_cv4(torch::nn::Conv2dOptions(32, 2, 1).bias(false)),
at_bn4(torch::nn::BatchNorm2d(2)),
at_fc1(2 * inputSize, outputSize),

// Value head
v_cv3(torch::nn::Conv2dOptions(128, 32, 1).bias(false)),
v_bn3(torch::nn::BatchNorm2d(32)),
v_cv4(torch::nn::Conv2dOptions(32, 2, 1).bias(false)),
v_bn4(torch::nn::BatchNorm2d(2)),
v_fc1(2 * inputSize, 1),
//v_fc2(256, 1),

// Score diff head
sc_cv3(torch::nn::Conv2dOptions(128, 32, 1).bias(false)),
sc_bn3(torch::nn::BatchNorm2d(32)),
sc_cv4(torch::nn::Conv2dOptions(32, 2, 1).bias(false)),
sc_bn4(torch::nn::BatchNorm2d(2)),
sc_fc1(2 * inputSize, 1),
//sc_fc2(256, 1),

// score map head
sc_map_cv3(torch::nn::Conv2dOptions(128, 32, 1).bias(false)),
sc_map_bn3(torch::nn::BatchNorm2d(32)),
sc_map_cv4(torch::nn::Conv2dOptions(32, 1, 1).bias(false)),

// capture head
cap_cv3(torch::nn::Conv2dOptions(128, 32, 1).bias(false)),
cap_bn3(torch::nn::BatchNorm2d(32)),
cap_cv4(torch::nn::Conv2dOptions(32, 1, 1).bias(false))

{
	blocks = register_module("blocks", torch::nn::ModuleList());

	for (int i = 0; i < blockSize; i++) {
		blocks->push_back(ResidualBlock(128));
	}
	
	register_module("cv1", cv1);
	register_module("bn1", bn1);

	register_module("at_cv3", at_cv3);
	register_module("at_bn3", at_bn3);
	register_module("at_cv4", at_cv4);
	register_module("at_bn4", at_bn4);
	register_module("at_fc1", at_fc1);

	register_module("v_cv3", v_cv3);
	register_module("v_bn3", v_bn3);
	register_module("v_cv4", v_cv4);
	register_module("v_bn4", v_bn4);
	register_module("v_fc1", v_fc1);
	// register_module("v_fc2", v_fc2);

	register_module("sc_cv3", sc_cv3);
	register_module("sc_bn3", sc_bn3);
	register_module("sc_cv4", sc_cv4);
	register_module("sc_bn4", sc_bn4);
	register_module("sc_fc1", sc_fc1);
	// register_module("sc_fc2", sc_fc2);

	register_module("sc_map_cv3", sc_map_cv3);
	register_module("sc_map_bn3", sc_map_bn3);
	register_module("sc_map_cv4", sc_map_cv4);

	register_module("cap_cv3", cap_cv3);
	register_module("cap_bn3", cap_bn3);
	register_module("cap_cv4", cap_cv4);
}

NNOutput Net::forward(const torch::Tensor& state)
{
	torch::Tensor x = torch::nn::functional::relu(bn1(cv1(state)));
	for (auto& block : *blocks) {
		x = block->as<ResidualBlock>()->forward(x);
	}

	// policy head
	torch::Tensor log_act = torch::nn::functional::relu(at_bn3(at_cv3(x)));
	log_act = at_bn4(at_cv4(log_act));
	log_act = log_act.view({ -1, 2 * inputSize });
	//log_act = torch::nn::functional::relu(at_fc1(log_act));
	log_act = at_fc1(log_act);

	// value head
	torch::Tensor val = torch::nn::functional::relu(v_bn3(v_cv3(x)));
	val = v_bn4(v_cv4(val));
	val = val.view({-1, 2 * inputSize});
	//val = torch::nn::functional::relu(v_fc1(val));
	val = v_fc1(val).tanh();

	// score head
	torch::Tensor score = torch::nn::functional::relu(sc_bn3(sc_cv3(x)));
	score = sc_bn4(sc_cv4(score));
	score = score.view({-1, 2 * inputSize});
	//score = torch::nn::functional::relu(sc_fc1(score));
	torch::Tensor exp_score_diff = sc_fc1(score);

	// score map head
	torch::Tensor exp_score_map = torch::nn::functional::relu(sc_map_bn3(sc_map_cv3(x)));
	exp_score_map = sc_map_cv4(exp_score_map).tanh();
	
	// capture map head
	torch::Tensor cap_prob_map = torch::nn::functional::relu(cap_bn3(cap_cv3(x)));
	cap_prob_map = cap_cv4(cap_prob_map).sigmoid();

	//std::cerr << log_act << " " << val << " " << exp_score_diff << " " << exp_score_map << " " << cap_prob_map << std::endl;
	return std::make_tuple(log_act, val, exp_score_diff, exp_score_map, cap_prob_map);
}

NNInput PolicyValueNet::getData(const Game& game){
    std::vector<float> ret(inputSize * globalConfig.inputChannel, 0.0f);
	Color turn = game.getTurn();
	Color oppturn = Game::reverseColor(turn);
	Color state;

	for(int i=0; i<inputSize; ++i){ // channel 0, 1, 2 : indicates location of black/white/neutral stones
		state = game.getBoard({i / colSize, i % colSize});
		if(state == turn)
			ret.at(i) = 1.0f;
		else if(state == oppturn)
			ret.at(inputSize + i) = 1.0f;
		else if(state == NEUTRAL)
			ret.at(2 * inputSize + i) = 1.0f;
	}

	Color terr;
	Color turnScore = (turn == BLACK) ? BSCORE : WSCORE;
	Color oppturnScore = (turn == BLACK) ? WSCORE : BSCORE;

	for(int i=0; i<inputSize; ++i){ // channel 3, 4 : indicates territory
		terr = game.getScoreBoard({i/colSize, i%colSize});
		if(terr == turnScore){
			ret.at(3*inputSize + i) = 1.0f;
		}
		else if(terr == oppturnScore){
			ret.at(4*inputSize + i) = 1.0f;
		}
	}

	float diff = game.scoreDiff(turn);
	for(int i=0; i<inputSize; ++i){ // channel 5 : difference of current score
		ret.at(5*inputSize + i) = diff;
	}

	// channel 6, 7 : last move and second last move
	Move lastMove = game.getLastMove(0);
	if(lastMove != PASSMOVE && lastMove != RESIGNMOVE){
		ret.at(6*inputSize + lastMove.first * colSize + lastMove.second) = 1.0f;
	}
	Move secondLastMove = game.getLastMove(1);
	if(secondLastMove != PASSMOVE && secondLastMove != RESIGNMOVE){
		ret.at(7*inputSize + secondLastMove.first * colSize + secondLastMove.second) = 1.0f;
	}

	// channel 8 ~ 17 : liberty count(inf if adjacent to territory)
	std::bitset<inputSize> mark;

	for(int i=0; i<inputSize; ++i){
		int cidx = game.getChainIdx(i);
		const Chain c = game.getChain(i);

		if(c.size != 0 && !mark[cidx]){
			mark[cidx] = true;

			auto head = game.getStone({i / colSize, i % colSize}).head;
			auto cur = head;
			auto state = game.getBoard({i / colSize, i % colSize});
			int liberty_count = 0;

			for(int j=0; j<boardSize; ++j){
				// if one of the liberty is unplayable by rule(=completely alive)
				if(c.liberties.test(j) && !game.isLegal(j/colSize, j%colSize)){
					liberty_count = 5;
					break;
				}
			}

			// if not completely alive.
			if(liberty_count == 0){
				liberty_count = std::min((int)c.liberties.count(), 4);
			}

			if(state == turn){ // my stone's liberties
				do {
					ret.at((7 + liberty_count)*inputSize + cur) = 1.0f;
					cur = game.getStone({cur/colSize, cur%colSize}).next;
				} while (cur != head);
			}
			else if(state == oppturn){ // opponent stone's liberties
				do {
					ret.at((12 + liberty_count)*inputSize + cur) = 1.0f;
					cur = game.getStone({cur/colSize, cur%colSize}).next;
				} while (cur != head);
			}
		}
	}

	// channel 18 : opponent's threat
	// channel 19 : best move to avoid threat
	Move threat = game.getThreat();
	if(threat != RESIGNMOVE){
		ret.at(18 * inputSize + threat.first * colSize + threat.second) = 1.0f;
		ret.at(19 * inputSize + game.getAvailableMoves()[0]) = 1.0f;
	}

	// channel 20 : winning move
	Move winmove = game.getWin();
	if(winmove != RESIGNMOVE){
		ret.at(20 * inputSize + winmove.first * colSize + winmove.second) = 1.0f;
	}

	// channel 21 : available moves
	for(const auto& m : game.getAvailableMoves()){
		if(m < inputSize)
			ret.at(21 * inputSize + m) = 1.0f;
	}

    return {ret};
}

NNInput PolicyValueNet::getData(const std::vector<const Game*>& gameBatch){
	NNInput ret;
	ret.reserve(gameBatch.size() * inputSize * globalConfig.inputChannel);

	for(int b=0; b<gameBatch.size(); ++b){
		auto data = getData(*gameBatch[b]);
		// data : std::tuple<std::vector, std::vector, std::vector> with size inputSize * globalConfig.inputChannel, boardSize + 1, boardSize + 1
		ret.insert(ret.end(), data.begin(), data.end());
	}
    return ret;
}

PolicyValueNet::PolicyValueNet(const std::string& model_file, const std::string& model_type, bool use_gpu):
 use_gpu(use_gpu), device(use_gpu ? torch::kCUDA : torch::kCPU), model_type(model_type)
{
	load_model(model_file);
}

PolicyValueNet::PolicyValueNet(const std::string& model_file, bool use_gpu): 
PolicyValueNet(model_file, globalConfig.modelPrefix, use_gpu)
{}

std::vector<PolicyValueOutput>
PolicyValueNet::batchEvaluate(const std::vector<const Game*>& gameBatch){
	using namespace torch::indexing;

	const int B = gameBatch.size();
	std::vector<PolicyValueOutput> outputs;
	outputs.reserve(B);

	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto gameData = getData(gameBatch);

	torch::Tensor inputBatch, policyBatch, valueBatch, scoreBatch, scoreMapBatch, captureMapBatch;

	inputBatch = torch::from_blob(gameData.data(), // [B, 18, 9, 9]
		{B, globalConfig.inputChannel, rowSize, colSize},
		options).to(device);

	torch::Tensor stones = inputBatch.index({Slice(), Slice(8, 12), Slice(), Slice()}).sum(1) + inputBatch.index({Slice(), Slice(13, 17), Slice(), Slice()}).sum(1);
	// ---- Forward pass ----
	torch::NoGradGuard no_grad;
	if(use_gpu){
		auto r = policy_value_net->forward(inputBatch);
		policyBatch = std::get<0>(r).to(torch::kCPU);  // [B, outputSize]
		valueBatch  = std::get<1>(r).to(torch::kCPU);  // [B, 1]
		scoreBatch = std::get<2>(r).to(torch::kCPU); // [B, 1]
		scoreMapBatch = std::get<3>(r).to(torch::kCPU); // [B, 1, 9, 9]
		captureMapBatch = (std::get<4>(r) * stones).to(torch::kCPU); // [B, 1, 9, 9]
	}
	else{
		auto r = policy_value_net->forward(inputBatch);
		policyBatch = std::get<0>(r);   // already CPU
		valueBatch  = std::get<1>(r);
		scoreBatch = std::get<2>(r);
		scoreMapBatch = std::get<3>(r);
		captureMapBatch = std::get<4>(r) * stones;
	}

	//std::cerr << policyBatch.dtype() << " " << policyBatch.device() << " " << policyBatch.sizes() << std::endl;
	// ---- Extract each result ----
	float* pP = policyBatch.data_ptr<float>();
	float* pV = valueBatch.data_ptr<float>();
	float* pS = scoreBatch.data_ptr<float>();
	float* pSm = scoreMapBatch.data_ptr<float>();
	float* pCm = captureMapBatch.data_ptr<float>();
	// float* pCc = capChanceBatch.data_ptr<float>();

	for(int b = 0; b < B; ++b) {
		// policy head: copy whole row
		float* src = pP + b * outputSize;
		std::vector<float> policy(src, src + outputSize);

		float value = pV[b];
		float s = pS[b];

		src = pSm + b * boardSize;
		std::vector<float> scMap(src, src + boardSize);

		src = pCm + b * boardSize;
		std::vector<float> capMap(src, src + boardSize);

		// float capProb = pCc[b];

		outputs.emplace_back(std::move(policy), value, s, std::move(scMap), std::move(capMap));
	}
	return outputs;
}

// std::vector<PolicyValueOutput>
// PolicyValueNet::backupEvaluate(const std::vector<const Game*>& gameBatch){
// 	const int B = gameBatch.size();
// 	std::vector<PolicyValueOutput> outputs;
// 	outputs.reserve(B);

// 	auto options = torch::TensorOptions().dtype(torch::kFloat32);
// 	std::vector<float> batchData = getData(gameBatch);

// 	assert(batchData.size() == B * globalConfig.inputChannel * rowSize * colSize);

// 	int cnt = 0;
// 	for(int game_id = 0; game_id < B; ++game_id){
// 		std::cerr << "game : " << game_id << std::endl;
// 		for(int channel = 0; channel < globalConfig.inputChannel; ++channel){
// 			std::cerr << "channel " << channel << std::endl;
// 			for(int row = 0; row < rowSize; ++row){
// 				for(int col = 0; col < colSize; ++col){
// 					std::cerr << batchData[cnt++] << " ";
// 				}
// 				std::cerr << std::endl;
// 			}
// 		}
// 		std::cerr << std::endl;
// 	}

// 	for(int b = 0; b < B; ++b) {
// 		std::vector<float> policy(0.0f, outputSize);
// 		std::vector<float> scMap(0.0f, boardSize);
// 		std::vector<float> capMap(0.0f, boardSize);
// 		outputs.emplace_back(std::move(policy), 0.0f, 0.0f, std::move(scMap), std::move(capMap));
// 	}
// 	return outputs;
// }

PolicyValueOutput PolicyValueNet::evaluate(const Game& game){
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto gameData = getData(game);

	torch::Tensor input = torch::from_blob(gameData.data(), // [B, 18, 9, 9]
		{1, globalConfig.inputChannel, rowSize, colSize},
		options).to(device);

	NNOutput res;
	torch::NoGradGuard no_grad;
	if (use_gpu) {
		auto r = policy_value_net->forward(input);
		get<0>(res) = get<0>(r).to(torch::kCPU); // policy
		get<1>(res) = get<1>(r).to(torch::kCPU); // evaluation (scalar)
		get<2>(res) = get<2>(r).to(torch::kCPU); // score differential (scalar)
		get<3>(res) = get<3>(r).to(torch::kCPU); // score map
		get<4>(res) = get<4>(r).to(torch::kCPU); // capture map
	}
	else {
		res = policy_value_net->forward(input);
	}

	std::vector<float> policy;
	float* pt = get<0>(res).data_ptr<float>();
	for (int i=0; i<outputSize; ++i) {
		policy.push_back(pt[i]);
	}

	std::vector<float> winprob;
	pt = get<1>(res).data_ptr<float>();
	winprob.push_back(pt[0]);
	

	std::vector<float> scoremap;
	scoremap.reserve(boardSize);
	pt = get<3>(res).data_ptr<float>();
	for(int i=0; i<boardSize; ++i){
		scoremap.push_back(pt[i]);
	}

	std::vector<float> capturemap;
	capturemap.reserve(boardSize);
	pt = get<4>(res).data_ptr<float>();
	for(int i=0; i<boardSize; ++i){
		capturemap.push_back(pt[i]);
	}

	return { policy, get<1>(res).index({0, 0}).item<float>(), get<2>(res).index({0, 0}).item<float>(), scoremap, capturemap };
}

// std::tuple<float, float, float> PolicyValueNet::trainCap(std::vector<float>& state_batch, std::vector<float>& nextmove_batch, 
// 	std::vector<float>& result_batch, std::vector<float>& capture_batch, float lr){
// 	// check first element of batch
// 	// std::cerr << "capture training" << std::endl;
// 	// for(int i=0; i<2; ++i){
// 	// 	for(int j=0; j<rowSize; ++j){
// 	// 		for(int k=0; k<colSize; ++k){
// 	// 			std::cerr << state_batch[i * boardSize + j * colSize + k];
// 	// 		}
// 	// 		std::cerr << std::endl;
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << nextmove_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// std::cerr << result_batch[0] << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << capture_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	int B = result_batch.size();
//     auto options = torch::TensorOptions().dtype(torch::kFloat32);

// 	auto sb = torch::from_blob(state_batch.data(),
//     {B, globalConfig.inputChannel, inputRow, inputCol},
//     options).to(device);

// 	auto mp = torch::from_blob(nextmove_batch.data(),
// 		{B, outputSize},
// 		options).to(device);

// 	auto wb = torch::from_blob(result_batch.data(),
// 		{B, 1},
// 		options).to(device);

// 	auto cb = torch::from_blob(capture_batch.data(),
// 		{B, 1, outputRow, outputCol},
// 		options).to(device);

// 	// auto cap_chanceb = torch::ones({B, 1}, options).to(device);


// 	float pLoss = 0.0f, vLoss = 0.0f, cLoss = 0.0f, cpLoss = 0.0f;
// 	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
// 	for(int i=0; i<globalConfig.epochs; ++i){
// 		optimizer->zero_grad();

// 		auto [r1, r2, r3, r4, r5] = policy_value_net->forward(sb); 

// 		torch::Tensor log_move_probs = torch::log_softmax(r1, 1);
// 		torch::Tensor policy_loss = -torch::mean(torch::sum(mp * log_move_probs, 1));

// 		torch::Tensor value_loss = torch::nn::functional::mse_loss(r2, wb);

// 		float pos_weight = 1.0f;
// 		torch::Tensor loss_tensor = -(cb * torch::log(r5 + 1e-9) * pos_weight + (1 - cb) * torch::log(1 - r5 + 1e-9));
// 		torch::Tensor capture_loss = loss_tensor.mean();

// 		// torch::Tensor cap_prob_loss = torch::nn::functional::mse_loss(r6, cap_chanceb);


// 		torch::Tensor loss = value_loss + policy_loss + capture_loss;
// 		pLoss += policy_loss.item().to<float>();
// 		vLoss += value_loss.item().to<float>();
// 		cLoss += capture_loss.item().to<float>();
// 		// cpLoss += cap_prob_loss.item().to<float>();
// 		// std::cout << "loss epoch " << i << " " << value_loss.item() << " " << policy_loss.item() << std::endl;
// 		loss.backward();
// 		torch::nn::utils::clip_grad_norm_(policy_value_net->parameters(), 1.0);
// 		optimizer->step();
// 	}

// 	return {pLoss / globalConfig.epochs, vLoss / globalConfig.epochs, cLoss / globalConfig.epochs};
// }

// std::tuple<float, float, float, float> PolicyValueNet::trainSc(std::vector<float>& state_batch, std::vector<float>& nextmove_batch, 
// 	std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& scoremap_batch, float lr) {
// 	// check first element of batch
// 	// std::cerr << "score training" << std::endl;
// 	// for(int i=0; i<2; ++i){
// 	// 	for(int j=0; j<rowSize; ++j){
// 	// 		for(int k=0; k<colSize; ++k){
// 	// 			std::cerr << state_batch[i * boardSize + j * colSize + k];
// 	// 		}
// 	// 		std::cerr << std::endl;
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << nextmove_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// std::cerr << result_batch[0] << " " << score_batch[0] << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << scoremap_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	int B = result_batch.size();
//     auto options = torch::TensorOptions().dtype(torch::kFloat32);

// 	auto sb = torch::from_blob(state_batch.data(),
//     {B, globalConfig.inputChannel, inputRow, inputCol},
//     options).to(device);

// 	auto mp = torch::from_blob(nextmove_batch.data(),
// 		{B, outputSize},
// 		options).to(device);

// 	auto wb = torch::from_blob(result_batch.data(),
// 		{B, 1},
// 		options).to(device);

// 	auto sd = torch::from_blob(score_batch.data(),
// 		{B, 1},
// 		options).to(device);

// 	auto smap = torch::from_blob(scoremap_batch.data(),
// 		{B, 1, outputRow, outputCol},
// 		options).to(device);

// 	// auto cap_chanceb = torch::zeros({B, 1}, options).to(device);


// 	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
// 	float pLoss = 0.0f, vLoss = 0.0f, sLoss = 0.0f, smLoss = 0.0f, cpLoss = 0.0f;
// 	torch::nn::HuberLoss huber_loss(torch::nn::HuberLossOptions().delta(5.0).reduction(torch::kMean));

// 	for(int i=0; i<globalConfig.epochs; ++i){
// 		optimizer->zero_grad();

// 		auto [r1, r2, r3, r4, r5] = policy_value_net->forward(sb); 

// 		torch::Tensor log_move_probs = torch::log_softmax(r1, 1);
// 		torch::Tensor policy_loss = -torch::mean(torch::sum(mp * log_move_probs, 1));

// 		torch::Tensor value_loss = torch::nn::functional::mse_loss(r2, wb);

// 		torch::Tensor score_loss = huber_loss(r3, sd);

// 		torch::Tensor score_map_loss = torch::nn::functional::mse_loss(r4, smap);

// 		// torch::Tensor cap_prob_loss = torch::nn::functional::mse_loss(r6, cap_chanceb);

// 		torch::Tensor loss = value_loss + policy_loss + score_loss + score_map_loss;
// 		pLoss += policy_loss.item().to<float>();
// 		vLoss += value_loss.item().to<float>();
// 		sLoss += score_loss.item().to<float>();
// 		smLoss += score_map_loss.item().to<float>();
// 		// cpLoss += cap_prob_loss.item().to<float>();
// 		loss.backward();
// 		torch::nn::utils::clip_grad_norm_(policy_value_net->parameters(), 1.0);
// 		optimizer->step();
// 	}

// 	return {pLoss / globalConfig.epochs, vLoss / globalConfig.epochs, sLoss / globalConfig.epochs, smLoss / globalConfig.epochs};
// }

std::tuple<float, float, float, float, float> PolicyValueNet::train(std::vector<float>& state_batch,
	 std::vector<float>& nextmove_batch, std::vector<float>& result_batch, std::vector<float>& score_batch,
	  std::vector<float>& map_batch, std::vector<Trainhead>& type_batch, float lr){

	int B = result_batch.size();
	auto options = torch::TensorOptions().dtype(torch::kFloat32);

	auto sb = torch::from_blob(state_batch.data(),
		{B, globalConfig.inputChannel, inputRow, inputCol},
		options).to(device);

	auto mp = torch::from_blob(nextmove_batch.data(),
		{B, outputSize},
		options).to(device);

	auto wb = torch::from_blob(result_batch.data(),
		{B, 1},
		options).to(device);

	auto sd = torch::from_blob(score_batch.data(),
		{B, 1},
		options).to(device);

	auto map = torch::from_blob(map_batch.data(),
		{B, 1, outputRow, outputCol},
		options).to(device);

	auto mask = torch::from_blob(type_batch.data(), {B, 1}, torch::kUInt8).to(device);

	auto pmask = (mask & POLICYHEAD).to(torch::kBool);
	auto smask = (mask & SCOREHEAD).to(torch::kBool);
	auto sm_mask = (mask & SMAPHEAD).to(torch::kBool).unsqueeze(-1).unsqueeze(-1);
	auto cap_mask = (mask & CMAPHEAD).to(torch::kBool).unsqueeze(-1).unsqueeze(-1);

	auto policy_active_elements = pmask.sum().item<float>() + 1e-8f;
	auto score_active_elements = smask.sum().item<float>() + 1e-8f;
	auto smap_active_elements = sm_mask.sum().item<float>() * outputRow * outputCol + 1e-8f;
	auto cap_active_elements = cap_mask.sum().item<float>() * outputRow * outputCol + 1e-8f;

	sd *= smask;
	auto smap = map * sm_mask;
	auto cmap = map * cap_mask;

	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
	torch::nn::HuberLoss huber_loss(torch::nn::HuberLossOptions().delta(5.0).reduction(torch::kSum));
	float pLoss = 0.0f, vLoss = 0.0f, sLoss = 0.0f, smLoss = 0.0f, cmLoss = 0.0f;

	//static int cntr = 0;
	for(int i=0; i<globalConfig.epochs; ++i){
		optimizer->zero_grad();

		auto [r1, r2, r3, r4, r5] = policy_value_net->forward(sb);
		// if(cntr % 1 == 0){
		// 	std::cout << "iter " << i << "\n";
		// 	for(int j=0; j<B; ++j){
		// 		if(type_batch[j] != NONE){
		// 			displayNNOutput({r1[j], r2[j], r3[j], r4[j], r5[j]});
		// 			break;
		// 		}
		// 	}
		// } 

		torch::Tensor log_move_probs = torch::log_softmax(r1, 1);
		torch::Tensor policy_loss_per_sample = -torch::sum(mp * log_move_probs, 1, /*keepdim=*/true); // [B, 1]
		torch::Tensor policy_loss = (policy_loss_per_sample * pmask.to(torch::kFloat32)).sum() / policy_active_elements;

		torch::Tensor value_loss = torch::nn::functional::mse_loss(r2, wb);

		auto masked_r3 = r3 * smask;
		torch::Tensor score_loss = huber_loss(masked_r3, sd);
		score_loss = score_loss / score_active_elements;

		auto masked_r4 = r4 * sm_mask;
		torch::Tensor score_map_loss = torch::nn::functional::mse_loss(masked_r4, smap, torch::nn::MSELossOptions().reduction(torch::kSum));
		score_map_loss = score_map_loss / smap_active_elements;

		float pos_weight = 1.0f;
		auto masked_r5 = r5 * cap_mask;
		torch::Tensor loss_tensor = -(cmap * torch::log(masked_r5 + 1e-9) * pos_weight + (1 - cmap) * torch::log(1 - masked_r5 + 1e-9));
		
		// Mask the loss tensor explicitly to zero out non-capture samples
		loss_tensor = loss_tensor * cap_mask; 
		torch::Tensor capture_loss = loss_tensor.sum() / cap_active_elements;
		/////////////////////////

		// TODO : find best ratio
		//torch::Tensor loss = policy_loss + value_loss*0.5f + score_loss*0.1f + score_map_loss*0.1f + capture_loss*0.1f;
		torch::Tensor loss = policy_loss + value_loss + score_loss + score_map_loss + capture_loss;

		pLoss += policy_loss.item<float>();
		vLoss += value_loss.item<float>();
		sLoss += score_loss.item<float>();
		smLoss += score_map_loss.item<float>();
		cmLoss += capture_loss.item<float>();

		loss.backward();
		torch::nn::utils::clip_grad_norm_(policy_value_net->parameters(), 1.0);
		optimizer->step();
	}
	//cntr++;

	return {pLoss / globalConfig.epochs, vLoss / globalConfig.epochs, sLoss / globalConfig.epochs, cmLoss / globalConfig.epochs, smLoss / globalConfig.epochs};
}

void PolicyValueNet::save_model(const std::string& model_file) const
{
	if(model_type == "A" || model_type == "B" || model_type == "C" || model_type == "E" || model_type == "F" || model_type == "G" || model_type == "H"
	|| model_type == "I"){
		auto net = std::dynamic_pointer_cast<Net>(policy_value_net);
		if(!net){
			throw std::runtime_error("Model type mismatch when saving: " + model_file);
		}
		torch::save(net, model_file);
	}
	else{
		throw std::runtime_error("Unknown model type when saving: " + model_type);
	}
}

void PolicyValueNet::load_model(const std::string& model_file){
	if (model_file.ends_with(".pt")) {
		std::shared_ptr<NetBase> net;

		if(model_type == "A"){
			net = std::make_shared<Net>(7, 12);
		}
		else if(model_type == "B"){
			net = std::make_shared<Net>(9, 12);
		}
		else if(model_type == "C"){
			net = std::make_shared<Net>(18, 12);
		}
		else if((model_type == "E") || (model_type == "F")){
			net = std::make_shared<Net>(18, 15);
		}
		else if((model_type == "G")){
			net = std::make_shared<Net>(18, 18);
		}
		else if(model_type == "H"){
			net = std::make_shared<Net>(18, 20);
		}
		else if(model_type == "I"){
			net = std::make_shared<Net>(22, 20);
		}
		else{
			throw std::runtime_error("Unknown model type: " + model_type);
		}
		torch::load(net, model_file);   
		policy_value_net = std::move(net);
	}   
	else{ // load default model to begin with.
		policy_value_net = std::make_shared<Net>(22, 20);
	}

	policy_value_net->to(device);
	torch::optim::AdamOptions opts(l2_const);
	optimizer = std::make_unique<torch::optim::Adam>(policy_value_net->parameters(), opts);
	std::cout << "Model loaded: " << model_file << std::endl;
}


// std::vector<float> PolicyValueNet::makeScoreDistributionBatch(
//     const std::vector<float>& scores,
//     float scoreRange,
//     float sigma,
//     int window) const
// {
//     int bins = 2 * scoreRange + 1;

//     auto dist = std::vector<float>(scores.size() * bins, 0.0f);

//     for (int i = 0; i < scores.size(); i++)
//     {
//         float s = std::clamp(scores[i], -scoreRange, scoreRange);

//         int start = std::max(-scoreRange, std::floor(s - window));
//         int end   = std::min(scoreRange,  std::ceil(s + window));
// 		float sum = 0.0f;

//         for (int b = start; b <= end; b++)
//         {
//             float diff = b - s;
//             float val = std::exp(-(diff * diff) / (2 * sigma * sigma));

//             dist[i * bins + b + scoreRange] = val;
// 			sum += val;
//         }

// 		for (int b = start; b <= end; b++)
//         {
//             dist[i * bins + b + scoreRange] /= sum;
//         }
//     }

//     return dist;
// }


void PolicyValueNet::displayNNOutput(const NNOutput& modelOut){
	// policyLogit : [82] winP : [1] scoreDiff : [1] scoreMap : [81] captureMap : [81]
 	const auto [policyLogit, winP, scoreDiff, scoreMap, captureMap] = modelOut;

	const torch::Tensor policy = torch::softmax(policyLogit, 0);
	
	auto to_printable = [](const torch::Tensor& t) {
		return t.detach().to(torch::kCPU);
	};

	std::cout << "WinP : " << to_printable(winP).item<float>() << "\n";
	std::cout << "Predicted Score Diff: " << to_printable(scoreDiff).item<float>() << "\n\n";

	// 2. Print Policy (Shape [82] -> 9x9 board positions + 1 pass move)
	std::cout << "Policy:\n";
	torch::Tensor board_policy = to_printable(policy).slice(0, 0, 81).view({9, 9});
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			// Format to 4 decimal places for clean grid alignment
			std::printf("%.4f ", board_policy[i][j].item<float>());
		}
		std::cout << "\n";
	}
	std::cout << to_printable(policy)[81].item<float>() << "\n\n";

	// 3. Print Spatial Feature Maps (scoreMap and captureMap reshaped to 9x9)
	torch::Tensor scoreGrid = to_printable(scoreMap).view({9, 9});
	std::cout << "Score Map:\n";
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			// Format to 4 decimal places for clean grid alignment
			std::printf("%8.4f ", scoreGrid[i][j].item<float>());
		}
		std::cout << "\n";
	}

	torch::Tensor captureGrid = to_printable(captureMap).view({9, 9});
	std::cout << "Capture Map:\n";
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			std::printf("%.4f ", captureGrid[i][j].item<float>());
		}
		std::cout << "\n";
	}

	
	std::cout << "=====================================\n";
}
