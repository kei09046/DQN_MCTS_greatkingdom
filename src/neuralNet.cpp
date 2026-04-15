#include "neuralNet.h"
#include "consts.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

// For 9*9 board.
Net::Net(int channelSize, int blockSize): channelSize(channelSize), cv1(torch::nn::Conv2dOptions(channelSize, 128, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(128)),

// Policy head
at_cv3(torch::nn::Conv2dOptions(128, 2, 1).bias(false)),
at_bn3(torch::nn::BatchNorm2d(2)),
at_fc1(2 * inputSize, outputSize),

// Value head
v_cv3(torch::nn::Conv2dOptions(128, 1, 1).bias(false)),
v_bn3(torch::nn::BatchNorm2d(1)),
v_fc1(inputSize, 256),
v_fc2(256, 4),

// Score diff head
sc_cv3(torch::nn::Conv2dOptions(128, 1, 1).bias(false)),
sc_bn3(torch::nn::BatchNorm2d(1)),
sc_fc1(inputSize, 256),
sc_fc2(256, 1),
sc_fc_belief(256, 31) // score difference from -15 ~ +15
{
	blocks = register_module("blocks", torch::nn::ModuleList());

	for (int i = 0; i < blockSize; i++) {
		blocks->push_back(ResidualBlock(128));
	}
	
	register_module("cv1", cv1);
	register_module("bn1", bn1);

	register_module("at_cv3", at_cv3);
	register_module("at_bn3", at_bn3);
	register_module("at_fc1", at_fc1);
	register_module("v_cv3", v_cv3);
	register_module("v_bn3", v_bn3);
	register_module("v_fc1", v_fc1);
	register_module("v_fc2", v_fc2);
	register_module("sc_cv3", sc_cv3);
	register_module("sc_bn3", sc_bn3);
	register_module("sc_fc1", sc_fc1);
	register_module("sc_fc2", sc_fc2);
	register_module("sc_fc_belief", sc_fc_belief);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Net::forward(const torch::Tensor& state)
{
	torch::Tensor x = torch::nn::functional::relu(bn1(cv1(state)));
	for (auto& block : *blocks) {
		x = block->as<ResidualBlock>()->forward(x);
	}
	torch::Tensor log_act = torch::nn::functional::relu(at_bn3(at_cv3(x)));
	log_act = log_act.view({ -1, 2 * inputSize });
	log_act = at_fc1(log_act);

	torch::Tensor val = torch::nn::functional::relu(v_bn3(v_cv3(x)));
	val = val.view({-1, inputSize});
	val = torch::nn::functional::relu(v_fc1(val));
	val = v_fc2(val);

	torch::Tensor score = torch::nn::functional::relu(sc_bn3(sc_cv3(x)));
	score = score.view({-1, inputSize});
	score = torch::nn::functional::relu(sc_fc1(score));
	torch::Tensor score_raw = sc_fc2(score);
	torch::Tensor log_score_dist = sc_fc_belief(score); 

	return std::make_tuple(log_act, val, score_raw, log_score_dist);
}

std::vector<float> PolicyValueNet::getData(const Game& game){
    std::vector<float> ret(inputSize * globalConfig.inputChannel, 0.0f);
	Color turn = game.getTurn();
	Color opp_turn = Game::reverseColor(turn);
	Color state;

	for(int i=0; i<inputSize; ++i){ // channel 0, 1, 2 : indicates location of black/white/neutral stones
		state = game.getBoard(i / colSize, i % colSize);
		if(state == turn)
			ret.at(i) = 1.0f;
		else if(state == opp_turn)
			ret.at(inputSize + i) = 1.0f;
		else if(state == NEUTRAL)
			ret.at(2 * inputSize + i) = 1.0f;
	}

	Color terr;
	for(int i=0; i<inputSize; ++i){ // channel 3, 4 : indicates territory
		terr = game.getScoreBoard(i/colSize, i%colSize);
		if(terr == turn){
			ret.at(3*inputSize + i) = 1.0f;
		}
		else if(terr == opp_turn){
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
	for(int i=0; i<inputSize; ++i){
		const Chain c = game.getChain(i);

		if(c.size != 0 && ret[8*inputSize + i] == 0 && ret[9*inputSize + i] == 0){
			auto head = game.getStone( i / colSize, i % colSize).head;
			auto cur = head;
			auto state = game.getBoard(i / colSize, i % colSize);
			int liberty_count = 0;

			for(int j=0; j<boardSize; ++j){
				// if one of the liberty is my territory(= completely alive group)
				// set the liberty count to 5. Note that my stone can't be adjacent to enemy territory.
				if(c.liberties.test(j) && ((ret[3*inputSize + j] == 1.0f) || (ret[4*inputSize + j] == 1.0f))){
					liberty_count = 5;
					break;
				}
			}

			if(liberty_count == 0){
				liberty_count = std::min((int)c.liberties.count(), 4);
			}

			if(state == turn){ // my stone's liberties
				do {
					ret.at((7 + liberty_count)*inputSize + cur) = 1.0f;
					cur = game.getStone(cur/colSize, cur%colSize).next;
				} while (cur != head);
			}
			else if(state == opp_turn){ // opponent stone's liberties
				do {
					ret.at((12 + liberty_count)*inputSize + cur) = 1.0f;
					cur = game.getStone(cur/colSize, cur%colSize).next;
				} while (cur != head);
			}
		}
	}

    return ret;
}

std::vector<float> PolicyValueNet::getData(const std::vector<const Game*>& gameBatch){
	std::vector<float> ret;
	ret.reserve(gameBatch.size() * globalConfig.inputChannel * inputSize);

	for(int b=0; b<gameBatch.size(); ++b){
		auto data = getData(*gameBatch[b]);
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
	//std::cerr << "batchEvaluate called by thread " << std::this_thread::get_id() << std::endl;
	const int B = gameBatch.size();
	std::vector<PolicyValueOutput> outputs;
	outputs.reserve(B);

	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	std::vector<float> batchData = getData(gameBatch);

	torch::Tensor batch, policyBatch, valueBatch, scoreBatch, distBatch;

	batch = torch::from_blob(batchData.data(),
		{B, globalConfig.inputChannel, rowSize, colSize},
		options).to(device);

	//batch = torch::tensor(batchData, options).view({B, globalConfig.inputChannel, rowSize, colSize}).to(device);

	// ---- Forward pass ----
	torch::NoGradGuard no_grad;
	if(use_gpu){
		auto r = policy_value_net->forward(batch);
		policyBatch = std::get<0>(r).to(torch::kCPU);  // [B, outputSize]
		valueBatch  = std::get<1>(r).to(torch::kCPU);  // [B, 4]
		scoreBatch = std::get<2>(r).to(torch::kCPU);
		distBatch = std::get<3>(r).to(torch::kCPU); // [B, 31]
	} else {
		auto r = policy_value_net->forward(batch);
		policyBatch = std::get<0>(r);   // already CPU
		valueBatch  = std::get<1>(r);
		scoreBatch = std::get<2>(r);
		distBatch = std::get<3>(r);
	}

	//std::cerr << policyBatch.dtype() << " " << policyBatch.device() << " " << policyBatch.sizes() << std::endl;
	// ---- Extract each result ----
	float* pP = policyBatch.data_ptr<float>();
	float* pV = valueBatch.data_ptr<float>();
	float* pS = scoreBatch.data_ptr<float>();
	float* pSd = distBatch.data_ptr<float>();

	for(int b = 0; b < B; ++b) {
		// policy head: copy whole row
		float* src = pP + b * outputSize;
		std::vector<float> policy(src, src + outputSize);

		src = pV + b * 4;
		std::vector<float> value(src, src + 4);

		float s = pS[b];

		src = pSd + b * 31;
		std::vector<float> value_dist(src, src + 31);

		outputs.emplace_back(std::move(policy), std::move(value), s, std::move(value_dist));
	}
	return outputs;
}

std::vector<PolicyValueOutput>
PolicyValueNet::backupEvaluate(const std::vector<const Game*>& gameBatch){
	const int B = gameBatch.size();
	std::vector<PolicyValueOutput> outputs;
	outputs.reserve(B);

	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	std::vector<float> batchData = getData(gameBatch);

	assert(batchData.size() == B * globalConfig.inputChannel * rowSize * colSize);

	int cnt = 0;
	for(int game_id = 0; game_id < B; ++game_id){
		std::cerr << "game : " << game_id << std::endl;
		for(int channel = 0; channel < globalConfig.inputChannel; ++channel){
			std::cerr << "channel " << channel << std::endl;
			for(int row = 0; row < rowSize; ++row){
				for(int col = 0; col < colSize; ++col){
					std::cerr << batchData[cnt++] << " ";
				}
				std::cerr << std::endl;
			}
		}
		std::cerr << std::endl;
	}

	for(int b = 0; b < B; ++b) {
		std::vector<float> policy(0.0f, outputSize);
		std::vector<float> value(0.0f, 4);
		std::vector<float> value_dist(0.0f, 31);
		outputs.emplace_back(std::move(policy), std::move(value), 0.0f, std::move(value_dist));
	}
	return outputs;
}

PolicyValueOutput PolicyValueNet::evaluate(const Game& game){
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto data = getData(game);
	torch::Tensor current_state = torch::tensor(data, options).view({ 1, globalConfig.inputChannel, rowSize, colSize }).to(device);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> res;
	torch::NoGradGuard no_grad;
	if (use_gpu) {
		auto r = policy_value_net->forward(current_state);
		get<0>(res) = get<0>(r).to(torch::kCPU); // policy
		get<1>(res) = get<1>(r).to(torch::kCPU); // 4 dim vector [my_score_win, my_capture_win, opp_score_win, opp_capture_win]
		get<2>(res) = get<2>(r).to(torch::kCPU); // score differential 
		get<3>(res) = get<3>(r).to(torch::kCPU); // score distribution
	}
	else {
		res = policy_value_net->forward(current_state);
	}

	std::vector<float> policy;
	float* pt = get<0>(res).data_ptr<float>();
	for (int i=0; i<outputSize; ++i) {
		policy.push_back(pt[i]);
	}

	std::vector<float> winprob;
	pt = get<1>(res).data_ptr<float>();
	for (int i=0; i<4; ++i) {
		policy.push_back(pt[i]);
	}

	std::vector<float> scoredist;
	scoredist.reserve(31);
	pt = get<3>(res).data_ptr<float>();
	for(int i=0; i<31; ++i){
		scoredist.push_back(pt[i]);
	}

	return { policy, winprob, get<2>(res).index({0, 0}).item<float>(), scoredist };
}

void PolicyValueNet::train_step(std::vector<float>& state_batch, 
    std::vector<float>& nextmove_batch, std::vector<float>& result_batch, std::vector<float>& score_batch, float lr) {
	// for(const float& v : score_batch)
	// 	std::cerr << v << " ";
	// std::cerr << std::endl;
	// for(const float& v : result_batch)
	// 	std::cerr << v << " ";
	// std::cerr << std::endl;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);

	auto sb = torch::from_blob(state_batch.data(),
    {globalConfig.batchSize, globalConfig.inputChannel, inputRow, inputCol},
    options).to(device);

	auto mp = torch::from_blob(nextmove_batch.data(),
		{globalConfig.batchSize, outputSize},
		options).to(device);

	auto wb = torch::from_blob(result_batch.data(),
		{globalConfig.batchSize},
		options).to(device, torch::kLong);

	auto sd = torch::from_blob(score_batch.data(),
		{globalConfig.batchSize},
		options).to(device);

	auto scoreDist = makeScoreDistributionBatch(score_batch, 15.0f, 1.0f, 5);
	auto sdd = torch::from_blob(scoreDist.data(),
		{globalConfig.batchSize, 31},
		options).to(device);

	auto mask = (wb % 2 == 0);
	// auto mask = torch::where(
	// 	(wb % 2 == 0),
	// 	torch::ones_like(wb, torch::kFloat),          // for score win → weight 1
	// 	torch::full_like(wb, 0.01f, torch::kFloat)      // for capture win → weight 0.1
	// );

	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
	for(int i=0; i<globalConfig.epochs; ++i){
		optimizer->zero_grad();

		torch::Tensor r1, r2, r3, r4;
		std::tie(r1, r2, r3, r4) = policy_value_net->forward(sb); 

		torch::Tensor log_move_probs = torch::log_softmax(r1, 1);
		torch::Tensor policy_loss = -torch::mean(torch::sum(mp * log_move_probs, 1));

		torch::Tensor value_loss = torch::nn::functional::cross_entropy(r2, wb);

		torch::Tensor diff = (r3.view(-1) - sd);
		torch::Tensor masked_score = diff * diff * mask;   // mask: shape [B], float (0 or 1)
		torch::Tensor score_loss = masked_score.sum() / (mask.sum() + 0.000001f);

		torch::Tensor log_score_predict = torch::log_softmax(r4, 1);
		torch::Tensor per_sample = -torch::sum(sdd * log_score_predict, 1); // [B]
		torch::Tensor masked_score_dist = per_sample * mask;  // [B]
		torch::Tensor score_dist_loss = masked_score_dist.sum() / (mask.sum() + 0.000001f);


		torch::Tensor loss = value_loss + policy_loss + score_loss + score_dist_loss;
		// std::cout << "loss epoch " << i << " " << value_loss.item() << " " << policy_loss.item() << " " << score_loss.item() << " " << score_dist_loss.item() << std::endl;
		loss.backward();
		torch::nn::utils::clip_grad_norm_(policy_value_net->parameters(), 1.0);
		optimizer->step();
	}
}

void PolicyValueNet::save_model(const std::string& model_file) const
{
	if(model_type == "A" || model_type == "B" || model_type == "C" || model_type == "E"){
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
		else if(model_type == "E"){
			net = std::make_shared<Net>(18, 15);
		}
		else{
			throw std::runtime_error("Unknown model type: " + model_type);
		}
		torch::load(net, model_file);   
		policy_value_net = std::move(net);
	}   
	else{ // load default model to begin with.
		policy_value_net = std::make_shared<Net>(18, 15);
	}

	policy_value_net->to(device);
	torch::optim::AdamOptions opts(l2_const);
	optimizer = std::make_unique<torch::optim::Adam>(policy_value_net->parameters(), opts);
	std::cout << "Model loaded: " << model_file << std::endl;
}


std::vector<float> PolicyValueNet::makeScoreDistributionBatch(
    const std::vector<float>& scores,
    float scoreRange,
    float sigma,
    int window) const
{
    int bins = 2 * scoreRange + 1;

    auto dist = std::vector<float>(scores.size() * bins, 0.0f);

    for (int i = 0; i < scores.size(); i++)
    {
        float s = std::clamp(scores[i], -scoreRange, scoreRange);

        int start = std::max(-scoreRange, std::floor(s - window));
        int end   = std::min(scoreRange,  std::ceil(s + window));
		float sum = 0.0f;

        for (int b = start; b <= end; b++)
        {
            float diff = b - s;
            float val = std::exp(-(diff * diff) / (2 * sigma * sigma));

            dist[i * bins + b + scoreRange] = val;
			sum += val;
        }

		for (int b = start; b <= end; b++)
        {
            dist[i * bins + b + scoreRange] /= sum;
        }
    }

    return dist;
}