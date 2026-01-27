#include "neuralNet.h"
#include <cmath>
#include <iostream>

// For 9*9 board.
Net::Net(int channelSize): channelSize(channelSize), cv1(torch::nn::Conv2dOptions(channelSize, 128, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(128)),

// Policy head
at_cv3(torch::nn::Conv2dOptions(128, 2, 1).bias(false)),
at_bn3(torch::nn::BatchNorm2d(2)),
at_fc1(2 * inputSize, outputSize),

// Value head
v_cv3(torch::nn::Conv2dOptions(128, 1, 1).bias(false)),
v_bn3(torch::nn::BatchNorm2d(1)),
v_fc1(inputSize, 256),
v_fc2(256, 1){
	for (int i = 1; i < 13; i++) {
        auto rb = ResidualBlock(128);
		register_module("rb" + std::to_string(i) + "_conv1", rb->conv1);
		register_module("rb" + std::to_string(i) + "_bn1",   rb->bn1);
		register_module("rb" + std::to_string(i) + "_conv2", rb->conv2);
		register_module("rb" + std::to_string(i) + "_bn2",   rb->bn2);
		blocks.push_back(rb);
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
}

std::tuple<torch::Tensor, torch::Tensor> Net::forward(const torch::Tensor& state)
{
	torch::Tensor x = torch::nn::functional::relu(bn1(cv1(state)));
	for (auto& rb : blocks) {
		x = rb->forward(x);
	}
	torch::Tensor log_act = torch::nn::functional::relu(at_bn3(at_cv3(x)));
	log_act = log_act.view({ -1, 2 * inputSize });
	log_act = at_fc1(log_act);

	torch::Tensor val = torch::nn::functional::relu(v_bn3(v_cv3(x)));
	val = val.view({-1, inputSize});
	val = torch::nn::functional::relu(v_fc1(val));
	val = v_fc2(val);
	val = torch::tanh(val);
	return std::make_tuple(log_act, val);
}

InputMatrix PolicyValueNet::getData(const Game& game){
    InputMatrix ret(inputSize * globalConfig.inputChannel, 0.0f);
	color turn = game.getTurn();
	color opp_turn = Game::reverseColor(turn);
	color state;

	for(size_t i=0; i<inputSize; ++i){ // channel 0, 1, 2 : indicates location of black/white/neutral stones
		state = game.getBoard(i / colSize, i % colSize);
		if(state == turn)
			ret[i] = 1.0f;
		else if(state == opp_turn)
			ret[inputSize + i] = 1.0f;
		else if(state == NEUTRAL)
			ret[2 * inputSize + i] = 1.0f;
	}

	// for(size_t i = 3*inputSize; i < 4*inputSize; ++i){ // channel 3 : indicates turn
	// 	ret[i] = (turn == turn) ? 0.0f : 1.0f;
	// }

	color terr;
	for(size_t i=0; i<inputSize; ++i){ // channel 4, 5 : indicates territory
		terr = game.getScoreBoard(i/colSize, i%colSize);
		if(terr == turn){
			ret[3*inputSize + i] = 1.0f;
		}
		else if(terr == opp_turn){
			ret[4*inputSize + i] = 1.0f;
		}
	}

	float diff = game.scoreDiff(turn) / boardSize;
	for(size_t i=0; i<inputSize; ++i){ // channel 6 : difference of score
		ret[5*inputSize + i] = diff;
	}

	// channel 7, 8 : last move and second last move
	Move lastMove = game.getLastMove(0);
	if(lastMove != PASSMOVE && lastMove != RESIGNMOVE){
		ret[6*inputSize + lastMove.first * colSize + lastMove.second] = 1.0f;
	}
	Move secondLastMove = game.getLastMove(1);
	if(secondLastMove != PASSMOVE && secondLastMove != RESIGNMOVE){
		ret[7*inputSize + secondLastMove.first * colSize + secondLastMove.second] = 1.0f;
	}

	// channel 9, 10 : liberty count(inf if adjacent to territory)
	for(size_t i=0; i<inputSize; ++i){
		const Chain c = game.getChain(i);

		if(c.size != 0 && ret[9*inputSize + i] == 0 && ret[10*inputSize + i] == 0){
			auto cur = c.head;
			auto state = game.getBoard(i / colSize, i % colSize);
			int liberty_count = 0;

			for(size_t j=0; j<boardSize; ++j){
				// if one of the liberty is my territory(= completely alive group)
				// set the liberty count to 5. Note that my stone can't be adjacent to enemy territory.
				if(c.liberties.test(j) && ((ret[4*inputSize + j] == 1.0f) || (ret[5*inputSize + j] == 1.0f))){
					liberty_count = 5;
					break;
				}
			}

			if(liberty_count == 0)
				liberty_count = std::min((int)c.liberties.count(), 4);

			if(state == turn){ // black stone's liberties
				do {
					ret[8*inputSize + cur] = liberty_count;
					cur = game.getStone(cur/colSize, cur%colSize).next;
				} while (cur != c.head);
			}
			else if(state == opp_turn){ // white stone's liberties
				do {
					ret[9*inputSize + cur] = liberty_count;
					cur = game.getStone(cur/colSize, cur%colSize).next;
				} while (cur != c.head);
			}
		}
	}

    return ret;
}

std::vector<float> PolicyValueNet::getData(const std::vector<const Game*>& gameBatch){
	std::vector<float> ret;
	ret.reserve(gameBatch.size() * globalConfig.inputChannel * inputSize);

	for(size_t b=0; b<gameBatch.size(); ++b){
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
    const int B = gameBatch.size();
    std::vector<PolicyValueOutput> outputs;
    outputs.reserve(B);

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto batchData = getData(gameBatch);
    torch::Tensor batch = torch::from_blob(batchData.data(), {B, globalConfig.inputChannel, rowSize, colSize}, options).to(device);

    // ---- Forward pass ----
    torch::Tensor policyBatch, valueBatch;

    if(use_gpu){
        auto r = policy_value_net->forward(batch);
        policyBatch = std::get<0>(r).to(torch::kCPU);  // [B, outputSize]
        valueBatch  = std::get<1>(r).to(torch::kCPU);  // [B, 1]
    } else {
        auto r = policy_value_net->forward(batch);
        policyBatch = std::get<0>(r);   // already CPU
        valueBatch  = std::get<1>(r);
    }

    // ---- Extract each result ----
    float* pP = policyBatch.data_ptr<float>();
    float* pV = valueBatch.data_ptr<float>();

	for(int b = 0; b < B; ++b) {
		// policy head: copy whole row
		float* src = pP + b * outputSize;
		std::vector<float> pvfn(src, src + outputSize);

		float v = pV[b];
		outputs.push_back({std::move(pvfn), v});
	}
    return outputs;
}

PolicyValueOutput PolicyValueNet::evaluate(const Game& game){
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto data = getData(game);
	torch::Tensor current_state = torch::from_blob(data.data(), { 1, globalConfig.inputChannel, rowSize, colSize }, options).to(device);
	std::tuple<torch::Tensor, torch::Tensor> res;
	if (use_gpu) {
		auto r = policy_value_net->forward(current_state);
		get<0>(res) = get<0>(r).to(torch::kCPU);
		get<1>(res) = get<1>(r).to(torch::kCPU);
	}
	else {
		res = policy_value_net->forward(current_state);
	}

	std::vector<float> pvfn;
	float* pt = get<0>(res).data_ptr<float>();
	for (size_t i=0; i<outputSize; ++i) {
		pvfn.push_back(pt[i]);
	}

	return { pvfn, get<1>(res).index({0, 0}).item<float>() };
}

void PolicyValueNet::train_step(std::vector<float>& state_batch, 
    std::vector<float>& nextmove_batch, std::vector<float>& winner_batch, float lr) {

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sb = torch::from_blob(state_batch.data(), { globalConfig.batchSize, globalConfig.inputChannel, inputRow,
		 inputCol }, options).to(device);
    torch::Tensor mp = torch::from_blob(nextmove_batch.data(), { globalConfig.batchSize, outputSize }, options).to(device);
    torch::Tensor wb = torch::from_blob(winner_batch.data(), { globalConfig.batchSize }, options).to(device);

    optimizer->zero_grad();
    static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);

    torch::Tensor r1, r2;
    std::tie(r1, r2) = policy_value_net->forward(sb);

    // Ensure r1 contains logits and apply log_softmax before computing policy loss
    torch::Tensor policy_loss = -torch::mean(torch::sum(mp * torch::log_softmax(r1, 1), 1));

    // Ensure r2 and wb are correctly shaped for MSE loss
    torch::Tensor value_loss = torch::nn::functional::mse_loss(r2.view(-1), wb);

    torch::Tensor loss = value_loss + policy_loss;

    loss.backward();
    optimizer->step();
}

void PolicyValueNet::save_model(const std::string& model_file) const
{
	if(model_type == "A" || model_type == "B" || model_type == "C"){
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
			net = std::make_shared<Net>(7);
		}
		else if(model_type == "B"){
			net = std::make_shared<Net>(9);
		}
		else if(model_type == "C"){
			net = std::make_shared<Net>(10);
		}
		else{
			throw std::runtime_error("Unknown model type: " + model_type);
		}
		torch::load(net, model_file);   
		policy_value_net = std::move(net);
	}   
	else{ // load default model to begin with.
		policy_value_net = std::make_shared<Net>(10);
	}

	policy_value_net->to(device);
	torch::optim::AdamOptions opts(l2_const);
	optimizer = std::make_unique<torch::optim::Adam>(policy_value_net->parameters(), opts);
	std::cout << "Model loaded: " << model_file << std::endl;
}