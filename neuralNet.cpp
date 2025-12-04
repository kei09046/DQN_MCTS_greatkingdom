#include "neuralNet.h"
#include <cmath>
#include <iostream>
using namespace std;


// net : action probability 의 log와 value (-1, 1)을 추정한다.
GNet::GNet(): cv1(torch::nn::Conv2dOptions(inputChannel, 128, 3).padding(1).bias(false)),
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
	for (int i = 1; i < 7; i++) {
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

std::tuple<torch::Tensor, torch::Tensor> GNet::forward(const torch::Tensor& state)
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
	return make_tuple(log_act, val);
}

InputMatrix PolicyValueNet::getData(const Game& game){
    InputMatrix ret;
	color turn = game.getTurn();
	color state;

	ret.fill(0.0f);
	for(size_t i=0; i<inputSize; ++i){ // channel 0, 1, 2 : indicates location of black/white/neutral stones
		state = game.getBoard(i / colSize, i % colSize);
		if(state == turn)
			ret[i] = 1.0f;
		else if(state == Game::reverseColor(turn))
			ret[inputSize + i] = 1.0f;
		else if(state == NEUTRAL)
			ret[2 * inputSize + i] = 1.0f;
	}

	for(size_t i = 3*inputSize; i < 4*inputSize; ++i){ // channel 3 : indicates turn
		ret[i] = (turn == BLACK) ? 0.0f : 1.0f;
	}

	color terr;
	for(size_t i=0; i<inputSize; ++i){ // channel 4, 5 : indicates territory
		terr = game.getScoreBoard(i/colSize, i%colSize);
		if(terr == BLACK){
			ret[4*inputSize + i] = 1.0f;
		}
		else if(terr == WHITE){
			ret[5*inputSize + i] = 1.0f;
		}
	}
    return ret;
}

std::vector<float> PolicyValueNet::getData(const std::vector<const Game*>& gameBatch){
	std::vector<float> ret(gameBatch.size() * inputChannel * inputSize, 0.0f);

	const int temp = inputChannel * inputSize;
	for(int num = 0; num < gameBatch.size(); ++num){
		color turn = gameBatch[num]->getTurn();
		color state;

		for(int i=0; i<inputSize; ++i){
			state = gameBatch[num]->getBoard(i / colSize, i % colSize);
			if(state == turn)
				ret[temp * num + i] = 1.0f;
			else if(state == Game::reverseColor(turn))
				ret[temp * num + inputSize + i] = 1.0f;
			else if(state == NEUTRAL)
				ret[temp * num + 2 * inputSize + i] = 1.0f;
		}

		for(int i = 3*inputSize; i < 4*inputSize; ++i){
			ret[temp * num + i] = static_cast<float>(turn);
		}

		color terr;
		for(size_t i=0; i<inputSize; ++i){ // channel 4, 5 : indicates territory
			terr = gameBatch[num]->getScoreBoard(i/colSize, i%colSize);
			if(terr == BLACK){
				ret[temp * num + 4*inputSize + i] = 1.0f;
			}
			else if(terr == WHITE){
				ret[temp * num + 5*inputSize + i] = 1.0f;
			}
		}
	}

    return ret;
}

PolicyValueNet::PolicyValueNet(const string& model_file, const string& model_type, bool use_gpu):
 use_gpu(use_gpu), device(use_gpu ? torch::kCUDA : torch::kCPU), model_type(model_type)
{
	if (model_file.ends_with(".pt")) {
		std::shared_ptr<GNet> net;

		if(model_type == "g")
			net = std::make_shared<GNet>();
		else{
			throw std::runtime_error("Unknown model type: " + model_type);
		}
		torch::load(net, model_file);   
		policy_value_net = std::move(net);
	}   
	else{
		policy_value_net = std::make_shared<GNet>();
	}

	policy_value_net->to(device);
	torch::optim::AdamOptions opts(l2_const);
	optimizer = std::make_unique<torch::optim::Adam>(policy_value_net->parameters(), opts);
	std::cout << "Model loaded: " << model_file << std::endl;
}

PolicyValueNet::PolicyValueNet(const string& model_file, bool use_gpu): PolicyValueNet(model_file, string(1, model_file[5]), use_gpu)
{} // model file name like "modelg10000.pt"

std::vector<PolicyValueOutput>
PolicyValueNet::batchEvaluate(const std::vector<const Game*>& gameBatch){
    const int B = gameBatch.size();
    std::vector<PolicyValueOutput> outputs;
    outputs.reserve(B);

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto batchData = getData(gameBatch);
    torch::Tensor batch = torch::from_blob(batchData.data(), {B, inputChannel, rowSize, colSize}, options).to(device);

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
	torch::Tensor current_state = torch::from_blob(data.data(), { 1, inputChannel, rowSize, colSize }, options).to(device);
	tuple<torch::Tensor, torch::Tensor> res;
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

void PolicyValueNet::train_step(array<float, batchSize * inputChannel * inputSize>& state_batch, 
    array<float, batchSize * outputSize>& nextmove_batch, array<float, batchSize>& winner_batch, float lr) {

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sb = torch::from_blob(state_batch.data(), { batchSize, inputChannel, inputRow, inputCol }, options).to(device);
    torch::Tensor mp = torch::from_blob(nextmove_batch.data(), { batchSize, outputSize }, options).to(device);
    torch::Tensor wb = torch::from_blob(winner_batch.data(), { batchSize }, options).to(device);

    optimizer->zero_grad();
    static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);

    torch::Tensor r1, r2;
    tie(r1, r2) = policy_value_net->forward(sb);

    // Ensure r1 contains logits and apply log_softmax before computing policy loss
    torch::Tensor policy_loss = -torch::mean(torch::sum(mp * torch::log_softmax(r1, 1), 1));

    // Ensure r2 and wb are correctly shaped for MSE loss
    torch::Tensor value_loss = torch::nn::functional::mse_loss(r2.view(-1), wb);

    torch::Tensor loss = value_loss + policy_loss;

    loss.backward();
    optimizer->step();
}

void PolicyValueNet::save_model(const string& model_file) const
{
	if(model_type == "g"){
		auto net = std::dynamic_pointer_cast<GNet>(policy_value_net);
		if(!net){
			throw std::runtime_error("Model type mismatch when saving: " + model_file);
		}
		torch::save(net, model_file);
	}
	else{
		throw std::runtime_error("Unknown model type when saving: " + model_type);
	}
}
