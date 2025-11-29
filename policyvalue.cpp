#include "policyvalue.h"
#include <cmath>
#include <iostream>
using namespace std;


// net : action probability 의 log와 value (-1, 1)을 추정한다.
NetImpl::NetImpl(bool use_gpu = false): cv1(torch::nn::Conv2dOptions(inputDepth, 128, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(128)),

// Residual block 1
rb1_conv1(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb1_bn1(torch::nn::BatchNorm2d(128)),
rb1_conv2(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb1_bn2(torch::nn::BatchNorm2d(128)),

// Residual block 2
rb2_conv1(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb2_bn1(torch::nn::BatchNorm2d(128)),
rb2_conv2(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb2_bn2(torch::nn::BatchNorm2d(128)),

// Residual block 3
rb3_conv1(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb3_bn1(torch::nn::BatchNorm2d(128)),
rb3_conv2(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb3_bn2(torch::nn::BatchNorm2d(128)),

// Residual block 4
rb4_conv1(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb4_bn1(torch::nn::BatchNorm2d(128)),
rb4_conv2(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb4_bn2(torch::nn::BatchNorm2d(128)),

// Residual block 5
rb5_conv1(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb5_bn1(torch::nn::BatchNorm2d(128)),
rb5_conv2(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb5_bn2(torch::nn::BatchNorm2d(128)),

// Residual block 6
rb6_conv1(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb6_bn1(torch::nn::BatchNorm2d(128)),
rb6_conv2(torch::nn::Conv2dOptions(128, 128, 3).padding(1).bias(false)),
rb6_bn2(torch::nn::BatchNorm2d(128)),

// Policy head (unchanged)
at_cv3(torch::nn::Conv2dOptions(128, 2, 1).bias(false)),
at_bn3(torch::nn::BatchNorm2d(2)),
at_fc1(2 * inputSize, outputSize),

// Value head (unchanged)
v_cv3(torch::nn::Conv2dOptions(128, 1, 1).bias(false)),
v_bn3(torch::nn::BatchNorm2d(1)),
v_fc1(inputSize, 256),
v_fc2(256, 1),
device(use_gpu ? torch::kCUDA : torch::kCPU){

	cv1->to(device);
	bn1->to(device);

	rb1_conv1->to(device);
	rb1_bn1->to(device);
	rb1_conv2->to(device);
	rb1_bn2->to(device);

	rb2_conv1->to(device);
	rb2_bn1->to(device);
	rb2_conv2->to(device);
	rb2_bn2->to(device);

	rb3_conv1->to(device);
	rb3_bn1->to(device);
	rb3_conv2->to(device);
	rb3_bn2->to(device);

	rb4_conv1->to(device);
	rb4_bn1->to(device);
	rb4_conv2->to(device);
	rb4_bn2->to(device);

	rb5_conv1->to(device);
	rb5_bn1->to(device);
	rb5_conv2->to(device);
	rb5_bn2->to(device);

	rb6_conv1->to(device);
	rb6_bn1->to(device);
	rb6_conv2->to(device);
	rb6_bn2->to(device);

	at_cv3->to(device);
	at_bn3->to(device);
	at_fc1->to(device);
	v_cv3->to(device);
	v_bn3->to(device);
	v_fc1->to(device);
	v_fc2->to(device);
	
	register_module("cv1", cv1);
	register_module("bn1", bn1);

	register_module("rb1_conv1", rb1_conv1);
	register_module("rb1_bn1", rb1_bn1);
	register_module("rb1_conv2", rb1_conv2);
	register_module("rb1_bn2", rb1_bn2);
	
	register_module("rb2_conv1", rb2_conv1);
	register_module("rb2_bn1", rb2_bn1);
	register_module("rb2_conv2", rb2_conv2);
	register_module("rb2_bn2", rb2_bn2);
	
	register_module("rb3_conv1", rb3_conv1);
	register_module("rb3_bn1", rb3_bn1);
	register_module("rb3_conv2", rb3_conv2);
	register_module("rb3_bn2", rb3_bn2);
	
	register_module("rb4_conv1", rb4_conv1);
	register_module("rb4_bn1", rb4_bn1);
	register_module("rb4_conv2", rb4_conv2);
	register_module("rb4_bn2", rb4_bn2);

	register_module("rb5_conv1", rb5_conv1);
	register_module("rb5_bn1", rb5_bn1);
	register_module("rb5_conv2", rb5_conv2);
	register_module("rb5_bn2", rb5_bn2);

	register_module("rb6_conv1", rb6_conv1);
	register_module("rb6_bn1", rb6_bn1);
	register_module("rb6_conv2", rb6_conv2);
	register_module("rb6_bn2", rb6_bn2);

	register_module("at_cv3", at_cv3);
	register_module("at_bn3", at_bn3);
	register_module("at_fc1", at_fc1);
	register_module("v_cv3", v_cv3);
	register_module("v_bn3", v_bn3);
	register_module("v_fc1", v_fc1);
	register_module("v_fc2", v_fc2);
}

std::tuple<torch::Tensor, torch::Tensor> NetImpl::forward(const torch::Tensor& state)
{
	torch::Tensor x = torch::nn::functional::relu(bn1(cv1(state)));
	x = torch::nn::functional::relu(rb1_bn1(rb1_conv1(x)));
	x = torch::nn::functional::relu(rb1_bn2(rb1_conv2(x)));

	x = torch::nn::functional::relu(rb2_bn1(rb2_conv1(x)));
	x = torch::nn::functional::relu(rb2_bn2(rb2_conv2(x)));

	x = torch::nn::functional::relu(rb3_bn1(rb3_conv1(x)));
	x = torch::nn::functional::relu(rb3_bn2(rb3_conv2(x)));

	x = torch::nn::functional::relu(rb4_bn1(rb4_conv1(x)));
	x = torch::nn::functional::relu(rb4_bn2(rb4_conv2(x)));

	x = torch::nn::functional::relu(rb5_bn1(rb5_conv1(x)));
	x = torch::nn::functional::relu(rb5_bn2(rb5_conv2(x)));

	x = torch::nn::functional::relu(rb6_bn1(rb6_conv1(x)));
	x = torch::nn::functional::relu(rb6_bn2(rb6_conv2(x)));

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

std::array<float, inputDepth * inputSize> PolicyValueNet::getData(const Game& game){
    std::array<float, inputDepth * inputSize> ret;
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
	std::vector<float> ret(gameBatch.size() * inputDepth * inputSize, 0.0f);

	const int temp = inputDepth * inputSize;
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
	}

    return ret;
}

PolicyValueNet::PolicyValueNet(const string& model_file, bool use_gpu): use_gpu(use_gpu), policy_value_net(use_gpu)
{
	if (model_file.ends_with(".pt")) {
		torch::load(policy_value_net, model_file);
		cout << "model_loaded" << endl;
	}

	optimizer = new torch::optim::Adam(policy_value_net->parameters(), l2_const);
	return;
}

std::vector<PolicyValueOutput>
PolicyValueNet::batchEvaluate(const std::vector<const Game*>& gameBatch){
    const int B = gameBatch.size();
    std::vector<PolicyValueOutput> outputs;
    outputs.reserve(B);

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
	auto batchData = getData(gameBatch);
    torch::Tensor batch = torch::from_blob(batchData.data(), {B, inputDepth, rowSize, colSize}, options).to(policy_value_net->device);

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

    // for(int b=0; b<B; ++b){
    //     // policy head
    //     std::vector<float> pvfn;
    //     pvfn.reserve(outputSize);
    //     float* src = pP + b * outputSize;
    //     for(int i=0; i<outputSize; ++i){
    //         pvfn.push_back(src[i]);
    //     }

    //     float v = pV[b];
    //     outputs.push_back({pvfn, v});
    // }

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
	torch::Tensor current_state = torch::from_blob(data.data(), { 1, inputDepth, rowSize, colSize }, options).to(policy_value_net->device);
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

// PolicyValueOutput PolicyValueNet::evaluate(const Game& game, const std::vector<std::pair<int, int> > legal)
// {
// 	auto options = torch::TensorOptions().dtype(torch::kFloat32);
// 	torch::Tensor current_state = torch::from_blob(getData(game).data(), { 1, inputDepth, rowSize, colSize }, options).to(policy_value_net->device);
// 	tuple<torch::Tensor, torch::Tensor> res;
// 	if (use_gpu) {
// 		auto r = policy_value_net->forward(current_state);
// 		get<0>(res) = get<0>(r).to(torch::kCPU);
// 		get<1>(res) = get<1>(r).to(torch::kCPU);
// 	}
// 	else {
// 		res = policy_value_net->forward(current_state);
// 	}

// 	std::vector<float> pvfn;
// 	float* pt = get<0>(res).data_ptr<float>();
// 	for (auto [r, c] : legal) {
// 		pvfn.push_back(pt[r * colSize + c]);
// 	}

// 	return { pvfn, get<1>(res).index({0, 0}).item<float>() };
// }

void PolicyValueNet::train_step(array<float, batchSize * inputDepth * inputSize>& state_batch, 
    array<float, batchSize * outputSize>& nextmove_batch, array<float, batchSize>& winner_batch, float lr) {
	// std::cout << "winner batch : " << winner_batch[0] << " " << winner_batch[1] << std::endl;
	// std::cout << "move batch" << std::endl;
	// for(int i=0; i<outputSize; ++i)
	// 	std::cout << nextmove_batch[i] << " ";
	// std::cout << std::endl;

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sb = torch::from_blob(state_batch.data(), { batchSize, inputDepth, inputRow, inputCol }, options).to(policy_value_net->device);
    torch::Tensor mp = torch::from_blob(nextmove_batch.data(), { batchSize, outputSize }, options).to(policy_value_net->device);
    torch::Tensor wb = torch::from_blob(winner_batch.data(), { batchSize }, options).to(policy_value_net->device);

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
	torch::save(policy_value_net, model_file);
}

void PolicyValueNet::load_model(const string& model_file){
	torch::load(policy_value_net, model_file);
}
