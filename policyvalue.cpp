#include "policyvalue.h"
#include <cmath>
#include <iostream>
using namespace std;


// net : action probability 의 log와 value (-1, 1)을 추정한다.
NetImpl::NetImpl(bool use_gpu = false): cv1(torch::nn::Conv2dOptions(inputDepth, 32, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(32)),
cv2(torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)),
bn2(torch::nn::BatchNorm2d(64)),
cv3(torch::nn::Conv2dOptions(64, 64, 3).padding(1).bias(false)),
bn3(torch::nn::BatchNorm2d(64)),
cv4(torch::nn::Conv2dOptions(64, 64, 3).padding(1).bias(false)),
bn4(torch::nn::BatchNorm2d(64)),
at_cv3(torch::nn::Conv2dOptions(64, 2, 1).padding(0).bias(false)),
at_bn3(torch::nn::BatchNorm2d(2)),
at_fc1(torch::nn::Linear(2 * inputSize, outputSize)),
v_cv3(torch::nn::Conv2dOptions(64, 1, 1).padding(0).bias(false)),
v_bn3(torch::nn::BatchNorm2d(1)),
v_fc1(torch::nn::Linear(inputSize, 128)),
v_fc2(torch::nn::Linear(128, 1)),
device(use_gpu ? torch::kCUDA : torch::kCPU){

	cv1->to(device);
	bn1->to(device);
	cv2->to(device);
	bn2->to(device);
	cv3->to(device);
	bn3->to(device);
	cv4->to(device);
	bn4->to(device);
	at_cv3->to(device);
	at_bn3->to(device);
	at_fc1->to(device);
	v_cv3->to(device);
	v_bn3->to(device);
	v_fc1->to(device);
	v_fc2->to(device);
	
	register_module("cv1", cv1);
	register_module("bn1", bn1);
	register_module("cv2", cv2);
	register_module("bn2", bn2);
	register_module("cv3", cv3);
	register_module("bn3", bn3);
	register_module("cv4", cv4);
	register_module("bn4", bn4);
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
	x = torch::nn::functional::relu(bn2(cv2(x)));
	x = torch::nn::functional::relu(bn3(cv3(x)));
	x = torch::nn::functional::relu(bn4(cv4(x)));

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

std::array<float, inputDepth * inputSize> PolicyValueNet::getData(const Game* game){
    std::array<float, inputDepth * inputSize> ret;
	color turn = game->getTurn();
	color state;

    ret.fill(0.0f);
	for(int i=0; i<inputSize; ++i){
		state = game->getBoard(i / colSize, i % colSize);
		if(state == turn)
			ret[i] = 1.0f;
		else if(state == Game::reverseColor(turn))
			ret[inputSize + i] = 1.0f;
		else if(state == NEUTRAL)
			ret[2 * inputSize + i] = 1.0f;
	}

	for(int i = 3*inputSize; i < 4*inputSize; ++i){
		ret[i] = static_cast<float>(turn);
	}
    return ret;
}

std::vector<float> PolicyValueNet::getData(std::vector<const Game*> gameBatch){
	//std::cout << "get data " << gameBatch.size() << std::endl;
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

	//std::cout << "Get data finished" << std::endl;
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


std::pair< std::vector<std::vector<float>>, std::vector<float> >
PolicyValueNet::batchEvaluate(std::vector<const Game*> gameBatch, const std::vector<std::vector<std::pair<int, int>>*> legalBatch)
{
	int size = gameBatch.size();
	//std::cout << "batch size : " << size << std::endl;

	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor current_state = torch::from_blob(getData(gameBatch).data(), { size, inputDepth, inputRow, inputCol}, options).to(policy_value_net->device);

	//std::cout << "batch evaluate" << std::endl;
	tuple<torch::Tensor, torch::Tensor> res;
	if (use_gpu) {
		auto r = policy_value_net->forward(current_state);
		get<0>(res) = get<0>(r).to(torch::kCPU);
		get<1>(res) = get<1>(r).to(torch::kCPU);
	}
	else {
		res = policy_value_net->forward(current_state);
	}

	float* pt = get<0>(res).data_ptr<float>();
	int temp;
	std::vector<std::vector<float>> pvfn(size);

	// TODO : fix
	for(int num = 0; num < size; ++num){
		temp = 0;
		for (pair<int, int> i : *(legalBatch[num])) {
			pt += (i.first * colSize + i.second) - temp;
			temp = i.first * colSize + i.second;
			pvfn[num].push_back(*pt);
		}
	}

	std::vector<float> values(size);
	torch::Tensor values_tensor = get<1>(res).flatten();  // Ensure it's a 1D tensor
    std::memcpy(values.data(), values_tensor.data_ptr<float>(), size * sizeof(float));

	//std::cout << "batch evaluate finished" << std::endl;
	return { pvfn, values };
}

PolicyValueOutput PolicyValueNet::evaluate(const Game* game, const std::vector<std::pair<int, int> > legal)
{
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor current_state = torch::from_blob(getData(game).data(), { 1, inputDepth, rowSize, colSize }, options).to(policy_value_net->device);
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
	for (auto [r, c] : legal) {
		pvfn.push_back(pt[r * colSize + c]);
	}

	return { pvfn, get<1>(res).index({0, 0}).item<float>() };
}

// pair<array<float, outputSize>, float> PolicyValueNet::policy_value_fn(array<float, inputDepth * inputSize> state) {
// 	auto options = torch::TensorOptions().dtype(torch::kFloat32);
// 	torch::Tensor current_state = torch::from_blob(state.data(), { 1, inputDepth, inputRow, inputCol }, options).to(policy_value_net->device);
// 	tuple<torch::Tensor, torch::Tensor> res;
// 	if (use_gpu) {
// 		auto r = policy_value_net->forward(current_state);
// 		get<0>(res) = get<0>(r).to(torch::kCPU);
// 		get<1>(res) = get<1>(r).to(torch::kCPU);
// 	}
// 	else {
// 		res = policy_value_net->forward(current_state);
// 	}

// 	float* pt = get<0>(res).data<float>();
// 	int temp = 0;

// 	for (int i = 0; i < outputSize; ++i) {
// 		pvfn[i] = exp(*pt);
// 		pt++;
// 	}

// 	return { pvfn, get<1>(res).index({0, 0}).item<float>() };
// }

// TODO : fix loss update function
// void PolicyValueNet::train_step(array<float, batchSize * inputDepth * inputSize>& state_batch, array<float, batchSize * outputSize>& nextmove_batch,
// 	array<float, batchSize>& winner_batch, float lr) {

// 	auto options = torch::TensorOptions().dtype(torch::kFloat32);
// 	torch::Tensor sb = torch::from_blob(state_batch.data(), { batchSize, inputDepth, inputRow, inputCol }, options).to(policy_value_net->device);
// 	torch::Tensor mp = torch::from_blob(nextmove_batch.data(), { batchSize, outputSize }, options).to(policy_value_net->device);
// 	torch::Tensor wb = torch::from_blob(winner_batch.data(), { batchSize }, options).to(policy_value_net->device);

// 	optimizer->zero_grad();
// 	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);

// 	torch::Tensor r1, r2;
// 	tie(r1, r2) = policy_value_net->forward(sb);

// 	torch::Tensor value_loss = torch::nn::functional::mse_loss(r2.view(-1), wb);
// 	torch::Tensor policy_loss = -torch::mean(torch::sum(mp * r1, 1));
// 	torch::Tensor loss = value_loss + policy_loss;

// 	loss.backward();
// 	optimizer->step();
// 	return;
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
