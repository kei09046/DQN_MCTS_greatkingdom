#ifndef POLICYVALUE_H
#define POLICYVALUE_H

#include <torch/torch.h>
#include "gamerules.h"
#include "consts.h"

class NetImpl : public torch::nn::Module{
public:
	NetImpl(bool use_gpu);
	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state);
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;

	torch::nn::Conv2d rb1_conv1;
	torch::nn::BatchNorm2d rb1_bn1;
	torch::nn::Conv2d rb1_conv2;
	torch::nn::BatchNorm2d rb1_bn2;

	torch::nn::Conv2d rb2_conv1;
	torch::nn::BatchNorm2d rb2_bn1;
	torch::nn::Conv2d rb2_conv2;
	torch::nn::BatchNorm2d rb2_bn2;

	torch::nn::Conv2d rb3_conv1;
	torch::nn::BatchNorm2d rb3_bn1;
	torch::nn::Conv2d rb3_conv2;
	torch::nn::BatchNorm2d rb3_bn2;

	torch::nn::Conv2d rb4_conv1;
	torch::nn::BatchNorm2d rb4_bn1;
	torch::nn::Conv2d rb4_conv2;
	torch::nn::BatchNorm2d rb4_bn2;

	torch::nn::Conv2d rb5_conv1;
	torch::nn::BatchNorm2d rb5_bn1;
	torch::nn::Conv2d rb5_conv2;
	torch::nn::BatchNorm2d rb5_bn2;

	torch::nn::Conv2d rb6_conv1;
	torch::nn::BatchNorm2d rb6_bn1;
	torch::nn::Conv2d rb6_conv2;
	torch::nn::BatchNorm2d rb6_bn2;
	
	torch::nn::Conv2d at_cv3;
	torch::nn::BatchNorm2d at_bn3;
	torch::nn::Linear at_fc1;
	torch::nn::Conv2d v_cv3;
	torch::nn::BatchNorm2d v_bn3;
	torch::nn::Linear v_fc1;
	torch::nn::Linear v_fc2;

	torch::Device device;
};
TORCH_MODULE(Net);

class PolicyValueNet {
private:
	bool use_gpu;
	float l2_const = 0.0001f;
	torch::optim::Adam* optimizer;

public:
	Net policy_value_net;

	PolicyValueNet(const std::string& model_file, bool use_gpu);

	static InputMatrix getData(const Game& game);

	static std::vector<float> getData(const std::vector<const Game*>& gameBatch);

	std::vector<PolicyValueOutput> batchEvaluate(const std::vector<const Game*>& gameBatch);

	PolicyValueOutput evaluate(const Game& game);

	//PolicyValueOutput evaluate(const Game& game, const std::vector<std::pair<int, int> > legal);

	void train_step(std::array<float, inputChannel * batchSize * inputSize>& state_batch, std::array<float, batchSize * outputSize>& mcts_probs,
		std::array<float, batchSize>& winner_batch, float lr);

	void save_model(const std::string& model_file) const;

	void load_model(const std::string& model_file);
};

#endif