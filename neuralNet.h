#ifndef POLICYVALUE_H
#define POLICYVALUE_H

#include <torch/torch.h>
#include "gamerules.h"
#include "consts.h"


struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    ResidualBlockImpl(size_t channels) {
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false));
        bn1 = torch::nn::BatchNorm2d(channels);
        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false));
        bn2 = torch::nn::BatchNorm2d(channels);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1(conv1(x)));
        out = bn2(conv2(out));
        return torch::relu(out + x);
    }
};
TORCH_MODULE(ResidualBlock);


class NetBase : public torch::nn::Module {
public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state) = 0;
    virtual ~NetBase() = default;
};


class GNet : public NetBase{
public:
	GNet();
	std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& state) override;
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;

	std::vector<ResidualBlock> blocks;
	
	torch::nn::Conv2d at_cv3;
	torch::nn::BatchNorm2d at_bn3;
	torch::nn::Linear at_fc1;
	torch::nn::Conv2d v_cv3;
	torch::nn::BatchNorm2d v_bn3;
	torch::nn::Linear v_fc1;
	torch::nn::Linear v_fc2;
};

class PolicyValueNet {
private:
	bool use_gpu;
	torch::Device device;
	float l2_const = 0.0001f;
	std::unique_ptr<torch::optim::Adam> optimizer;
	const std::string model_type;

public:
	std::shared_ptr<NetBase> policy_value_net;

	PolicyValueNet(const std::string& model_file, const std::string& model_type, bool use_gpu);

	PolicyValueNet(const std::string& model_file, bool use_gpu);

	static InputMatrix getData(const Game& game);

	static std::vector<float> getData(const std::vector<const Game*>& gameBatch);

	std::vector<PolicyValueOutput> batchEvaluate(const std::vector<const Game*>& gameBatch);

	PolicyValueOutput evaluate(const Game& game);

	void train_step(std::array<float, inputChannel * batchSize * inputSize>& state_batch, std::array<float, batchSize * outputSize>& mcts_probs,
		std::array<float, batchSize>& winner_batch, float lr);

	void save_model(const std::string& model_file) const;
};

#endif