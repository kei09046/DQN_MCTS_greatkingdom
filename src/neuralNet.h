#ifndef NEURALNET_H
#define NEURALNET_H

#include <torch/torch.h>
#include "gamerules.h"

using NNOutput = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    ResidualBlockImpl(int channels) {
        conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false))
        );
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));

        conv2 = register_module(
            "conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false))
        );
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
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
    virtual NNOutput forward(const torch::Tensor& state) = 0;
    virtual ~NetBase() = default;
};


class Net : public NetBase{
public:
	Net(int channelSize, int blockSize);
	int channelSize;

	NNOutput forward(const torch::Tensor& state) override;
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;

	torch::nn::ModuleList blocks;
	
	// action(policy)
	torch::nn::Conv2d at_cv3;
	torch::nn::BatchNorm2d at_bn3;
	torch::nn::Conv2d at_cv4;
	torch::nn::BatchNorm2d at_bn4;
	torch::nn::Linear at_fc1;
	torch::nn::Linear at_fc2;

	// value
	torch::nn::Conv2d v_cv3;
	torch::nn::BatchNorm2d v_bn3;
	torch::nn::Conv2d v_cv4;
	torch::nn::BatchNorm2d v_bn4;
	torch::nn::Linear v_fc1;
	torch::nn::Linear v_fc2;

	// score scalar
	torch::nn::Conv2d sc_cv3;
	torch::nn::BatchNorm2d sc_bn3;
	torch::nn::Conv2d sc_cv4;
	torch::nn::BatchNorm2d sc_bn4;
	torch::nn::Linear sc_fc1;
	torch::nn::Linear sc_fc2;
	
	// score map
	torch::nn::Conv2d sc_map_cv3;
	torch::nn::BatchNorm2d sc_map_bn3;
	torch::nn::Conv2d sc_map_cv4;

	// capture
	torch::nn::Conv2d cap_cv3;
	torch::nn::BatchNorm2d cap_bn3;
	torch::nn::Conv2d cap_cv4;
};


class PolicyValueNet {
private:
	bool use_gpu;
	torch::Device device;
	float l2_const = 0.0001f;
	std::unique_ptr<torch::optim::Adam> optimizer;
	const std::string model_type;

	// std::vector<float> makeScoreDistributionBatch(const std::vector<float>& scores, float scoreRange, float sigma, int window) const;

	void displayNNOutput(const NNOutput& modelOut);

public:
	std::shared_ptr<NetBase> policy_value_net;

	PolicyValueNet(const std::string& model_file, const std::string& model_type, bool use_gpu);

	PolicyValueNet(const std::string& model_file, bool use_gpu);

	static std::vector<float> getData(const Game& game);

	static std::vector<float> getData(const std::vector<const Game*>& gameBatch);

	std::vector<PolicyValueOutput> batchEvaluate(const std::vector<const Game*>& gameBatch);

	std::vector<PolicyValueOutput> backupEvaluate(const std::vector<const Game*>& gameBatch); // run when batchEvaluate fails twice in a row. Checks gameBatch info as well.

	PolicyValueOutput evaluate(const Game& game);

	// void train_step(std::array<float, inputChannel * batchSize * inputSize>& state_batch, std::array<float, batchSize * outputSize>& nextmove_batch,
	// 	std::array<float, batchSize>& result_batch, float lr);

	std::tuple<float, float, float> trainCap(std::vector<float>& state_batch, std::vector<float>& nextmove_batch,
		std::vector<float>& result_batch, std::vector<float>& capture_batch, float lr);

	std::tuple<float, float, float, float> trainSc(std::vector<float>& state_batch, std::vector<float>& nextmove_batch,
		std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& scoremap_batch, float lr);
	
	std::tuple<float, float, float, float, float> train(std::vector<float>& state_batch, std::vector<float>& nextmove_batch,
		std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& map_batch, std::vector<Trainhead>& type_batch, float lr);

	void save_model(const std::string& model_file) const;

	void load_model(const std::string& model_file);
};

#endif