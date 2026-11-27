#ifndef NEURALNET_H
#define NEURALNET_H

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <memory>
#include <optional>
#include <string>
#include "gamerules.h"

// policy logits, value, score, score (occupancy) map, capture map logits
using NNOutput = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using NNInput = std::vector<float>;

// page-locked host buffers for PolicyValueNet::batchEvaluate. Owned by whoever runs the evaluations (one per Evaluator) :
// as a thread_local they were freed in the thread-exit destructors, after libtorch's own per-thread CUDA state, and crashed.
struct PinnedBuffers{
	torch::Tensor in, out;
	void free(){ in.reset(); out.reset(); }
};

// board, moveprob, value target, score diff, (score map/capture map), heads to train, game result z.
// The value target may mix z with the search's value; z is kept separately to measure how well the value head predicts games.
using TrainData = std::tuple<NNInput, std::vector<float>, float, float, std::vector<float>, Trainhead, float>;


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
    // inference with each BatchNorm folded into the convolution before it (same outputs, fewer GPU kernels). fuse()
    // computes the folded weights from the current weights and running statistics; forward uses them in eval mode
    // until unfuse(). Call fuse() again after the weights change.
    virtual void fuse() = 0;
    virtual void unfuse() = 0;
    virtual ~NetBase() = default;
};

// a convolution with its BatchNorm folded in : BN(conv(x)) = conv(x; a W) + b, a = gamma / sqrt(var + eps), b = beta - a mean.
struct FusedConv {
	torch::Tensor weight, bias;
	int padding;

	FusedConv(const torch::nn::Conv2d& conv, const torch::nn::BatchNorm2d& bn);
	torch::Tensor operator()(const torch::Tensor& x) const { return torch::conv2d(x, weight, bias, 1, padding); }
};


class Net : public NetBase{
public:
	Net(int channelSize, int blockSize, int headChannels = 32, int valueHidden = 256, bool captureHead = false);
	int channelSize;
	int headChannels;
	bool captureHead;

	NNOutput forward(const torch::Tensor& state) override;
	void fuse() override;
	void unfuse() override;
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;

	torch::nn::ModuleList blocks;

	// action(policy)
	torch::nn::Conv2d atCv;
	torch::nn::BatchNorm2d atBn;
	torch::nn::Linear atFc;

	// value
	torch::nn::Conv2d vCv;
	torch::nn::BatchNorm2d vBn;
	torch::nn::Linear vFc1;
	torch::nn::Linear vFc2;

	// score scalar
	torch::nn::Conv2d scCv;
	torch::nn::BatchNorm2d scBn;
	torch::nn::Linear scFc1;
	torch::nn::Linear scFc2;

	// score map
	torch::nn::Conv2d scMapCv1;
	torch::nn::BatchNorm2d scMapBn1;
	torch::nn::Conv2d scMapCv2;

	// capture map
	torch::nn::Conv2d capMapCv1{nullptr};
	torch::nn::BatchNorm2d capMapBn1{nullptr};
	torch::nn::Conv2d capMapCv2{nullptr};

private:
	// folded convolutions (see NetBase::fuse), not saved with the model. Empty when not fused.
	struct Fused {
		std::vector<FusedConv> stem, blocks; // blocks : conv1, conv2 of each residual block
		std::vector<FusedConv> at, v, sc, scMap, capMap;
	};
	std::optional<Fused> fused;

	NNOutput forward_fused(const torch::Tensor& state);
};


// model file name <type>_<blocks>b_<tag>.pt, e.g. K_10b_13440.pt (tag : games played, optionally + "B" for a best model).
// <type> is a capital letter (architecture) optionally followed by letters / digits, e.g. MnS_12b_13440.pt.
struct ModelName {
	std::string type;
	int blocks;
	std::optional<int> games; // games played, if the tag starts with a number
};

struct PolicyValueOutput{
    std::vector<float> policy;
    float winP;
    float scoreDiff;
    std::vector<float> scoreMap;
    std::vector<float> captureMap;
};

class PolicyValueNet {
public:
	static constexpr float capturePosWeight = 5.0f; // BCE weight of captured points in the capture map loss

	std::shared_ptr<NetBase> policyValueNet;

	// model type and block count from the file name (see ModelName); newNet's if the name doesn't give them.
	PolicyValueNet(const std::string& modelFile, const NetConfig& newNet, bool useGpu);

	// nullopt if the file name doesn't follow either naming scheme.
	static std::optional<ModelName> parse_model_name(const std::string& modelFile);

	// retrieve NN input from the game
	static NNInput get_data(const Game& game);

	static NNInput get_data(const std::vector<const Game*>& gameBatch);

	// Gives model name prefix. ex) K_10b_
	std::string model_prefix() const;

	const std::string& model_type() const { return modelType; }

	int num_blocks() const { return numBlocks; }


	// input : batchSize positions from getData, back to back (batchSize * inputChannel * inputSize floats).
	// pinned : reused for the host <-> GPU transfer; without it the transfer goes through pageable memory and the
	// recorded graphs (capture_graphs) aren't used.
	std::vector<PolicyValueOutput> batch_evaluate(NNInput& input, int batchSize, PinnedBuffers* pinned = nullptr);

	//std::vector<PolicyValueOutput> backupEvaluate(const std::vector<const Game*>& gameBatch); // run when batchEvaluate fails twice in a row. Checks gameBatch info as well.

	// input : one position from getData.
	PolicyValueOutput evaluate(NNInput& input);

	// batchNorm2d + conv2d fusion.
	void fuse_batch_norm() { policyValueNet->fuse(); graphs.clear(); }

	// Records the fused forward pass as one CUDA graph per batch size, so that batch_evaluate with pinned buffers
	// launches a single graph instead of every kernel. Run while other threads are paused.
	void capture_graphs();
	
	// returns policy, value, score, capture map, score map loss, the entropy of the policy targets and the value head's
	// squared error against the game result z alone (outcome_batch; not trained on, for comparing value targets).
	// update = false only measures the losses (inference mode, no optimizer step), e.g. on validation data.
	// step : passes over the batch (update only) and which losses are trained.
	std::tuple<float, float, float, float, float, float, float> train(std::vector<float>& stateBatch,
         std::vector<float>& nextmoveBatch,
		std::vector<float>& resultBatch, std::vector<float>& outcomeBatch, std::vector<float>& scoreBatch, std::vector<float>& mapBatch, std::vector<Trainhead>& typeBatch, float lr,
		const TrainStepConfig& step, bool update = true);

	void save_model(const std::string& modelFile) const;

	void load_model(const std::string& modelFile);

	// replaces the weights (and BatchNorm statistics) with the ones in modelFile, keeping the optimizer state.
	void load_weights(const std::string& modelFile);

	// Adam state (moment estimates), so a resumed run continues where the optimizer stopped.
	void save_optimizer(const std::string& file) const;

	void load_optimizer(const std::string& file);

	// learning rate stored in the optimizer: saved and loaded with its state. train() sets it on every call.
	float get_learning_rate() const;

	void set_learning_rate(float lr);


private:
	bool useGpu;
	torch::Device device;
	float l2Const = 0.0001f;
	std::unique_ptr<torch::optim::Adam> optimizer;
	const std::string modelType;
	const int numBlocks;

	// used for model name parsing.
	static ModelName name_or_new(const std::string& modelFile, const NetConfig& newNet);

	// untrained network of this->model_type.
	std::shared_ptr<NetBase> make_net() const;

	// for debugging.
	void display_nn_output(const NNOutput& modelOut);

	// the fused forward pass recorded as a CUDA graph for one batch size (see capture_graphs).
	struct GraphBucket{
		int batchSize;
		std::unique_ptr<at::cuda::CUDAGraph> graph;
		torch::Tensor in, out; // the graph's fixed input [batchSize, ...] and merged output [batchSize, outWidth] on the GPU
	};
	std::vector<GraphBucket> graphs; // ascending batchSize. Empty : batch_evaluate runs the net op by op.
};

#endif