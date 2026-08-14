#ifndef NEURALNET_H
#define NEURALNET_H

#include <torch/torch.h>
#include "gamerules.h"

using NNOutput = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using NNInput = std::vector<float>;

// board, moveprob, result, score diff, (score map/capture map), wintype
using TrainData = std::tuple<NNInput, std::vector<float>, float, float, std::vector<float>, Trainhead>; 


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


// class PureTensorGroupedMaxInfConverterImpl : public torch::nn::Module {
// public:
// public:
//     PureTensorGroupedMaxInfConverterImpl() {}

//     /**
//      * @param x: [BatchSize, N] - The float/half input data tensor
//      * @param group_ids: [BatchSize, N] - Long tensor mapping elements to group IDs. Use -1 for omitted elements.
//      * @param first_indices: [BatchSize, MaxNumGroups] - Long tensor marking target positions for group sums. Pad with -1.
//      */
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& first_indices, const torch::Tensor& group_ids) {
//         int64_t batch_size = x.size(0);
//         int64_t num_elements = x.size(1);
//         int64_t max_groups = first_indices.size(1);

//         // 1. Identify valid elements (Omitted elements are marked -1)
//         auto valid_elements_mask = (group_ids != -1);

//         // 2. Globalize group IDs across the batch to avoid inter-batch contamination
//         auto batch_offsets = torch::arange(0, batch_size, group_ids.options()).view({batch_size, 1}) * max_groups;
        
//         // Wherever group_ids is -1, we temporarily set it to 0 so flat scattering doesn't break.
//         // We will mask out invalid inputs by multiplying their values by 0.
//         auto safe_group_ids = torch::where(valid_elements_mask, group_ids + batch_offsets, 0);
//         auto flat_global_ids = safe_group_ids.reshape({-1});

//         // Step 1 variation: Set omitted input elements to 0 so they contribute nothing to the sum
//         auto filtered_x = torch::where(valid_elements_mask, x, 0.0);
//         auto flat_filtered_x = filtered_x.reshape({-1});

//         // 3. Compute group sums in parallel using non-blocking scatter_reduce
//         int64_t total_global_groups = batch_size * max_groups;
//         auto global_group_sum = torch::zeros({total_global_groups}, x.options());
        
//         // Optimized: No more index_select or nonzero. We scatter the entire flat array.
//         // Omitted inputs are now 0.0 and scatter into index 0, which gets overwritten/corrected 
//         // if index 0 is an actual valid group, or ignored later if it's an unused group.
//         global_group_sum = torch::scatter_reduce(
//             global_group_sum, /*dim=*/0, flat_global_ids, flat_filtered_x, /*reduce=*/"sum", /*include_self=*/false
//         );

//         // 4. Map calculated global sums back onto the first index positions
//         // Initialize the final background canvas with 0.0 (or -inf if your architecture expects it)
//         auto out_flat = torch::zeros({batch_size * num_elements}, x.options());
        
//         // Calculate global absolute target indices for scattering
//         auto element_offsets = torch::arange(0, batch_size, first_indices.options()).view({batch_size, 1}) * num_elements;
//         auto valid_first_mask = (first_indices != -1);
        
//         // Temporarily map invalid target indices to 0 to keep the flat scatter operations safe
//         auto safe_first_indices = torch::where(valid_first_mask, first_indices + element_offsets, 0);
//         auto flat_first_ids = safe_first_indices.reshape({-1});
        
//         auto scatter_sources = torch::arange(0, total_global_groups, first_indices.options());

//         // Optimized: Scatter everything unconditionally to avoid dynamic branching
//         auto gathered_sums = global_group_sum.index_select(0, scatter_sources);
//         out_flat = torch::scatter(out_flat, /*dim=*/0, flat_first_ids, gathered_sums);

//         // Clean up: If a target index was originally -1, force its final output back to 0.0 (or -inf)
//         // This ensures the dummy index 0 write doesn't corrupt actual data
//         auto final_mask = valid_first_mask.reshape({-1});
//         out_flat = torch::where(final_mask, out_flat, 0.0);

//         // 5. Reshape back cleanly to match the original [BatchSize, N] layout
//         return out_flat.reshape({batch_size, num_elements});
//     }
// };
// TORCH_MODULE(PureTensorGroupedMaxInfConverter);


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
	//PureTensorGroupedMaxInfConverter at_cvtr;

	// value
	torch::nn::Conv2d v_cv3;
	torch::nn::BatchNorm2d v_bn3;
	torch::nn::Conv2d v_cv4;
	torch::nn::BatchNorm2d v_bn4;
	torch::nn::Linear v_fc1;
	// torch::nn::Linear v_fc2;

	// score scalar
	torch::nn::Conv2d sc_cv3;
	torch::nn::BatchNorm2d sc_bn3;
	torch::nn::Conv2d sc_cv4;
	torch::nn::BatchNorm2d sc_bn4;
	torch::nn::Linear sc_fc1;
	// torch::nn::Linear sc_fc2;
	
	// score map
	torch::nn::Conv2d sc_map_cv3;
	torch::nn::BatchNorm2d sc_map_bn3;
	torch::nn::Conv2d sc_map_cv4;

	// capture
	// torch::nn::Conv2d cap_cv3;
	// torch::nn::BatchNorm2d cap_bn3;
	// torch::nn::Conv2d cap_cv4;
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

	static NNInput getData(const Game& game);

	static NNInput getData(const std::vector<const Game*>& gameBatch);

	std::vector<PolicyValueOutput> batchEvaluate(const std::vector<const Game*>& gameBatch);

	//std::vector<PolicyValueOutput> backupEvaluate(const std::vector<const Game*>& gameBatch); // run when batchEvaluate fails twice in a row. Checks gameBatch info as well.

	PolicyValueOutput evaluate(const Game& game);

	// void train_step(std::array<float, inputChannel * batchSize * inputSize>& state_batch, std::array<float, batchSize * outputSize>& nextmove_batch,
	// 	std::array<float, batchSize>& result_batch, float lr);

	// std::tuple<float, float, float> trainCap(std::vector<float>& state_batch, std::vector<float>& nextmove_batch,
	// 	std::vector<float>& result_batch, std::vector<float>& capture_batch, float lr);

	// std::tuple<float, float, float, float> trainSc(std::vector<float>& state_batch, std::vector<float>& nextmove_batch,
	// 	std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& scoremap_batch, float lr);
	
	std::tuple<float, float, float, float, float> train(std::vector<float>& state_batch,
         std::vector<float>& nextmove_batch,
		std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& map_batch, std::vector<Trainhead>& type_batch, float lr);

	void save_model(const std::string& model_file) const;

	void load_model(const std::string& model_file);
};

#endif