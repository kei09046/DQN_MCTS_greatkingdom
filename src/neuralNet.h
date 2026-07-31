#ifndef NEURALNET_H
#define NEURALNET_H

#include <torch/torch.h>
#include "gamerules.h"

using NNOutput = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;
using NNInput = std::tuple<std::vector<float>, std::vector<int>, std::vector<int>>;

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


class PureTensorGroupedMaxInfConverterImpl : public torch::nn::Module {
public:
    PureTensorGroupedMaxInfConverterImpl() {}

    /**
     * @param x: [BatchSize, N] - The float/half input data tensor
     * @param group_ids: [BatchSize, N] - Long tensor mapping elements to group IDs. Use -1 for omitted elements.
     * @param first_indices: [BatchSize, MaxNumGroups] - Long tensor marking target positions for group maxes. Pad with -1.
     */

    // corresponds to state, availableMoves, transferList.
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& first_indices, const torch::Tensor& group_ids) {
        // x : [B, outputSize]
        // first_indices : [B, boardSize + 1]
        // group_ids : [B, boardSize + 1]
        int64_t batch_size = x.size(0);
        int64_t num_elements = x.size(1);
        int64_t max_groups = first_indices.size(1);

        // debug: catch out-of-range group_ids/first_indices synchronously here, before the async
        // CUDA scatter/gather kernels below turn a bad index into an opaque device-side assert.
        // {
        //     int64_t max_gid = group_ids.max().item<int64_t>();
        //     int64_t min_gid = group_ids.min().item<int64_t>();
        //     int64_t max_fi = first_indices.max().item<int64_t>();
        //     int64_t min_fi = first_indices.min().item<int64_t>();
        //     if(max_gid >= max_groups || min_gid < -1 || max_fi >= num_elements || min_fi < -1){
        //         std::cerr << "BAD CONVERTER INPUT: max_gid=" << max_gid << " min_gid=" << min_gid
        //                   << " max_fi=" << max_fi << " min_fi=" << min_fi
        //                   << " max_groups=" << max_groups << " num_elements=" << num_elements
        //                   << " batch_size=" << batch_size << std::endl;
        //         std::cerr << "group_ids:\n" << group_ids.to(torch::kCPU) << std::endl;
        //         std::cerr << "first_indices:\n" << first_indices.to(torch::kCPU) << std::endl;
        //         std::abort();
        //     }
        // }

        // 1. Isolate valid group elements and mask omitted elements (-1) to -inf
        auto valid_elements_mask = (group_ids != -1);
        auto inf_tensor = torch::full_like(x, -std::numeric_limits<float>::infinity());
        auto filtered_x = torch::where(valid_elements_mask, x, inf_tensor);

        // 2. Globalize group IDs across the batch to avoid inter-batch contamination
        // Global group IDs will span from 0 to (BatchSize * MaxNumGroups - 1)
        auto batch_offsets = torch::arange(0, batch_size, group_ids.options()).view({batch_size, 1}) * max_groups;
        
        // Wherever group_ids is -1, keep it -1; otherwise, add the batch offset
        auto global_group_ids = torch::where(valid_elements_mask, group_ids + batch_offsets, -1);

        // 3. Compute group maximums in parallel using a flattened scatter_reduce
        int64_t total_global_groups = batch_size * max_groups;
        auto global_group_max = torch::full({total_global_groups}, -std::numeric_limits<float>::infinity(), x.options());
        
        // scatter_reduce naturally ignores indices that are negative when using valid dimensions, 
        // but to be perfectly safe with all LibTorch versions, we flatten everything except the -1s.
        auto flat_global_ids = global_group_ids.reshape({-1});
        auto flat_filtered_x = filtered_x.reshape({-1});
        
        // Filter out the -1 entries to perform a clean parallel reduction
        auto valid_reduce_mask = (flat_global_ids >= 0);
        auto reduce_ids = flat_global_ids.index_select(0, torch::nonzero(valid_reduce_mask).squeeze(1));
        auto reduce_vals = flat_filtered_x.index_select(0, torch::nonzero(valid_reduce_mask).squeeze(1));

        if (reduce_ids.size(0) > 0) {
            global_group_max = torch::scatter_reduce(
                global_group_max, /*dim=*/0, reduce_ids, reduce_vals, /*reduce=*/"amax", /*include_self=*/false
            );
        }

        // 4. Map the calculated global maximums back onto the first index positions
        auto out_flat = torch::full({batch_size * num_elements}, -std::numeric_limits<float>::infinity(), x.options());
        
        // Calculate global absolute target indices for scattering
        auto element_offsets = torch::arange(0, batch_size, first_indices.options()).view({batch_size, 1}) * num_elements;
        auto valid_first_mask = (first_indices != -1);
        auto global_first_indices = torch::where(valid_first_mask, first_indices + element_offsets, -1);

        // Flatten and extract only valid scatter instructions
        auto flat_first_ids = global_first_indices.reshape({-1});
        auto valid_scatter_mask = (flat_first_ids >= 0);
        
        auto scatter_destinations = flat_first_ids.index_select(0, torch::nonzero(valid_scatter_mask).squeeze(1));
        auto scatter_sources = torch::arange(0, total_global_groups, first_indices.options())
                                    .index_select(0, torch::nonzero(valid_scatter_mask).squeeze(1));

        if (scatter_destinations.size(0) > 0) {
            auto gathered_maxes = global_group_max.index_select(0, scatter_sources);
            out_flat = torch::scatter(out_flat, /*dim=*/0, scatter_destinations, gathered_maxes);
        }

        // 5. Reshape back cleanly to match the original [BatchSize, N] layout
        return out_flat.reshape({batch_size, num_elements});
    }
};
TORCH_MODULE(PureTensorGroupedMaxInfConverter);


class NetBase : public torch::nn::Module {
public:
    virtual NNOutput forward(const torch::Tensor& state, const torch::Tensor& available_moves, const torch::Tensor& transfer_table) = 0;
    virtual ~NetBase() = default;
};


class Net : public NetBase{
public:
	Net(int channelSize, int blockSize);
	int channelSize;

	NNOutput forward(const torch::Tensor& state, const torch::Tensor& available_moves, const torch::Tensor& transfer_table) override;
	torch::nn::Conv2d cv1;
	torch::nn::BatchNorm2d bn1;

	torch::nn::ModuleList blocks;
	
	// action(policy)
	torch::nn::Conv2d at_cv3;
	torch::nn::BatchNorm2d at_bn3;
	torch::nn::Conv2d at_cv4;
	torch::nn::BatchNorm2d at_bn4;
	torch::nn::Linear at_fc1;
	PureTensorGroupedMaxInfConverter at_cvtr;

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
	
	std::tuple<float, float, float, float, float> train(std::vector<float>& state_batch, std::vector<float>& available_batch, std::vector<float>& transfer_batch,
         std::vector<float>& nextmove_batch,
		std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& map_batch, std::vector<Trainhead>& type_batch, float lr);

	void save_model(const std::string& model_file) const;

	void load_model(const std::string& model_file);
};

#endif