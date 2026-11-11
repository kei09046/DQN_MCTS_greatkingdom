#include "neuralNet.h"
#include "consts.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <filesystem>
#include <regex>

// For 9*9 board.
Net::Net(int channelSize, int blockSize, int headChannels, int valueHidden, bool captureHead): channelSize(channelSize), headChannels(headChannels),
captureHead(captureHead),
cv1(torch::nn::Conv2dOptions(channelSize, 128, 3).padding(1).bias(false)),
bn1(torch::nn::BatchNorm2d(128)),

// Policy head
atCv(torch::nn::Conv2dOptions(128, headChannels, 1).bias(false)),
atBn(torch::nn::BatchNorm2d(headChannels)),
atFc(headChannels * inputSize, outputSize),

// Value head
vCv(torch::nn::Conv2dOptions(128, headChannels, 1).bias(false)),
vBn(torch::nn::BatchNorm2d(headChannels)),
vFc1(headChannels * inputSize, valueHidden),
vFc2(valueHidden, 1),

// Score diff head
scCv(torch::nn::Conv2dOptions(128, headChannels, 1).bias(false)),
scBn(torch::nn::BatchNorm2d(headChannels)),
scFc1(headChannels * inputSize, valueHidden),
scFc2(valueHidden, 1),

// Score map head
scMapCv1(torch::nn::Conv2dOptions(128, headChannels, 1).bias(false)),
scMapBn1(torch::nn::BatchNorm2d(headChannels)),
scMapCv2(torch::nn::Conv2dOptions(headChannels, 1, 1).bias(false))
{
	blocks = register_module("blocks", torch::nn::ModuleList());
	for (int i = 0; i < blockSize; i++) {
		blocks->push_back(ResidualBlock(128));
	}

	register_module("cv1", cv1);
	register_module("bn1", bn1);

	register_module("at_cv", atCv);
	register_module("at_bn", atBn);
	register_module("at_fc", atFc);

	register_module("v_cv", vCv);
	register_module("v_bn", vBn);
	register_module("v_fc1", vFc1);
	register_module("v_fc2", vFc2);

	register_module("sc_cv", scCv);
	register_module("sc_bn", scBn);
	register_module("sc_fc1", scFc1);
	register_module("sc_fc2", scFc2);

	register_module("sc_map_cv1", scMapCv1);
	register_module("sc_map_bn1", scMapBn1);
	register_module("sc_map_cv2", scMapCv2);

	if(captureHead){
		capMapCv1 = register_module("cap_map_cv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, headChannels, 1).bias(false)));
		capMapBn1 = register_module("cap_map_bn1", torch::nn::BatchNorm2d(headChannels));
		capMapCv2 = register_module("cap_map_cv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(headChannels, 1, 1)));
	}
}

FusedConv::FusedConv(const torch::nn::Conv2d& conv, const torch::nn::BatchNorm2d& bn){
	torch::NoGradGuard noGrad;
	const torch::Tensor a = bn->weight / torch::sqrt(bn->running_var + bn->options.eps());
	weight = (conv->weight * a.view({-1, 1, 1, 1})).contiguous();
	bias = bn->bias - a * bn->running_mean;
	if(conv->bias.defined())
		bias += a * conv->bias;
	padding = static_cast<int>(conv->weight.size(2) / 2); // "same" padding, as every convolution here uses
}

void Net::fuse(){
	Fused f;
	f.stem.emplace_back(cv1, bn1);
	for(auto& block : *blocks){
		auto b = block->as<ResidualBlock>();
		f.blocks.emplace_back(b->conv1, b->bn1);
		f.blocks.emplace_back(b->conv2, b->bn2);
	}
	f.at.emplace_back(atCv, atBn);
	f.v.emplace_back(vCv, vBn);
	f.sc.emplace_back(scCv, scBn);
	f.scMap.emplace_back(scMapCv1, scMapBn1);
	if(captureHead)
		f.capMap.emplace_back(capMapCv1, capMapBn1);
	fused = std::move(f);
}

void Net::unfuse(){
	fused.reset();
}

NNOutput Net::forward_fused(const torch::Tensor& state)
{
	const Fused& f = *fused;

	torch::Tensor x = torch::relu(f.stem[0](state));
	for(size_t i = 0; i < f.blocks.size(); i += 2)
		x = torch::relu(f.blocks[i + 1](torch::relu(f.blocks[i](x))) + x);

	torch::Tensor logAct = atFc(torch::relu(f.at[0](x)).view({-1, headChannels * inputSize}));

	torch::Tensor val = torch::relu(f.v[0](x)).view({-1, headChannels * inputSize});
	val = vFc2(torch::relu(vFc1(val))).tanh();

	torch::Tensor score = torch::relu(f.sc[0](x)).view({-1, headChannels * inputSize});
	torch::Tensor expScoreDiff = scFc2(torch::relu(scFc1(score)));

	torch::Tensor expScoreMap = scMapCv2(torch::relu(f.scMap[0](x))).tanh();

	torch::Tensor captureLogit = captureHead ? capMapCv2(torch::relu(f.capMap[0](x))) : torch::zeros_like(expScoreMap);

	return std::make_tuple(logAct, val, expScoreDiff, expScoreMap, captureLogit);
}

NNOutput Net::forward(const torch::Tensor& state)
{
	// during self-play, use fused net instead.
	if(fused && !is_training())
		return forward_fused(state);

	torch::Tensor x = torch::nn::functional::relu(bn1(cv1(state)));
	for (auto& block : *blocks) {
		x = block->as<ResidualBlock>()->forward(x);
	}

	// policy head
	torch::Tensor logAct = torch::nn::functional::relu(atBn(atCv(x))).view({-1, headChannels * inputSize});
	logAct = atFc(logAct);

	// value head
	torch::Tensor val = torch::nn::functional::relu(vBn(vCv(x))).view({-1, headChannels * inputSize});
	val = vFc2(torch::nn::functional::relu(vFc1(val))).tanh();

	// score head
	torch::Tensor score = torch::nn::functional::relu(scBn(scCv(x))).view({-1, headChannels * inputSize});
	torch::Tensor expScoreDiff = scFc2(torch::nn::functional::relu(scFc1(score)));

	// score map head
	torch::Tensor expScoreMap = torch::nn::functional::relu(scMapBn1(scMapCv1(x)));
	expScoreMap = scMapCv2(expScoreMap).tanh();

	// capture map head : logits, sigmoid is applied in the loss.
	torch::Tensor captureLogit = captureHead
		? capMapCv2(torch::nn::functional::relu(capMapBn1(capMapCv1(x))))
		: torch::zeros_like(expScoreMap);

	return std::make_tuple(logAct, val, expScoreDiff, expScoreMap, captureLogit);
}

NNInput PolicyValueNet::get_data(const Game& game){
    std::vector<float> ret(inputSize * globalConfig.inputChannel, 0.0f);
	Color turn = game.get_turn();
	Color oppturn = Game::reverse_color(turn);
	Color state;

	// channel 0, 1, 2 : my/opponent's/neutral stones in current position
	// channel 3, 4, 5 : same, one turn ago (before the last move)
	// channel 6, 7, 8 : same, two turns ago (before the last two moves)

	const Move lastMove = game.get_last_move(1);
	const Move secondLastMove = game.get_last_move(0);
	const auto isStone = [](Move m){ return m != PASSMOVE && m != RESIGNMOVE; };
	const int lastIdx = isStone(lastMove) ? lastMove.first * colSize + lastMove.second : -1;
	const int secondLastIdx = isStone(secondLastMove) ? secondLastMove.first * colSize + secondLastMove.second : -1;

	for(int i=0; i<inputSize; ++i){
		state = game.get_board({i / colSize, i % colSize});
		int ch;
		if(state == turn)
			ch = 0;
		else if(state == oppturn)
			ch = 1;
		else if(state == NEUTRAL)
			ch = 2;
		else
			continue;

		ret.at(ch * inputSize + i) = 1.0f;
		if(i != lastIdx){
			ret.at((3 + ch) * inputSize + i) = 1.0f;
			if(i != secondLastIdx)
				ret.at((6 + ch) * inputSize + i) = 1.0f;
		}
	}

	Color terr;
	Color turnScore = (turn == BLACK) ? BSCORE : WSCORE;
	Color oppturnScore = (turn == BLACK) ? WSCORE : BSCORE;

	for(int i=0; i<inputSize; ++i){ // channel 9, 10 : indicates territory
		terr = game.get_score_board({i/colSize, i%colSize});
		if(terr == turnScore){
			ret.at(9*inputSize + i) = 1.0f;
		}
		else if(terr == oppturnScore){
			ret.at(10*inputSize + i) = 1.0f;
		}
	}

	float diff = game.score_diff(turn);
	for(int i=0; i<inputSize; ++i){ // channel 11 : difference of current score
		ret.at(11*inputSize + i) = diff / boardSize;
	}

	// channel 12 ~ 21 : liberty count(inf if adjacent to territory)
	std::bitset<inputSize> mark;

	for(int i=0; i<inputSize; ++i){
		int cidx = game.get_chain_idx(i);
		const Chain c = game.get_chain(i);

		if(c.size != 0 && !mark[cidx]){
			mark[cidx] = true;

			auto head = game.get_stone({i / colSize, i % colSize}).head;
			auto cur = head;
			auto state = game.get_board({i / colSize, i % colSize});
			int libertyCount = 0;

			for(int j=0; j<boardSize; ++j){
				// if one of the liberty is unplayable by rule(=completely alive)
				if(c.liberties.test(j) && !game.is_legal(j/colSize, j%colSize)){
					libertyCount = 5;
					break;
				}
			}

			// if not completely alive.
			if(libertyCount == 0){
				libertyCount = std::min((int)c.liberties.count(), 4);
			}

			if(state == turn){ // my stone's liberties
				do {
					ret.at((11 + libertyCount)*inputSize + cur) = 1.0f;
					cur = game.get_stone({cur/colSize, cur%colSize}).next;
				} while (cur != head);
			}
			else if(state == oppturn){ // opponent stone's liberties
				do {
					ret.at((16 + libertyCount)*inputSize + cur) = 1.0f;
					cur = game.get_stone({cur/colSize, cur%colSize}).next;
				} while (cur != head);
			}
		}
	}

	// channel 22 : available moves	
	for(const auto& m : game.get_available_moves()){
		if(m < boardSize){
			ret.at(22 * inputSize + m) = 1.0f;
		}
	}

	// channel 23 : territory building move (change : may not be legal right now(due to threat))
	// right now : keep original version.
	for(const auto& m : game.maximize_score(turn, 1)){
		ret.at(23 * inputSize + m) = (ret.at(22 * inputSize + m) == 1.0f) ? 1.0f : 0.0f;
	}

	// channel 24 : opponent's territory building move
	for(const auto& m : game.maximize_score(oppturn, 1)){
		ret.at(24 * inputSize + m) = 1.0f;
	}

	// channel 25 : komi from the side to move's perspective (komi is added to white's score).
	const float komi = (turn == WHITE) ? game.get_komi() : -game.get_komi();
	for(int i=0; i<inputSize; ++i)
		ret.at(25*inputSize + i) = komi / boardSize;

    return {ret};
}

NNInput PolicyValueNet::get_data(const std::vector<const Game*>& gameBatch){
	NNInput ret;
	ret.reserve(gameBatch.size() * inputSize * globalConfig.inputChannel);

	for(int b=0; b<gameBatch.size(); ++b){
		auto data = get_data(*gameBatch[b]);
		// data : std::tuple<std::vector, std::vector, std::vector> with size inputSize * globalConfig.inputChannel, boardSize + 1, boardSize + 1
		ret.insert(ret.end(), data.begin(), data.end());
	}
    return ret;
}


PolicyValueNet::PolicyValueNet(const std::string& modelFile, const std::string& modelType, int numBlocks, bool useGpu):
 useGpu(useGpu), device(useGpu ? torch::kCUDA : torch::kCPU), modelType(modelType), numBlocks(numBlocks)
{
	// cuDNN times its convolution algorithms the first time it sees an input shape and keeps the fastest one.
	// (Each new batch size pays that once.)
	if(useGpu)
		at::globalContext().setBenchmarkCuDNN(true);
	load_model(modelFile);
}

PolicyValueNet::PolicyValueNet(const std::string& modelFile, bool useGpu): 
PolicyValueNet(modelFile, parse_model_name(modelFile).type, parse_model_name(modelFile).blocks, useGpu)
{}

ModelName PolicyValueNet::parse_model_name(const std::string& modelFile){
	const std::string name = std::filesystem::path(modelFile).filename().string();
	static const std::regex current("([A-Z])_(\\d+)b_(.*)\\.pt"), old("([A-Z])(\\d+)B?\\.pt");
	std::smatch m;
	if(std::regex_match(name, m, current)){
		std::smatch g;
		const std::string tag = m[3].str();
		std::optional<int> games;
		if(std::regex_search(tag, g, std::regex("^(\\d+)")))
			games = std::stoi(g[1].str());
		return {m[1].str(), std::stoi(m[2].str()), games};
	}
	if(std::regex_match(name, m, old))
		return {m[1].str(), 10, std::stoi(m[2].str())}; // every model before numHiddenBlock had 10 blocks
	return {globalConfig.modelPrefix, globalConfig.numHiddenBlock, std::nullopt};
}

std::string PolicyValueNet::model_prefix() const{
	return modelType + "_" + std::to_string(numBlocks) + "b_";
}

std::vector<PolicyValueOutput>
PolicyValueNet::batch_evaluate(NNInput& input, int batchSize, PinnedBuffers* pinned){
	const int B = batchSize;
	assert(input.size() == static_cast<size_t>(B) * globalConfig.inputChannel * inputSize);
	std::vector<PolicyValueOutput> outputs;
	outputs.reserve(B);

	// every output of a position side by side in one row : policy, value, score, score map, capture map.
	constexpr int pOff = 0, vOff = outputSize, sOff = vOff + 1, smOff = sOff + 1, cmOff = smOff + boardSize;
	constexpr int outWidth = cmOff + boardSize;

	const auto options = torch::TensorOptions().dtype(torch::kFloat32);
	const torch::Tensor cpuIn = torch::from_blob(input.data(), {B, globalConfig.inputChannel, rowSize, colSize}, options);
	const auto merge = [B](const NNOutput& r){
		return torch::cat({std::get<0>(r).view({B, outputSize}), std::get<1>(r).view({B, 1}), std::get<2>(r).view({B, 1}),
			std::get<3>(r).view({B, boardSize}), std::get<4>(r).view({B, boardSize})}, 1);
	};

	torch::NoGradGuard noGrad;
	torch::Tensor out; // [B, outWidth] on the CPU
	if(useGpu && pinned){
		if(!pinned->in.defined() || pinned->in.size(0) < B){
			pinned->in = torch::empty({B, globalConfig.inputChannel, rowSize, colSize}, options.pinned_memory(true));
			pinned->out = torch::empty({B, outWidth}, options.pinned_memory(true));
		}
		torch::Tensor in = pinned->in.narrow(0, 0, B);
		in.copy_(cpuIn);
		const NNOutput r = policyValueNet->forward(in.to(device, /*non_blocking=*/true));
		out = pinned->out.narrow(0, 0, B);
		out.copy_(merge(r)); // blocking : waits for the forward pass, so pinned->in is free again for the next batch
	}
	else if(useGpu)
		out = merge(policyValueNet->forward(cpuIn.to(device))).to(torch::kCPU);
	else
		out = merge(policyValueNet->forward(cpuIn)).contiguous();

	const float* row = out.data_ptr<float>();
	for(int b = 0; b < B; ++b, row += outWidth) {
		outputs.emplace_back(std::vector<float>(row + pOff, row + pOff + outputSize), row[vOff], row[sOff],
			std::vector<float>(row + smOff, row + smOff + boardSize), std::vector<float>(row + cmOff, row + cmOff + boardSize));
	}
	return outputs;
}

// std::vector<PolicyValueOutput>
// PolicyValueNet::backupEvaluate(const std::vector<const Game*>& gameBatch){
// 	const int B = gameBatch.size();
// 	std::vector<PolicyValueOutput> outputs;
// 	outputs.reserve(B);

// 	auto options = torch::TensorOptions().dtype(torch::kFloat32);
// 	std::vector<float> batchData = getData(gameBatch);

// 	assert(batchData.size() == B * globalConfig.inputChannel * rowSize * colSize);

// 	int cnt = 0;
// 	for(int game_id = 0; game_id < B; ++game_id){
// 		std::cerr << "game : " << game_id << std::endl;
// 		for(int channel = 0; channel < globalConfig.inputChannel; ++channel){
// 			std::cerr << "channel " << channel << std::endl;
// 			for(int row = 0; row < rowSize; ++row){
// 				for(int col = 0; col < colSize; ++col){
// 					std::cerr << batchData[cnt++] << " ";
// 				}
// 				std::cerr << std::endl;
// 			}
// 		}
// 		std::cerr << std::endl;
// 	}

// 	for(int b = 0; b < B; ++b) {
// 		std::vector<float> policy(0.0f, outputSize);
// 		std::vector<float> scMap(0.0f, boardSize);
// 		std::vector<float> capMap(0.0f, boardSize);
// 		outputs.emplace_back(std::move(policy), 0.0f, 0.0f, std::move(scMap), std::move(capMap));
// 	}
// 	return outputs;
// }

PolicyValueOutput PolicyValueNet::evaluate(NNInput& input){
	return std::move(batch_evaluate(input, 1).front());
}

// std::tuple<float, float, float> PolicyValueNet::trainCap(std::vector<float>& state_batch, std::vector<float>& nextmove_batch, 
// 	std::vector<float>& result_batch, std::vector<float>& capture_batch, float lr){
// 	// check first element of batch
// 	// std::cerr << "capture training" << std::endl;
// 	// for(int i=0; i<2; ++i){
// 	// 	for(int j=0; j<rowSize; ++j){
// 	// 		for(int k=0; k<colSize; ++k){
// 	// 			std::cerr << state_batch[i * boardSize + j * colSize + k];
// 	// 		}
// 	// 		std::cerr << std::endl;
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << nextmove_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// std::cerr << result_batch[0] << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << capture_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	int B = result_batch.size();
//     auto options = torch::TensorOptions().dtype(torch::kFloat32);

// 	auto sb = torch::from_blob(state_batch.data(),
//     {B, globalConfig.inputChannel, inputRow, inputCol},
//     options).to(device);

// 	auto mp = torch::from_blob(nextmove_batch.data(),
// 		{B, outputSize},
// 		options).to(device);

// 	auto wb = torch::from_blob(result_batch.data(),
// 		{B, 1},
// 		options).to(device);

// 	auto cb = torch::from_blob(capture_batch.data(),
// 		{B, 1, outputRow, outputCol},
// 		options).to(device);

// 	// auto cap_chanceb = torch::ones({B, 1}, options).to(device);


// 	float pLoss = 0.0f, vLoss = 0.0f, cLoss = 0.0f, cpLoss = 0.0f;
// 	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
// 	for(int i=0; i<globalConfig.epochs; ++i){
// 		optimizer->zero_grad();

// 		auto [r1, r2, r3, r4, r5] = policy_value_net->forward(sb); 

// 		torch::Tensor log_move_probs = torch::log_softmax(r1, 1);
// 		torch::Tensor policy_loss = -torch::mean(torch::sum(mp * log_move_probs, 1));

// 		torch::Tensor value_loss = torch::nn::functional::mse_loss(r2, wb);

// 		float pos_weight = 1.0f;
// 		torch::Tensor loss_tensor = -(cb * torch::log(r5 + 1e-9) * pos_weight + (1 - cb) * torch::log(1 - r5 + 1e-9));
// 		torch::Tensor capture_loss = loss_tensor.mean();

// 		// torch::Tensor cap_prob_loss = torch::nn::functional::mse_loss(r6, cap_chanceb);


// 		torch::Tensor loss = value_loss + policy_loss + capture_loss;
// 		pLoss += policy_loss.item().to<float>();
// 		vLoss += value_loss.item().to<float>();
// 		cLoss += capture_loss.item().to<float>();
// 		// cpLoss += cap_prob_loss.item().to<float>();
// 		// std::cout << "loss epoch " << i << " " << value_loss.item() << " " << policy_loss.item() << std::endl;
// 		loss.backward();
// 		torch::nn::utils::clip_grad_norm_(policy_value_net->parameters(), 1.0);
// 		optimizer->step();
// 	}

// 	return {pLoss / globalConfig.epochs, vLoss / globalConfig.epochs, cLoss / globalConfig.epochs};
// }

// std::tuple<float, float, float, float> PolicyValueNet::trainSc(std::vector<float>& state_batch, std::vector<float>& nextmove_batch, 
// 	std::vector<float>& result_batch, std::vector<float>& score_batch, std::vector<float>& scoremap_batch, float lr) {
// 	// check first element of batch
// 	// std::cerr << "score training" << std::endl;
// 	// for(int i=0; i<2; ++i){
// 	// 	for(int j=0; j<rowSize; ++j){
// 	// 		for(int k=0; k<colSize; ++k){
// 	// 			std::cerr << state_batch[i * boardSize + j * colSize + k];
// 	// 		}
// 	// 		std::cerr << std::endl;
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << nextmove_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	// std::cerr << result_batch[0] << " " << score_batch[0] << std::endl;

// 	// for(int i=0; i<rowSize; ++i){
// 	// 	for(int j=0; j<colSize; ++j){
// 	// 		std::cerr << scoremap_batch[i * colSize + j];
// 	// 	}
// 	// 	std::cerr << std::endl;
// 	// }
// 	// std::cerr << std::endl;

// 	int B = result_batch.size();
//     auto options = torch::TensorOptions().dtype(torch::kFloat32);

// 	auto sb = torch::from_blob(state_batch.data(),
//     {B, globalConfig.inputChannel, inputRow, inputCol},
//     options).to(device);

// 	auto mp = torch::from_blob(nextmove_batch.data(),
// 		{B, outputSize},
// 		options).to(device);

// 	auto wb = torch::from_blob(result_batch.data(),
// 		{B, 1},
// 		options).to(device);

// 	auto sd = torch::from_blob(score_batch.data(),
// 		{B, 1},
// 		options).to(device);

// 	auto smap = torch::from_blob(scoremap_batch.data(),
// 		{B, 1, outputRow, outputCol},
// 		options).to(device);

// 	// auto cap_chanceb = torch::zeros({B, 1}, options).to(device);


// 	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
// 	float pLoss = 0.0f, vLoss = 0.0f, sLoss = 0.0f, smLoss = 0.0f, cpLoss = 0.0f;
// 	torch::nn::HuberLoss huber_loss(torch::nn::HuberLossOptions().delta(5.0).reduction(torch::kMean));

// 	for(int i=0; i<globalConfig.epochs; ++i){
// 		optimizer->zero_grad();

// 		auto [r1, r2, r3, r4, r5] = policy_value_net->forward(sb); 

// 		torch::Tensor log_move_probs = torch::log_softmax(r1, 1);
// 		torch::Tensor policy_loss = -torch::mean(torch::sum(mp * log_move_probs, 1));

// 		torch::Tensor value_loss = torch::nn::functional::mse_loss(r2, wb);

// 		torch::Tensor score_loss = huber_loss(r3, sd);

// 		torch::Tensor score_map_loss = torch::nn::functional::mse_loss(r4, smap);

// 		// torch::Tensor cap_prob_loss = torch::nn::functional::mse_loss(r6, cap_chanceb);

// 		torch::Tensor loss = value_loss + policy_loss + score_loss + score_map_loss;
// 		pLoss += policy_loss.item().to<float>();
// 		vLoss += value_loss.item().to<float>();
// 		sLoss += score_loss.item().to<float>();
// 		smLoss += score_map_loss.item().to<float>();
// 		// cpLoss += cap_prob_loss.item().to<float>();
// 		loss.backward();
// 		torch::nn::utils::clip_grad_norm_(policy_value_net->parameters(), 1.0);
// 		optimizer->step();
// 	}

// 	return {pLoss / globalConfig.epochs, vLoss / globalConfig.epochs, sLoss / globalConfig.epochs, smLoss / globalConfig.epochs};
// }

std::tuple<float, float, float, float, float, float, float> PolicyValueNet::train(std::vector<float>& stateBatch,
	 std::vector<float>& nextmoveBatch, std::vector<float>& resultBatch, std::vector<float>& outcomeBatch, std::vector<float>& scoreBatch,
	  std::vector<float>& mapBatch, std::vector<Trainhead>& typeBatch, float lr, bool update){

	int B = resultBatch.size();
	auto options = torch::TensorOptions().dtype(torch::kFloat32);

	auto sb = torch::from_blob(stateBatch.data(),
		{B, globalConfig.inputChannel, inputRow, inputCol},
		options).to(device);

	auto mp = torch::from_blob(nextmoveBatch.data(),
		{B, outputSize},
		options).to(device);

	auto wb = torch::from_blob(resultBatch.data(),
		{B, 1},
		options).to(device);

	auto zb = torch::from_blob(outcomeBatch.data(),
		{B, 1},
		options).to(device);

	auto sd = torch::from_blob(scoreBatch.data(),
		{B, 1},
		options).to(device);

	auto map = torch::from_blob(mapBatch.data(),
		{B, 1, outputRow, outputCol},
		options).to(device);

	auto mask = torch::from_blob(typeBatch.data(), {B, 1}, torch::kUInt8).to(device);

	auto pmask = (mask & POLICYHEAD).to(torch::kBool);
	auto vmask = (mask & VALUEHEAD).to(torch::kBool);
	auto smask = (mask & SCOREHEAD).to(torch::kBool);
	auto smMask = (mask & OCCUPYHEAD).to(torch::kBool).unsqueeze(-1).unsqueeze(-1);
	auto capMask = (mask & CMAPHEAD).to(torch::kBool).unsqueeze(-1).unsqueeze(-1);

	auto policyActiveElements = pmask.sum().item<float>() + 1e-8f;
	auto valueActiveElements = vmask.sum().item<float>() + 1e-8f;
	auto scoreActiveElements = smask.sum().item<float>() + 1e-8f;
	auto smapActiveElements = smMask.sum().item<float>() * outputRow * outputCol + 1e-8f;
	auto capActiveElements = capMask.sum().item<float>() * outputRow * outputCol + 1e-8f;

	sd *= smask;
	auto smap = map * smMask;
	// capture map target : the map holds the capture map for games ending by capture. For games ending by score it holds
	// the occupancy map (OCCUPYHEAD), and the capture target is all zero : nothing was captured.
	auto cmap = map * (~smMask) * capMask;

	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
	torch::nn::HuberLoss huberLoss(torch::nn::HuberLossOptions().delta(5.0).reduction(torch::kSum));
	float pLoss = 0.0f, vLoss = 0.0f, sLoss = 0.0f, smLoss = 0.0f, cmLoss = 0.0f, vzLoss = 0.0f;

	// entropy of the policy targets: the lowest policy loss reachable on this batch. policy loss - entropy = KL(target || net).
	float pEntropy;
	{
		torch::NoGradGuard noGrad;
		auto entropyPerSample = -torch::sum(mp * torch::log(mp.clamp_min(1e-12f)), 1, /*keepdim=*/true); // [B, 1]
		pEntropy = ((entropyPerSample * pmask.to(torch::kFloat32)).sum() / policyActiveElements).item<float>();
	}

	// BatchNorm uses batch statistics (and updates its running statistics) only while training. The weights change from
	// here on, so the folded inference weights (if any) are dropped.
	policyValueNet->unfuse();
	policyValueNet->train(update);
	torch::AutoGradMode gradMode(update);
	const int passes = update ? globalConfig.epochs : 1;

	//static int cntr = 0;
	for(int i=0; i<passes; ++i){
		optimizer->zero_grad();

		auto [r1, r2, r3, r4, r5] = policyValueNet->forward(sb);
		// if(cntr % 1 == 0){
		// 	std::cout << "iter " << i << "\n";
		// 	for(int j=0; j<B; ++j){
		// 		if(type_batch[j] != NONE){
		// 			displayNNOutput({r1[j], r2[j], r3[j], r4[j], r5[j]});
		// 			break;
		// 		}
		// 	}
		// } 

		torch::Tensor logMoveProbs = torch::log_softmax(r1, 1);
		torch::Tensor policyLossPerSample = -torch::sum(mp * logMoveProbs, 1, /*keepdim=*/true); // [B, 1]
		torch::Tensor policyLoss = (policyLossPerSample * pmask.to(torch::kFloat32)).sum() / policyActiveElements;

		torch::Tensor valueLoss = ((r2 - wb).pow(2) * vmask.to(torch::kFloat32)).sum() / valueActiveElements;

		auto maskedR3 = r3 * smask;
		torch::Tensor scoreLoss = huberLoss(maskedR3, sd);
		scoreLoss = scoreLoss / scoreActiveElements;

		auto maskedR4 = r4 * smMask;
		torch::Tensor scoreMapLoss = torch::nn::functional::mse_loss(maskedR4, smap, torch::nn::MSELossOptions().reduction(torch::kSum));
		scoreMapLoss = scoreMapLoss / smapActiveElements;

		// captured stones are few of 81 points : weight them up so that predicting "nothing captured" everywhere doesn't pay.
		torch::Tensor captureLossPerPoint = torch::nn::functional::binary_cross_entropy_with_logits(r5, cmap,
			torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(torch::full({1}, capturePosWeight, options).to(device))
				.reduction(torch::kNone));
		torch::Tensor captureLoss = (captureLossPerPoint * capMask.to(torch::kFloat32)).sum() / capActiveElements;

		// TODO : find best ratio
		//torch::Tensor loss = policy_loss + value_loss*0.5f + score_loss*0.1f + score_map_loss*0.1f + capture_loss*0.1f;
		torch::Tensor loss = policyLoss + valueLoss + scoreLoss * 0.25f + scoreMapLoss * 0.25f + captureLoss * 0.25f;

		pLoss += policyLoss.item<float>();
		vLoss += valueLoss.item<float>();
		// diagnostic only: same positions, measured against the game result z instead of the (possibly mixed) target.
		vzLoss += ((r2.detach() - zb).pow(2) * vmask.to(torch::kFloat32)).sum().item<float>() / valueActiveElements;
		sLoss += scoreLoss.item<float>();
		smLoss += scoreMapLoss.item<float>();
		cmLoss += captureLoss.item<float>();

		if(update){
			// a single non-finite loss or gradient would write NaN into every weight: skip the step instead.
			if(!std::isfinite(loss.item<float>())){
				std::cerr << "non-finite loss: optimizer step skipped" << std::endl;
				continue;
			}
			loss.backward();
			const double gradNorm = torch::nn::utils::clip_grad_norm_(policyValueNet->parameters(), 1.0);
			if(!std::isfinite(gradNorm)){
				std::cerr << "non-finite gradient norm: optimizer step skipped" << std::endl;
				continue; // zero_grad() at the start of the next pass clears it
			}
			optimizer->step();
		}
	}
	//cntr++;
	policyValueNet->eval();

	return {pLoss / passes, vLoss / passes, sLoss / passes, cmLoss / passes, smLoss / passes, pEntropy, vzLoss / passes};
}

void PolicyValueNet::save_model(const std::string& modelFile) const
{
	auto net = std::dynamic_pointer_cast<Net>(policyValueNet);
	if(!net){
		throw std::runtime_error("Model type mismatch when saving: " + modelFile);
	}
	torch::save(net, modelFile);
}

std::shared_ptr<NetBase> PolicyValueNet::make_net() const{
	if(modelType == "J") //  23 channel model
		return std::make_shared<Net>(23, numBlocks);
	if(modelType == "K") // J + capture map head
		return std::make_shared<Net>(23, numBlocks, 32, 256, true);
	if(modelType == "L")
		return std::make_shared<Net>(25, numBlocks, 32, 256, true);
	if(modelType == "M") // L + komi input channel
		return std::make_shared<Net>(26, numBlocks, 32, 256, true);
	throw std::runtime_error("Unsupported model type: " + modelType);
}

void PolicyValueNet::load_model(const std::string& modelFile){
	std::shared_ptr<NetBase> net = make_net();
	if (modelFile.ends_with(".pt")) {
		torch::load(net, modelFile);
	}
	policyValueNet = std::move(net);

	policyValueNet->to(device);
	// inference mode: BatchNorm normalizes with its running statistics, so an evaluation doesn't depend on the rest of the batch.
	policyValueNet->eval();
	policyValueNet->fuse();

	// learning rate would be modified by train.cpp later.
	optimizer = std::make_unique<torch::optim::Adam>(policyValueNet->parameters(), 
	torch::optim::AdamOptions(0.0002f)
	.betas({0.9, 0.999})
	.weight_decay(l2Const));

	std::cout << "Model loaded: " << modelFile << " (type " << modelType << ", " << numBlocks << " blocks)" << std::endl;
}

void PolicyValueNet::save_optimizer(const std::string& file) const{
	torch::save(*optimizer, file);
}

void PolicyValueNet::load_optimizer(const std::string& file){
	torch::load(*optimizer, file, device);
	std::cout << "Optimizer state loaded: " << file << std::endl;
}

float PolicyValueNet::get_learning_rate() const{
	return static_cast<float>(static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr());
}

void PolicyValueNet::set_learning_rate(float lr){
	static_cast<torch::optim::AdamOptions&>(optimizer->param_groups()[0].options()).lr(lr);
}

void PolicyValueNet::load_weights(const std::string& modelFile){
	std::shared_ptr<NetBase> loaded = make_net();
	torch::load(loaded, modelFile);

	// copy into the existing tensors: the optimizer refers to them, so its state carries over.
	torch::NoGradGuard noGrad;
	auto dstParams = policyValueNet->named_parameters();
	for(const auto& item : loaded->named_parameters())
		dstParams[item.key()].copy_(item.value());
	auto dstBuffers = policyValueNet->named_buffers();
	for(const auto& item : loaded->named_buffers())
		dstBuffers[item.key()].copy_(item.value());

	policyValueNet->eval();
	policyValueNet->fuse();
	std::cout << "Model weights loaded: " << modelFile << std::endl;
}


// std::vector<float> PolicyValueNet::makeScoreDistributionBatch(
//     const std::vector<float>& scores,
//     float scoreRange,
//     float sigma,
//     int window) const
// {
//     int bins = 2 * scoreRange + 1;

//     auto dist = std::vector<float>(scores.size() * bins, 0.0f);

//     for (int i = 0; i < scores.size(); i++)
//     {
//         float s = std::clamp(scores[i], -scoreRange, scoreRange);

//         int start = std::max(-scoreRange, std::floor(s - window));
//         int end   = std::min(scoreRange,  std::ceil(s + window));
// 		float sum = 0.0f;

//         for (int b = start; b <= end; b++)
//         {
//             float diff = b - s;
//             float val = std::exp(-(diff * diff) / (2 * sigma * sigma));

//             dist[i * bins + b + scoreRange] = val;
// 			sum += val;
//         }

// 		for (int b = start; b <= end; b++)
//         {
//             dist[i * bins + b + scoreRange] /= sum;
//         }
//     }

//     return dist;
// }


void PolicyValueNet::display_nn_output(const NNOutput& modelOut){
	// policyLogit : [82] winP : [1] scoreDiff : [1] scoreMap : [81] captureMap : [81]
 	const auto [policyLogit, winP, scoreDiff, scoreMap, captureMap] = modelOut;

	const torch::Tensor policy = torch::softmax(policyLogit, 0);
	
	auto toPrintable = [](const torch::Tensor& t) {
		return t.detach().to(torch::kCPU);
	};

	std::cout << "WinP : " << toPrintable(winP).item<float>() << "\n";
	std::cout << "Predicted Score Diff: " << toPrintable(scoreDiff).item<float>() << "\n\n";

	// 2. Print Policy (Shape [82] -> 9x9 board positions + 1 pass move)
	std::cout << "Policy:\n";
	torch::Tensor boardPolicy = toPrintable(policy).slice(0, 0, 81).view({9, 9});
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			// Format to 4 decimal places for clean grid alignment
			std::printf("%.4f ", boardPolicy[i][j].item<float>());
		}
		std::cout << "\n";
	}
	std::cout << toPrintable(policy)[81].item<float>() << "\n\n";

	// 3. Print Spatial Feature Maps (scoreMap and captureMap reshaped to 9x9)
	torch::Tensor scoreGrid = toPrintable(scoreMap).view({9, 9});
	std::cout << "Score Map:\n";
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			// Format to 4 decimal places for clean grid alignment
			std::printf("%8.4f ", scoreGrid[i][j].item<float>());
		}
		std::cout << "\n";
	}

	// torch::Tensor captureGrid = to_printable(captureMap).view({9, 9});
	// std::cout << "Capture Map:\n";
	// for (int i = 0; i < 9; ++i) {
	// 	for (int j = 0; j < 9; ++j) {
	// 		std::printf("%.4f ", captureGrid[i][j].item<float>());
	// 	}
	// 	std::cout << "\n";
	// }

	
	std::cout << "=====================================\n";
}
