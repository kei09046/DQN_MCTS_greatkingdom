#include "replaybuffer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
	static_assert(rowSize == colSize, "dihedral augmentation assumes a square board");

	// symTable[s][i] : the point board point i moves to under symmetry s. s = 0 is the identity.
	const std::array<std::array<uint8_t, boardSize>, 8> symTable = [] {
		constexpr int N = rowSize;
		std::array<std::array<uint8_t, boardSize>, 8> t{};
		for(int s = 0; s < 8; ++s){
			for(int r = 0; r < N; ++r){
				for(int c = 0; c < N; ++c){
					int nr = r, nc = c;
					switch(s){
						case 1: nr = c;         nc = N - 1 - r; break; // rot90
						case 2: nr = N - 1 - r; nc = N - 1 - c; break; // rot180
						case 3: nr = N - 1 - c; nc = r;         break; // rot270
						case 4: nr = r;         nc = N - 1 - c; break; // flip horizontal
						case 5: nr = N - 1 - r; nc = c;         break; // flip vertical
						case 6: nr = N - 1 - c; nc = N - 1 - r; break; // anti-diagonal
						case 7: nr = c;         nc = r;         break; // diagonal
					}
					t[s][r * N + c] = static_cast<uint8_t>(nr * N + nc);
				}
			}
		}
		return t;
	}();
}

ReplayBuffer::ReplayBuffer(size_t capacity) : capacity(capacity), rng(std::random_device{}()) {
	if(capacity == 0)
		throw std::invalid_argument("replay buffer capacity must be positive");
}

PackedSample ReplayBuffer::pack(const TrainData& data){
	const auto& [state, policy, result, score, map, type] = data;
	PackedSample s;

	if(state.size() % inputSize != 0)
		throw std::runtime_error("input size is not a multiple of the board size: " + std::to_string(state.size()));
	const int channels = state.size() / inputSize;
	s.planes.resize(channels);

	for(int ch = 0; ch < channels; ++ch){
		const float* plane = state.data() + ch * inputSize;
		const bool binary = std::all_of(plane, plane + inputSize, [](float v){ return v == 0.0f || v == 1.0f; });
		if(binary){
			for(int i = 0; i < inputSize; ++i)
				s.planes[ch][i] = (plane[i] == 1.0f);
		}
		else if(std::all_of(plane, plane + inputSize, [&](float v){ return v == plane[0]; })){
			s.constPlanes.emplace_back(ch, plane[0]);
		}
		else{
			s.densePlanes.emplace_back(ch, std::vector<float>(plane, plane + inputSize));
		}
	}

	for(int i = 0; i < static_cast<int>(policy.size()); ++i){
		if(policy[i] != 0.0f)
			s.policy.emplace_back(i, policy[i]);
	}

	for(int i = 0; i < boardSize; ++i){
		const float v = map.at(i);
		if(v != -1.0f && v != 0.0f && v != 1.0f)
			throw std::runtime_error("occupy map value out of {-1, 0, 1}: " + std::to_string(v));
		s.map[i] = static_cast<int8_t>(v);
	}

	s.result = result;
	s.score = score;
	s.type = type;
	return s;
}

void ReplayBuffer::unpack(const PackedSample& s, int sym, float* state, float* policy, float* map){
	const auto& to = symTable[sym];

	for(int ch = 0; ch < static_cast<int>(s.planes.size()); ++ch){
		float* plane = state + ch * inputSize;
		for(int i = 0; i < inputSize; ++i)
			plane[to[i]] = s.planes[ch][i] ? 1.0f : 0.0f;
	}
	for(const auto& [ch, v] : s.constPlanes)
		std::fill(state + ch * inputSize, state + (ch + 1) * inputSize, v);
	for(const auto& [ch, values] : s.densePlanes){
		float* plane = state + ch * inputSize;
		for(int i = 0; i < inputSize; ++i)
			plane[to[i]] = values[i];
	}

	// pass (index boardSize) is not a board point and stays where it is.
	std::fill(policy, policy + outputSize, 0.0f);
	for(const auto& [idx, p] : s.policy)
		policy[(idx < boardSize) ? to[idx] : idx] = p;

	for(int i = 0; i < boardSize; ++i)
		map[to[i]] = s.map[i];
}

void ReplayBuffer::add(const TrainData& data){
	PackedSample s = pack(data); // outside the lock

	std::lock_guard<std::mutex> lock(mtx);
	if(samples.size() < capacity){
		samples.push_back(std::move(s));
		count.store(samples.size());
	}
	else{
		samples[next] = std::move(s);
		next = (next + 1) % capacity;
	}
}

void ReplayBuffer::sampleBatch(int B, std::vector<float>& state_batch, std::vector<float>& nextmove_batch, std::vector<float>& result_batch,
	std::vector<float>& score_batch, std::vector<float>& map_batch, std::vector<Trainhead>& type_batch){
	std::lock_guard<std::mutex> lock(mtx);
	if(samples.empty())
		throw std::runtime_error("sampling from an empty replay buffer");

	const int channels = samples.front().planes.size();
	if(state_batch.size() < static_cast<size_t>(B) * channels * inputSize || nextmove_batch.size() < static_cast<size_t>(B) * outputSize
		|| map_batch.size() < static_cast<size_t>(B) * boardSize || result_batch.size() < static_cast<size_t>(B)
		|| score_batch.size() < static_cast<size_t>(B) || type_batch.size() < static_cast<size_t>(B))
		throw std::runtime_error("batch arrays are smaller than the requested batch");

	std::uniform_int_distribution<size_t> pickSample(0, samples.size() - 1);
	std::uniform_int_distribution<int> pickSym(0, 7);

	for(int b = 0; b < B; ++b){
		const PackedSample& s = samples[pickSample(rng)];
		unpack(s, pickSym(rng), state_batch.data() + b * channels * inputSize,
			nextmove_batch.data() + b * outputSize, map_batch.data() + b * boardSize);
		result_batch[b] = s.result;
		score_batch[b] = s.score;
		type_batch[b] = s.type;
	}
}
