#include "replaybuffer.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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

namespace {
	// file layout : header, then per position (oldest first)
	//   planes : channels x 2 uint64 (bit i = point i), constPlanes : u8 count + (u8 channel, float),
	//   densePlanes : u8 count + (u8 channel, boardSize floats), policy : u8 count + (u8 index, float),
	//   map : boardSize int8, result float, score float, type u8.
	constexpr char fileMagic[4] = {'G', 'K', 'R', 'B'};
	constexpr uint32_t fileVersion = 1;
	static_assert(boardSize <= 128, "planes are stored as two 64-bit words");

	template <typename T>
	void put(std::ostream& out, const T& v){
		out.write(reinterpret_cast<const char*>(&v), sizeof(T));
	}

	template <typename T>
	T get(std::istream& in){
		T v;
		in.read(reinterpret_cast<char*>(&v), sizeof(T));
		if(!in)
			throw std::runtime_error("replay buffer file is truncated");
		return v;
	}

	void writeSample(std::ostream& out, const PackedSample& s){
		for(const auto& plane : s.planes){
			uint64_t words[2] = {0, 0};
			for(int i = 0; i < boardSize; ++i)
				if(plane[i])
					words[i / 64] |= uint64_t{1} << (i % 64);
			put(out, words[0]);
			put(out, words[1]);
		}
		put(out, static_cast<uint8_t>(s.constPlanes.size()));
		for(const auto& [ch, v] : s.constPlanes){
			put(out, ch);
			put(out, v);
		}
		put(out, static_cast<uint8_t>(s.densePlanes.size()));
		for(const auto& [ch, values] : s.densePlanes){
			put(out, ch);
			out.write(reinterpret_cast<const char*>(values.data()), boardSize * sizeof(float));
		}
		put(out, static_cast<uint8_t>(s.policy.size()));
		for(const auto& [idx, p] : s.policy){
			put(out, idx);
			put(out, p);
		}
		out.write(reinterpret_cast<const char*>(s.map.data()), boardSize);
		put(out, s.result);
		put(out, s.score);
		put(out, s.type);
	}

	PackedSample readSample(std::istream& in, uint32_t channels){
		PackedSample s;
		s.planes.resize(channels);
		for(auto& plane : s.planes){
			const uint64_t words[2] = {get<uint64_t>(in), get<uint64_t>(in)};
			for(int i = 0; i < boardSize; ++i)
				plane[i] = (words[i / 64] >> (i % 64)) & 1;
		}
		for(int k = get<uint8_t>(in); k > 0; --k){
			const auto ch = get<uint8_t>(in);
			s.constPlanes.emplace_back(ch, get<float>(in));
		}
		for(int k = get<uint8_t>(in); k > 0; --k){
			const auto ch = get<uint8_t>(in);
			std::vector<float> values(boardSize);
			in.read(reinterpret_cast<char*>(values.data()), boardSize * sizeof(float));
			s.densePlanes.emplace_back(ch, std::move(values));
		}
		for(int k = get<uint8_t>(in); k > 0; --k){
			const auto idx = get<uint8_t>(in);
			s.policy.emplace_back(idx, get<float>(in));
		}
		in.read(reinterpret_cast<char*>(s.map.data()), boardSize);
		s.result = get<float>(in);
		s.score = get<float>(in);
		s.type = get<Trainhead>(in);
		return s;
	}
}

ReplayBuffer::ReplayBuffer(size_t capacity, size_t minWindow, float windowFraction)
	: capacity(capacity), minWindow(minWindow), windowFraction(windowFraction), rng(std::random_device{}()) {
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

namespace {
	// name of the first part of data holding a non-finite number, or nullptr if everything is finite.
	const char* nonFinitePart(const TrainData& data){
		const auto& [state, policy, result, score, map, type] = data;
		const auto finite = [](const std::vector<float>& v){
			return std::all_of(v.begin(), v.end(), [](float x){ return std::isfinite(x); });
		};
		if(!finite(state)) return "input";
		if(!finite(policy)) return "policy target";
		if(!std::isfinite(result)) return "value target";
		if(!std::isfinite(score)) return "score target";
		if(!finite(map)) return "map target";
		return nullptr;
	}
}

void ReplayBuffer::add(const TrainData& data){
	if(const char* bad = nonFinitePart(data)){
		++droppedCount;
		std::cerr << "replay buffer : dropped a position with a non-finite " << bad << std::endl;
		return;
	}
	addPacked(pack(data)); // packed outside the lock
}

void ReplayBuffer::addPacked(PackedSample s){
	std::lock_guard<std::mutex> lock(mtx);
	++added;
	if(samples.size() < capacity){
		samples.push_back(std::move(s));
		count.store(samples.size());
	}
	else{
		samples[next] = std::move(s);
		next = (next + 1) % capacity;
	}
}

size_t ReplayBuffer::windowLocked() const{
	const size_t grown = static_cast<size_t>(windowFraction * added);
	return std::min(std::max(minWindow, grown), samples.size());
}

size_t ReplayBuffer::totalAdded(){
	std::lock_guard<std::mutex> lock(mtx);
	return added;
}

size_t ReplayBuffer::window(){
	std::lock_guard<std::mutex> lock(mtx);
	return windowLocked();
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

	// age 0 is the newest position. Before the buffer is full the newest is at the back, afterwards just before next.
	const size_t newest = (samples.size() < capacity) ? samples.size() - 1 : (next + capacity - 1) % capacity;
	std::uniform_int_distribution<size_t> pickAge(0, windowLocked() - 1);
	std::uniform_int_distribution<int> pickSym(0, 7);

	for(int b = 0; b < B; ++b){
		const PackedSample& s = samples[(newest + samples.size() - pickAge(rng)) % samples.size()];
		unpack(s, pickSym(rng), state_batch.data() + b * channels * inputSize,
			nextmove_batch.data() + b * outputSize, map_batch.data() + b * boardSize);
		result_batch[b] = s.result;
		score_batch[b] = s.score;
		type_batch[b] = s.type;
	}
}

void ReplayBuffer::save(const std::string& path){
	std::lock_guard<std::mutex> lock(mtx);

	const std::filesystem::path target(path);
	if(target.has_parent_path())
		std::filesystem::create_directories(target.parent_path());
	const std::string tmp = path + ".tmp";

	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		if(!out)
			throw std::runtime_error("cannot open " + tmp + " for writing");

		const uint32_t channels = samples.empty() ? 0 : samples.front().planes.size();
		out.write(fileMagic, sizeof(fileMagic));
		put(out, fileVersion);
		put(out, static_cast<uint32_t>(boardSize));
		put(out, channels);
		put(out, static_cast<uint64_t>(samples.size()));
		put(out, static_cast<uint64_t>(added));

		// oldest first. Once the buffer is full, the oldest position is the one next overwrites.
		const bool full = samples.size() == capacity;
		for(size_t k = 0; k < samples.size(); ++k)
			writeSample(out, samples[full ? (next + k) % capacity : k]);

		out.close();
		if(!out)
			throw std::runtime_error("failed writing " + tmp);
	}
	std::filesystem::rename(tmp, target);
}

std::vector<PackedSample> ReplayBuffer::readFile(const std::string& path, size_t& added){
	std::ifstream in(path, std::ios::binary);
	if(!in)
		throw std::runtime_error("cannot open replay buffer file " + path);

	char magic[sizeof(fileMagic)];
	in.read(magic, sizeof(magic));
	if(!in || !std::equal(magic, magic + sizeof(magic), fileMagic))
		throw std::runtime_error(path + " is not a replay buffer file");
	if(const auto version = get<uint32_t>(in); version != fileVersion)
		throw std::runtime_error(path + " has unsupported version " + std::to_string(version));
	if(const auto points = get<uint32_t>(in); points != boardSize)
		throw std::runtime_error(path + " is for a board of " + std::to_string(points) + " points");

	const auto channels = get<uint32_t>(in);
	const auto count = get<uint64_t>(in);
	added = get<uint64_t>(in);

	std::vector<PackedSample> result;
	result.reserve(count);
	for(uint64_t i = 0; i < count; ++i)
		result.push_back(readSample(in, channels));
	return result;
}

void ReplayBuffer::load(const std::string& path){
	size_t fileAdded = 0;
	std::vector<PackedSample> loaded = readFile(path, fileAdded);
	if(loaded.size() > capacity) // keep the newest
		loaded.erase(loaded.begin(), loaded.end() - capacity);

	std::lock_guard<std::mutex> lock(mtx);
	samples = std::move(loaded);
	next = 0; // stored oldest first: when full, index 0 is the oldest and is overwritten next
	added = std::max(fileAdded, samples.size());
	count.store(samples.size());
}
