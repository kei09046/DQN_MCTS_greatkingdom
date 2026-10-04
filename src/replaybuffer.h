#ifndef REPLAYBUFFER_H
#define REPLAYBUFFER_H

#include "neuralNet.h"
#include "consts.h"
#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

// One self-play position in compact form (~0.7KB instead of ~8KB as floats).
// Stored once, without symmetry copies; a random dihedral transform is applied when it is drawn for training.
struct PackedSample {
	std::vector<std::bitset<boardSize>> planes; // one per input channel. Holds the channel if it is binary (0 / 1), all zero otherwise.
	std::vector<std::pair<uint8_t, float>> constPlanes; // non-binary channels with the same value on every point (e.g. score difference)
	std::vector<std::pair<uint8_t, std::vector<float>>> densePlanes; // any other non-binary channel, stored as is
	std::vector<std::pair<uint8_t, float>> policy; // nonzero entries of the move probability (index into outputSize)
	std::array<int8_t, boardSize> map; // occupy map, values in {-1, 0, 1}
	float result;
	float score;
	Trainhead type;
};

// Fixed-capacity ring buffer of self-play positions. Thread-safe: self-play threads add, the train thread samples.
// Samples are drawn from a window of the newest positions that grows with the number of positions added,
// so early in training the latest games aren't drowned out by the first, weakest ones.
class ReplayBuffer {
private:
	std::vector<PackedSample> samples;
	const size_t capacity;
	const size_t minWindow;
	const float windowFraction;
	size_t added = 0; // positions added in total, including overwritten ones
	size_t next = 0; // slot overwritten by the next add() once the buffer is full
	std::atomic<size_t> count = 0;
	std::mutex mtx;
	std::mt19937 rng; // used only under mtx

	static PackedSample pack(const TrainData& data);

	size_t windowLocked() const; // window(), with mtx already held

	// writes s, transformed by symmetry sym, into the given rows of the batch arrays.
	static void unpack(const PackedSample& s, int sym, float* state, float* policy, float* map);

public:
	ReplayBuffer(size_t capacity, size_t minWindow, float windowFraction);

	void add(const TrainData& data);

	void addPacked(PackedSample s);

	// writes every stored position, oldest first, to path (directories are created). Atomic: written to path.tmp, then renamed.
	void save(const std::string& path);

	// replaces the contents with the positions in path (keeping the newest capacity of them).
	void load(const std::string& path);

	// positions in a file written by save(), oldest first. added : total positions added when it was saved.
	static std::vector<PackedSample> readFile(const std::string& path, size_t& added);

	size_t size() const { return count.load(); }

	size_t totalAdded();

	// number of newest positions sampleBatch draws from: max(minWindow, windowFraction * totalAdded()), at most size().
	size_t window();

	// Draws B positions uniformly at random (with replacement) from the window and applies a random symmetry to each.
	// The batch vectors must already hold at least B rows.
	void sampleBatch(int B, std::vector<float>& state_batch, std::vector<float>& nextmove_batch, std::vector<float>& result_batch,
		std::vector<float>& score_batch, std::vector<float>& map_batch, std::vector<Trainhead>& type_batch);
};

#endif
