#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <vector>
#include <utility>

// search options that can differ between engines (e.g. two engines in evaluate_two / evaluate_multi).
struct SearchParams {
    float fpu;
    float minVisitRatio; // each root child is forced at least nPlayout * minVisitRatio visits. <= 0 disables it.

    // forced visits per root child for a search of nPlayout playouts. 0 if minVisitRatio <= 0.
    int minVisits(int nPlayout) const {
        if (minVisitRatio <= 0.0f) return 0;
        return static_cast<int>(nPlayout * minVisitRatio + 1e-4f); // epsilon guards float error, e.g. 400 * 0.005f
    }
};

struct Config {
    // features
    bool transTable;
    bool dirichletNoise;
    bool googleDrive;
    bool detailedStat;
    float fpu;

    //path
    std::string modelPath;
    std::string modelPrefix;
    std::string drivePath;
    std::string dataPath; // replay buffer snapshots

    // board
    float komi;
    std::vector<std::pair<int, int>> neutrals; 

    //neuralNet constants
    int batchSize;
    int inputChannel;

    //dirichlet noise
    float alpha; // dirichlet noise parameter
    float eps;   // dirichlet noise weight

    // mcts
    std::string mode; // either time or playout.
    int time;
    int nPlayout;
    float cPuct;
    float temp;
    float minVisitRatio; // each root child is forced at least nPlayout * minVisitRatio visits. <= 0 disables it.
    // playout cap randomization (self-play only) : each move gets a full search (nPlayout, recorded for training) with
    // probability fullSearchProb, otherwise a fast search (fastPlayout, not recorded). 1 = every move is a full search.
    float fullSearchProb;
    int fastPlayout;

    // per-engine overrides of fpu / minVisitRatio, read from "engines" in the config. Index = engine order.
    std::vector<SearchParams> engines;

    SearchParams searchParams() const {
        return {fpu, minVisitRatio};
    }

    // search params for engine idx; falls back to the global values if no override is given.
    SearchParams engineParams(int idx) const {
        return (idx < static_cast<int>(engines.size())) ? engines[idx] : searchParams();
    }

    int minVisits(int nPlayout) const {
        return searchParams().minVisits(nPlayout);
    }

    //cache
    int tableSize;
    int mutexPoolSize; // shardCount * capPerShard -> maximum cache size

    //rating
    float initK;
    float initRating;

    //train
    int play_batch_size;
    int epochs;
    int check_freq;
    int compare_game_cnt; // number of games played to compare models
    int compare_thread_num; // should divide compare_game_cnt / 2
    int search_thread_num; // if apv-MCTS : # of PMCTS threads if not : # of simultaneous evaluation for one thread
    int train_wait_time; // decide train/inference thread balance.
    int save_freq; // 96
    int capacity;
    int trainStartPoint; // the train thread starts once the replay buffer holds this many positions. >= batchSize.
    float valueQMix; // value target = (1 - valueQMix) * game result + valueQMix * root value after search. 0 = game result only.
    float windowFraction; // training samples from the newest max(trainStartPoint, windowFraction * positions added) positions, up to capacity.
    float fallbackWinrate; // a checkpoint winning less than this against the current best is replaced by the best model's weights.
    int lrPatience; // comparisons in a row without a clear improvement before the learning rate is halved.
    float lrImprovementWinrate; // win rate against the current best that counts as a clear improvement (resets the lrPatience counter).
    float initialLearningRate; // used unless the optimizer state of the starting model is loaded.
};
extern Config globalConfig;

Config loadConfig(const std::string& path);

#endif