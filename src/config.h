#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <bitset>

// Configuration, split by consumer. load_config reads a json file into a Config, and each object is handed (and keeps)
// only the part it reads : MCTS a SearchConfig, Evaluator a CacheConfig, EloCalculator a RatingConfig, and so on.
// Nothing here is global.

// starting position of a game : "board" in the json.
struct BoardConfig {
    float komi;
    std::vector<std::pair<uint8_t, uint8_t>> neutrals;
};

// how a Game lists the moves search considers (Game::set_policy_mask) and gives them priors (Game::transfer_logit).
// Not read from the json files. For ablation run only.
struct RuleConfig {
    bool prune = true;    // search only maximal scoring moves. false : every legal move is searched.
    bool transfer = true; // a kept move's prior logit is the max over the logits of the squares it would claim.
                          // false : each move keeps its own logit. Only applies with prune.
    bool oppTerritory = true; // input channel 24 marks the opponent's territory building moves. false : it is all 0.
                              // A model trained with it off must be evaluated with it off too.
};

// one search engine (MCTS) : "mcts", "features", "dirichlet_noise" and train.search_thread_num in the json.
struct SearchConfig {
    std::string mode; // "playout" or "time"
    int time;         // seconds per move in time mode
    int nPlayout;
    int fastPlayout;  // playouts of a fast search (playout cap randomization, see TrainConfig::fullSearchProb)
    float cPuct;
    float temp;       // move selection temperature
    float fpu;        // < 0 : unvisited children are valued 0. >= 0 : parent value - fpu * sqrt(policy mass of visited children).
    float minVisitRatio; // each root child is forced at least nPlayout * minVisitRatio visits. <= 0 disables it.
    bool transTable;
    bool dirichletNoise;
    float alpha;      // dirichlet noise parameter
    float eps;        // dirichlet noise weight
    int searchThreadNum; // leaves evaluated at once by one search
    RuleConfig rules;

    // forced visits per root child for a search of nPlayout playouts. 0 if minVisitRatio <= 0.
    int min_visits(int playouts) const {
        if (minVisitRatio <= 0.0f) return 0;
        return static_cast<int>(playouts * minVisitRatio + 1e-4f); // epsilon guards float error, e.g. 400 * 0.005f
    }
};

// evaluation cache of an Evaluator : "cache" in the json (stored as log2 sizes there).
struct CacheConfig {
    int tableSize;
    int mutexPoolSize;
};

// network built when a model file name doesn't say its type / size (e.g. "none" : an untrained network).
struct NetConfig {
    std::string modelPrefix; // path.model_prefix
    int numHiddenBlock;      // nn.numHiddenBlock
};

// one optimizer step of PolicyValueNet::train.
struct TrainStepConfig {
    int epochs;                 // passes over each batch
    bool captureMapLoss = true; // false (./play train -NcaptureMap) : the capture map loss has weight 0.
};

// self-play training (TrainPipeline) : "train" in the json.
struct TrainConfig {
    int batchSize;
    TrainStepConfig step;
    int checkFreq;
    int compareGameCnt;   // number of games played to compare models
    int compareThreadNum; // should divide compareGameCnt / 2
    int trainWaitTime;    // decides train / inference thread balance
    int synchronizeFreq;
    int capacity;
    int trainStartPoint;  // the train thread starts once the replay buffer holds this many positions. >= batchSize.
    float valueQMix;      // value target = (1 - l) * game result + l * root value after search, l = valueQMix * (n - i) / n at move i of n.
    float windowFraction; // training samples from the newest max(trainStartPoint, windowFraction * positions added) positions, up to capacity.
    float fallbackWinrate;      // a checkpoint winning less than this against the current best is replaced by the best model's weights.
    int lrPatience;             // comparisons in a row without a clear improvement before the learning rate is halved.
    float lrImprovementWinrate; // win rate against the current best that counts as a clear improvement.
    float initialLearningRate;  // used unless the optimizer state of the starting model is loaded.
    // playout cap randomization : each move gets a full search (recorded for training) with probability fullSearchProb,
    // otherwise a fast search (SearchConfig::fastPlayout, not recorded). 1 = every move is a full search.
    float fullSearchProb;
    float fairKomiNoise;  // self-play / model comparison : komi is the setup's fair komi (fairkomi.h) +- up to this
    float wideKomiProb;   // self-play only : chance of a game with komi fair +- up to wideKomiNoise instead
    float wideKomiNoise;
};

struct PathConfig {
    std::string modelPath;
    std::string drivePath;
    std::string dataPath; // replay buffer snapshots
    bool googleDrive;     // also save models to drivePath
};

struct RatingConfig {
    float initK;
    float initRating;
};

// everything in one json file.
struct Config {
    PathConfig paths;
    BoardConfig board;
    NetConfig net;
    SearchConfig search;
    CacheConfig cache;
    TrainConfig train;
    RatingConfig rating;
    // per-engine overrides of fpu / minVisitRatio / prune / transfer ("engines" in the json), applied to search.
    std::vector<SearchConfig> engines;

    // search config of engine idx (evaluate_two / evaluate_multi); search if the json gives no override for it.
    const SearchConfig& engine(int idx) const {
        return (idx < static_cast<int>(engines.size())) ? engines[idx] : search;
    }
};

Config load_config(const std::string& path);

#endif
