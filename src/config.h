#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <vector>
#include <utility>

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
};
extern Config globalConfig;

Config loadConfig(const std::string& path);

#endif