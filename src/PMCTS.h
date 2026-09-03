#ifndef PMCTS_H
#define PMCTS_H

#include "consts.h"
#include "evaluator.h"
#include "gamerules.h"
#include <vector>
#include <utility>
#include <unordered_map>

class Node;
using TransTable = std::unordered_map<HashValue, std::pair<Node*, unsigned int>>;

class alignas(64) Node{
public:
    Node(const Game& g, const HashValue hashValue, TransTable* const transposTable);

    Move selectMove(float temperature);

    MoveData selectMoveProb(float temperature);

    Node* jump(Move move);

    void addDirichletNoise(Evaluator* evaluator);

    void deleteTree();

    void deleteTree(Node* exception);

    inline const Game& game_() const{
        return game;
    }

private:
    friend class MCTS;

    Game game; // includes position, territory, valid moves etc. for heuristic
    float N, W, initQ, S, Wp; // N : # of visits, W : total action-value Q : mean action-value P : prior evaluation from nn
    // S : mean score difference Wp : mean win probability
    std::vector<float> edgeP, edgeN; // edge statistics. When transposition table is used, edgeN < childN is possible.
    const Color turn;
    const HashValue hashValue; // hash value needed for transition table and evaluation hash, for each dihedral transformation
    std::vector<Node*> child;

    int forcedState; // if 0 : not forced win/loss +k : win in k move -k : lose in k move
    bool expanded; // check if node has been expanded.
    std::shared_ptr<PolicyValueOutput> evaluation; // stores evaluation for this node. Used to temporarily hold evaluation.
    
    TransTable* const transposTable;

    void addChild(const Move& move, int idx = -1); // add child to the node. Will be added at the end by default.

    void expand();

    int selectChildInSearch();
};


class alignas(64) MCTS{
public:
    MCTS(Evaluator* evaluator);
    ~MCTS();
    MCTS(MCTS&& other) noexcept;

    void runSimulation(const int playMode, const int nPlayout, const int timeLimit);

    Move getMove(float temperature);

    MoveData getMoveProb(float temperature);

    // getEval returns evaluation from player's perspective, which is opposite from internally stored value.
    std::pair<float, float> getEval();

    void printVariation();

    // Prints root winrate + per-candidate-move (visits, policy prior, winrate) breakdown
    // for analysis mode. Requires globalConfig.detailedStat = true (winrate is otherwise
    // never accumulated).
    void printAnalysis();

    bool jump(Move move);

    // resets MCTS tree and set root to startPosition.
    void reset(const Game& startPos);

    // exposes the root node's internal game state, for debugging desyncs against an external mirror (e.g. TrainPipeline::game_manager).
    inline const Game& currentGame() const{
        return root->game_();
    }

    inline int rootForcedState() const{
        return root->forcedState;
    }

    inline int rootVisits() const{
        return static_cast<int>(root->N);
    }

    // Debug mode: while on, every playout's search path and leaf NN evaluation is printed to
    // stdout the moment that playout finishes (see printPlayoutDebugLine in PMCTS.cpp) --
    // nothing is stored or accumulated on the C++ side; the GUI (Python) owns collecting and
    // displaying the resulting stream. Off by default, and always off for every MCTS instance
    // training ever touches (nothing in train.cpp calls this), so training performance is
    // unaffected. reset()/jump() moving the root to a new position don't need to clear anything
    // here for the same reason -- there's nothing accumulated to clear.
    void setDebugMode(bool on);

    inline bool getDebugMode() const{
        return debugMode;
    }

    #ifdef measureTime
    std::vector<int> getTimeStats() const;

    void resetTimeStats();
    #endif

private:
    Node* root;
    Evaluator* evaluator; // shared along multiple MCTS instances
    TransTable* transposTable;

    bool debugMode = false;

    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation,
        std::vector<std::vector<Node*>>& updateQueue, std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int nPlayout, const int timeLimit);

    // Walks from `node` picking the highest-edgeN child at each step (same selection rule as
    // printVariation()'s main line), falling back to the forced win/loss chain when applicable.
    // Used by printAnalysis() to attach a short follow-up line to each candidate move.
    std::vector<Move> followUpFrom(Node* node, int maxDepth);

    void updateEval(const std::shared_ptr<NNResultBuf> buf, const std::vector<Node*> path, Node* cur);

    void propagate(const std::vector<Node*>& path, float evalQ, float evalW=0.0f, float evalS=0.0f);

    // Debug mode only (see setDebugMode above): prints one playout's search path and leaf NN
    // evaluation straight to stdout, without storing anything -- the GUI (Python) is what
    // accumulates/manages the resulting stream of lines. `path` is exactly playout()'s
    // root->leaf node chain, already computed for search/backprop regardless of debug mode, so
    // recovering the actual moves just means, for each consecutive pair, finding which of the
    // parent's children the next node is -- no extra bookkeeping needed on the hot path to
    // make this possible. Only ever called when debugMode is on.
    void printPlayoutDebugLine(const std::vector<Node*>& path, float winP, float scoreEXP, int forcedState);
};
#endif