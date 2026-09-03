#ifndef PMCTS_H
#define PMCTS_H

#include "consts.h"
#include "evaluator.h"
#include "gamerules.h"
#include <vector>
#include <utility>
#include <unordered_map>

class Node;
// Analysis (analysis.hpp/analysis.cpp) reads MCTS's and Node's internal search state directly
// (root, evaluator, child/edgeN/edgeP/forcedState/...) to print it -- exactly what MCTS's own
// analysis methods used to do before that code moved into its own class, so it gets the same
// friend access those methods relied on. Only forward-declared here: neither this header nor
// Node/MCTS's own definitions need Analysis to be a complete type, and analysis.hpp includes
// this header (not the other way around), so keeping this a forward declaration avoids a cycle.
class Analysis;
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
    friend class Analysis;

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

    // Associates this MCTS instance with the Analysis object that owns debug-mode state (see
    // Analysis::debugMode in analysis.hpp) -- purely so playout()/updateEval() below can reach
    // it for the debug-line streaming they do mid-search (see printPlayoutDebugLine's own
    // comment). Never called for any MCTS instance training touches, so `analysis` stays
    // nullptr there and the hot-path checks below cost nothing more than the old inline bool
    // did. Non-owning: whoever constructs the Analysis (ModelCompare::analyze) still owns it.
    inline void attachAnalysis(Analysis* a){
        analysis = a;
    }

    #ifdef measureTime
    std::vector<int> getTimeStats() const;

    void resetTimeStats();
    #endif

private:
    friend class Analysis;

    Node* root;
    Evaluator* evaluator; // shared along multiple MCTS instances
    TransTable* transposTable;

    // Non-owning; null unless attachAnalysis() was called (see above). Only playout()/updateEval()
    // below read this, to stream debug lines through it mid-search -- everything else Analysis
    // does (printAnalysis, printVariation, ...) is driven the other way, by a caller handing its
    // own MCTS instance into an Analysis method, so doesn't need MCTS to hold this at all.
    Analysis* analysis = nullptr;

    void playout(int& searchCounter, int& evaluateCounter, std::vector<Node*>& inEvaluation,
        std::vector<std::vector<Node*>>& updateQueue, std::vector<std::shared_ptr<NNResultBuf>>& resultBuffer, bool& searchStuck,
    const int playMode, const int nPlayout, const int timeLimit);

    void updateEval(const std::shared_ptr<NNResultBuf> buf, const std::vector<Node*> path, Node* cur);

    void propagate(const std::vector<Node*>& path, float evalQ, float evalW=0.0f, float evalS=0.0f);
};
#endif