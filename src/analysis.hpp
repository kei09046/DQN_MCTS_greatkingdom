#ifndef ANALYSIS_HPP
#define ANALYSIS_HPP

#include "PMCTS.h"

// Analysis-mode-only functionality for the MCTS engine, as its own class rather than methods
// tacked onto MCTS: used exclusively by the interactive "analyze" REPL
// (ModelCompare::analyze, src/modelcompare.cpp) and its GUI (game.py). Nothing in the core
// search loop (playout/updateEval/propagate, still in PMCTS.cpp) or in training (train.cpp;
// grep shows it never constructs an Analysis or calls attachAnalysis) touches any of it except
// for the debug-line hot-path hook noted below -- so none of this can affect training
// performance, even by accident, just by editing this file.
//
// Analysis holds no reference to any particular MCTS instance -- every method that needs one
// takes it as a parameter (an Analysis is reusable across differently-configured MCTS players,
// e.g. an engine-vs-engine match driving two of them, though nothing in this codebase does that
// yet). It gets MCTS's and Node's private search state (root, evaluator, child/edgeN/edgeP/
// forcedState/...) via `friend class Analysis` on both, the same access their own analysis
// methods relied on before this class existed, rather than a large getter/setter surface.
//
// debugMode lives here now, not on MCTS -- but printPlayoutDebugLine is called from *inside*
// MCTS::playout()/updateEval() while a playout is streaming, which can't reach into a caller's
// separate Analysis object on its own. MCTS::attachAnalysis() (see PMCTS.h) is the minimal hook
// back the other way: an optional non-owning Analysis* on MCTS, null unless a caller (only
// ModelCompare::analyze does) wires one up, so the hot-path check stays exactly as cheap as the
// inline bool it replaces for every MCTS instance training ever touches.
class Analysis{
public:
    // Prints root winrate + per-candidate-move (visits, policy prior, winrate) breakdown for
    // the "analyze" REPL. Requires globalConfig.detailedStat = true (winrate is otherwise never
    // accumulated).
    void printAnalysis(MCTS& mcts);

    // One-line-per-ply dump of the current principal variation.
    void printVariation(MCTS& mcts);

    // Debug mode: while on, every playout's search path and leaf NN evaluation is printed to
    // stdout the moment that playout finishes (see printPlayoutDebugLine below) -- nothing is
    // stored or accumulated here either; the GUI (Python) owns collecting and displaying the
    // resulting stream. Off by default. reset()/jump() moving an attached MCTS's root to a new
    // position don't need to clear anything here -- there's nothing accumulated to clear.
    inline void setDebugMode(bool on){
        debugMode = on;
    }

    inline bool getDebugMode() const{
        return debugMode;
    }

    // Debug mode only: prints one playout's search path and leaf NN evaluation straight to
    // stdout, without storing anything -- the GUI (Python) is what accumulates/manages the
    // resulting stream of lines. `path` is exactly MCTS::playout()'s root->leaf node chain,
    // already computed for search/backprop regardless of debug mode, so recovering the actual
    // moves just means, for each consecutive pair, finding which of the parent's children the
    // next node is -- no extra bookkeeping needed on the hot path to make this possible. Takes
    // no MCTS& (unlike printAnalysis/printVariation above) since it needs nothing from one --
    // `path`'s own Node*s are all it reads. Only ever called (via MCTS::analysis, see
    // attachAnalysis in PMCTS.h) when debugMode is on.
    void printPlayoutDebugLine(const std::vector<Node*>& path, float winP, float scoreEXP, int forcedState);

private:
    bool debugMode = false;

    // Walks from `node` picking the highest-edgeN child at each step (same selection rule as
    // printVariation()'s main line), falling back to the forced win/loss chain when applicable.
    // Used by printAnalysis() to attach a short follow-up line to each candidate move. Takes a
    // plain Node* rather than an MCTS& like printAnalysis/printVariation -- it walks down from
    // wherever it's pointed (a specific child, not necessarily an MCTS's own root) and needs
    // nothing else from the owning MCTS to do that.
    std::vector<Move> followUpFrom(Node* node, int maxDepth);
};

#endif
