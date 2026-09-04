#include "analysis.hpp"
#include <iostream>

// See analysis.hpp for what lives here and why it's split into its own class.

void Analysis::printVariation(MCTS& mcts){
    Node* node = mcts.root;

    while(node->N > 1){
        int maxv = -1;
        int maxi = -1;
        int m;

        if(node->forcedState == 0){
            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                // if(!globalConfig.transTable)
                //     assert(node->edgeN[i] == node->child[i]->N);
                // else
                //     assert(node->edgeN[i] <= node->child[i]->N);

                if(node->edgeN[i] > maxv){
                    maxv = node->edgeN[i];
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;

            m = node->game.getAvailableMoves()[maxi];
        }
        else if(node->forcedState > 0){
            for(int i=0; i<node->child.size(); ++i){
                if(node->child[i]->forcedState < 0){
                    m = node->game.getAvailableMoves()[i];
                }
            }
            // std::cerr << "only move : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;

            if(node->forcedState == 2)
                break;

            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                if(node->game.getAvailableMoves()[i] == m){
                    maxi = i;
                    break;
                }
            }
            // if(maxi == -1){
            //     for(const auto& options : node->game.getAvailableMoves())
            //         std::cerr << "options : " << static_cast<int>(options.first) << " " << static_cast<int>(options.second) << std::endl;
            // }
        }
        else{ // losing. Delay losing as much as possible.
            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }

            if(maxi == -1)
                break;
            m = node->game.getAvailableMoves()[maxi];
        }

        int visit = node->edgeN[maxi];
        node = node->child[maxi];
        if(node == nullptr)
            return;
        std::cout << (int)m / colSize << " " << (int)m % colSize << " " << visit << " forced " << node->forcedState << " Q: " <<
            node->W/node->N << " initQ : " << node->initQ << " Wp : " << node->Wp/node->N
            << " S : " << node->S / node->N << std::endl;
    }
}

std::vector<Move> Analysis::followUpFrom(Node* node, int maxDepth){
    std::vector<Move> seq;

    while(node != nullptr && node->N > 1 && (int)seq.size() < maxDepth){
        int maxv = -1;
        int maxi = -1;
        int m;

        if(node->forcedState == 0){
            for(int i=0; i<node->game.getAvailableMoves().size(); ++i){
                if(node->edgeN[i] > maxv){
                    maxv = node->edgeN[i];
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;
            m = node->game.getAvailableMoves()[maxi];
        }
        else if(node->forcedState > 0){
            // A node's forcedState can be set at birth (Node::addChild, when the move that
            // created it already ended the game) without ever going through expand() filling
            // in `child` -- expand() deliberately skips that for any already-forced node, so
            // `child` can be shorter than game.getAvailableMoves() (even empty) here. Bound by
            // the smaller of the two so we never index past either array; an empty/partial
            // child list just means no continuation to show, which the maxi==-1 check below
            // already handles gracefully.
            int limit = (int)node->child.size() < (int)node->game.getAvailableMoves().size()
                ? (int)node->child.size() : (int)node->game.getAvailableMoves().size();
            for(int i=0; i<limit; ++i){
                if(node->child[i] != nullptr && node->child[i]->forcedState < 0){
                    m = node->game.getAvailableMoves()[i];
                    maxi = i;
                    break;
                }
            }
            if(maxi == -1 || node->forcedState == 2)
                break;
        }
        else{ // losing. Delay losing as much as possible.
            int limit = (int)node->child.size() < (int)node->game.getAvailableMoves().size()
                ? (int)node->child.size() : (int)node->game.getAvailableMoves().size();
            for(int i=0; i<limit; ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;
            m = node->game.getAvailableMoves()[maxi];
        }

        seq.push_back({static_cast<uint8_t>(m / colSize), static_cast<uint8_t>(m % colSize)});
        node = node->child[maxi];
    }

    return seq;
}

void Analysis::printPlayoutDebugLine(const std::vector<Node*>& path, float winP, float scoreEXP, int forcedState){
    std::cout << "playout forced " << forcedState << " winp " << winP << " score " << scoreEXP << " path";
    for(size_t i=0; i+1<path.size(); ++i){
        const std::vector<Node*>& child = path[i]->child;
        for(size_t a=0; a<child.size(); ++a){
            if(child[a] == path[i+1]){
                int m = path[i]->game.getAvailableMoves()[a];
                std::cout << " " << (m / colSize) << " " << (m % colSize);
                break;
            }
        }
    }
    std::cout << std::endl;
}

void Analysis::printAnalysis(MCTS& mcts){
    Node* root = mcts.root;

    std::cout << "analysis begin" << std::endl;

    if(root->forcedState > 0)
        std::cout << "winrate : 1" << std::endl;
    else if(root->forcedState < 0)
        std::cout << "winrate : -1" << std::endl;
    else if(root->N > 0)
        // Every playout adds root's OWN Wp contribution with a sign already oriented so that
        // root->Wp/root->N == -(visit-weighted average of the *children's* raw Wp/N below) --
        // verified numerically against real search output. Since the per-move winrate below is
        // deliberately left unnegated (child->Wp/N, matching how selectChildInSearch() reads
        // child->W directly as its PUCT preference -- the heaviest-visited child empirically has
        // the *highest* unnegated value, not the lowest), root's own line needs the negation to
        // land on the same scale: "root winrate" ends up the weighted average of the move winrates
        // printed below, rather than their complement.
        std::cout << "winrate : " << (-root->Wp / root->N) << std::endl;
    else
        std::cout << "winrate : 0" << std::endl;
    std::cout << "visits : " << root->N << std::endl;
    // Once forcedState != 0 the engine has proven a win/loss and the "analyze" command's search
    // loop (see ModelCompare::analyze) stops for good right here, well short of the requested
    // playout target -- expose this explicitly so the GUI can tell "search is genuinely done"
    // apart from "still climbing toward the target", instead of just comparing visits to target.
    std::cout << "forced : " << root->forcedState << std::endl;
    // root->initQ is set exactly once, the first time this exact position is ever evaluated by
    // the network (calculateQ's blended-utility output) -- i.e. the raw value-head verdict before
    // any tree search refined it. Comparing it against the searched winrate above shows how much
    // search moved the evaluation away from the network's first guess.
    std::cout << "initQ : " << root->initQ << std::endl;
    // Search-refined score-head estimate: root->S accumulates a scoreEXP contribution from every
    // playout along the search tree (MCTS::propagate), negated per ply exactly like W/Wp above --
    // same backup mechanism, same "root->S/root->N is the negated weighted average of the moves
    // below" relationship, and same visit-count denominator. Unlike scoreExp printed further down
    // (always just the network's untouched one-shot guess at the current position), this one moves
    // as search deepens, so together the pair mirrors initQ/winrate for the score head.
    if(root->N > 0)
        std::cout << "scoreSearch : " << (-root->S / root->N) << std::endl;
    else
        std::cout << "scoreSearch : 0" << std::endl;

    // scoreExp / scoreMap are per-point (or, for scoreExp, whole-board) NN outputs for the
    // *current* root position, not per-child search stats -- fetch them directly
    // (evaluator->evaluate hits the cache, since root was already evaluated by the very first
    // playout, so this is cheap). (There used to be a third, captureMap, per-point capture-risk
    // head here too; it's been removed from the model, so PolicyValueOutput is just these two.)
    // scoreExp: the score head's raw scalar output -- net (Black - White) territory/capture
    // margin, from whoever is to move at root's perspective (turn-relative, same convention as
    // winrate/scoreMap below), without komi applied (see PolicyValueOutput in consts.h and the
    // training label in train.cpp, `-scoreDiff(startingTurn)`).
    // scoreMap: tanh output in [-1, 1], matching the training label convention in
    // Game::makeMove's end-of-game scoreMap (-1 = Black-owned point, +1 = White-owned point).
    {
        auto buf = std::make_shared<NNResultBuf>();
        mcts.evaluator->evaluate(buf, &root->game, root->hashValue);
        const auto& [policy, wp, sd, scoreMap] = *buf->result;
        std::cout << "scoreExp : " << sd << std::endl;
        std::cout << "scoreMap :";
        for(float v : scoreMap) std::cout << " " << v;
        std::cout << std::endl;
    }

    // per-move breakdown is only meaningful once the root has been expanded with real moves.
    if(root->forcedState == 0){
        const auto& moves = root->game.getAvailableMoves();
        for(int i=0; i<moves.size(); ++i){
            int r = moves[i] / colSize;
            int c = moves[i] % colSize;

            float visits = (i < root->edgeN.size()) ? root->edgeN[i] : 0.0f;
            float prior = (i < root->edgeP.size()) ? root->edgeP[i] : 0.0f;
            float winrate = 0.0f;
            float q = 0.0f;
            // child->forcedState, from the CHILD's own mover's perspective (see
            // Node::selectChildInSearch, src/PMCTS.cpp:266-281): negative means the child's own
            // mover (the opponent, one ply down) loses there -- i.e. THIS move is a proven win
            // for root's mover -- and positive means the opposite, a proven loss for root's
            // mover. selectChildInSearch short-circuits to the first negative child it finds
            // (skipping the ordinary pref comparison entirely) and suppresses positive ones to a
            // fixed -2.0 + ... pref instead of the ordinary q + cPuct*prior*.../(1+visits) --
            // neither of which the GUI's own pref column could reproduce without this field.
            int childForced = (i < root->child.size() && root->child[i] != nullptr)
                                   ? root->child[i]->forcedState : 0;
            if(i < root->child.size() && root->child[i] != nullptr && root->child[i]->N > 0){
                winrate = root->child[i]->Wp / root->child[i]->N;
                q = root->child[i]->W / root->child[i]->N; // blended utility actually used for PUCT selection
            }

            std::cout << "move " << r << " " << c
                       << " visits " << visits
                       << " prior " << prior
                       << " winrate " << winrate
                       << " q " << q
                       << " forced " << childForced
                       << " variation";
            if(i < root->child.size() && root->child[i] != nullptr){
                for(const Move& fm : followUpFrom(root->child[i], 6)){
                    std::cout << " " << (int)fm.first << " " << (int)fm.second;
                }
            }
            std::cout << std::endl;
        }
    }
    // Once root is forced, every other child's search stats are stale (search stopped the
    // instant forcedState flipped), so the full breakdown above doesn't apply -- but the GUI
    // still wants to see *which* move wins/delays and how the line continues, instead of an
    // empty move list. Find that single move the same way followUpFrom finds it one ply down
    // (a child already proven losing for the opponent when winning, or the least-bad delaying
    // child when losing), then print it as the lone "move" line with its own variation tail.
    else{
        const auto& moves = root->game.getAvailableMoves();
        int limit = (int)root->child.size() < (int)moves.size() ? (int)root->child.size() : (int)moves.size();
        int maxi = -1;
        int maxv = -1;
        if(root->forcedState > 0){
            for(int i=0; i<limit; ++i){
                if(root->child[i] != nullptr && root->child[i]->forcedState < 0){
                    maxi = i;
                    break;
                }
            }
        }
        else{
            for(int i=0; i<limit; ++i){
                if(root->child[i] != nullptr && root->child[i]->forcedState > maxv){
                    maxv = root->child[i]->forcedState;
                    maxi = i;
                }
            }
        }

        if(maxi != -1){
            int r = moves[maxi] / colSize;
            int c = moves[maxi] % colSize;
            Node* chosen = root->child[maxi];

            float visits = (maxi < (int)root->edgeN.size()) ? root->edgeN[maxi] : 0.0f;
            float prior = (maxi < (int)root->edgeP.size()) ? root->edgeP[maxi] : 0.0f;
            // Clamped to the proven result, same as root's own "winrate" line just above --
            // NOT the raw chosen->Wp/N, chosen->W/N running average. That average only reflects
            // whatever ordinary (pre-proof) visits this child happened to accumulate before its
            // win/loss was discovered, which can sit far from +-1 if most of its search history
            // predates the proof; and since the search loop stops for good the instant
            // root->forcedState flips, it would never get the chance to climb back toward +-1
            // on its own. Same (unnegated) sign scale as root's clamped line above, per the
            // comment there: root->forcedState > 0 means THIS move -- the one that makes that
            // true -- is a proven win for root's own mover, and vice versa for < 0.
            float clamped = (root->forcedState > 0) ? 1.0f : -1.0f;
            float winrate = clamped;
            float q = clamped;
            int childForced = chosen->forcedState;

            std::cout << "move " << r << " " << c
                       << " visits " << visits
                       << " prior " << prior
                       << " winrate " << winrate
                       << " q " << q
                       << " forced " << childForced
                       << " variation";
            for(const Move& fm : followUpFrom(chosen, 6)){
                std::cout << " " << (int)fm.first << " " << (int)fm.second;
            }
            std::cout << std::endl;
        }
    }

    // Debug mode's per-playout lines (see printPlayoutDebugLine) are NOT printed here -- they
    // stream individually, straight from playout()/updateEval(), the moment each one finishes,
    // rather than being batched up and dumped once per chunk. So they can appear interleaved
    // with, and outside of, any "analysis begin"/"analysis end" block; the GUI's reader just
    // recognizes "playout ..." lines wherever they show up and accumulates them itself.

    std::cout << "analysis end" << std::endl;
}
