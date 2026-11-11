#include "analysis.hpp"
#include <iostream>
#include <sstream>

// score map and capture map of game's position. Neither is part of the search's evaluation (nor its cache), so both come
// from one forward pass of their own.
struct BoardMaps {
    std::vector<float> score; // tanh output in [-1, 1] per point
    std::vector<float> capture; // probability that the stone on each point is captured; empty without a capture head
};

static BoardMaps board_maps(PolicyValueNet* net, const Game& game){
    NNInput input = PolicyValueNet::get_data(game);
    PolicyValueOutput out = net->evaluate(input);
    BoardMaps maps{std::move(out.scoreMap), {}};

    const auto model = std::dynamic_pointer_cast<Net>(net->policyValueNet);
    if(!model || !model->captureHead)
        return maps;

    // captureMap holds logits. The head is trained with BCE weight w on captured points, which makes it learn
    // s = w p / (1 - p + w p) instead of p, so p = s / (w - (w - 1) s).
    const float w = PolicyValueNet::capturePosWeight;
    maps.capture.resize(boardSize);
    for(int i = 0; i < boardSize; ++i){
        const float s = 1.0f / (1.0f + std::exp(-out.captureMap[i]));
        maps.capture[i] = s / (w - (w - 1.0f) * s);
    }
    return maps;
}

// See analysis.hpp for what lives here and why it's split into its own class.

// optional "setup <komi> <nNeutrals> [r c]..." right after a request's id. Without it : config komi / neutrals.
// false on a malformed setup. iss is left at the first move either way.
static bool read_request_setup(std::istringstream& iss, Game& game){
    const auto pos = iss.tellg();
    std::string tok;
    if(!(iss >> tok) || tok != "setup"){
        iss.clear();
        iss.seekg(pos);
        game = Game();
        return true;
    }
    float komi;
    std::vector<Move> neutrals;
    if(!Game::read_setup(iss, komi, neutrals))
        return false;
    game = Game(komi, neutrals);
    return true;
}

void Analysis::nn_input_server(){
    std::string line;
    while(std::getline(std::cin, line)){
        std::istringstream iss(line);
        std::string cmd, id;
        iss >> cmd;
        if(cmd == "quit")
            break;
        if(cmd != "nninput" || !(iss >> id))
            continue;

        Game game;
        bool legal = read_request_setup(iss, game);
        int r, c;
        while(legal && iss >> r >> c){
            const Move m = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
            if(m == PASSMOVE || game.is_legal(m))
                game.make_move(m);
            else
                legal = false;
        }
        if(!legal){
            std::cout << "nninput " << id << " error" << std::endl;
            continue;
        }

        // The available-move list (and everything derived from it) is only filled in by
        // set_policy_mask(), which search runs on a node's first visit before the net
        // evaluates it -- apply it here too, to dump exactly what the net sees.
        game.set_policy_mask();
        const NNInput input = PolicyValueNet::get_data(game);
        std::cout << "nninput " << id << " " << input.size() / inputSize;
        for(float v : input)
            std::cout << " " << v;
        std::cout << std::endl;
    }
}

void Analysis::rules_server(){
    std::cout << "rules ready " << rowSize << " " << colSize << " " << globalConfig.komi << " " << globalConfig.neutrals.size();
    for(const auto& n : globalConfig.neutrals)
        std::cout << " " << static_cast<int>(n.first) << " " << static_cast<int>(n.second);
    std::cout << std::endl;

    std::string line;
    while(std::getline(std::cin, line)){
        std::istringstream iss(line);
        std::string cmd, id;
        iss >> cmd;
        if(cmd == "quit")
            break;
        if(cmd != "rules" || !(iss >> id))
            continue;

        Game game;
        if(!read_request_setup(iss, game)){
            std::cout << "rules " << id << " badsetup" << std::endl;
            continue;
        }
        Color winner = EMPTY;
        const char* status = "ongoing";
        bool prevPass = false;
        int ply = 0, illegalPly = -1;
        int r, c;
        while(iss >> r >> c){
            const Move m = {static_cast<uint8_t>(r), static_cast<uint8_t>(c)};
            // a move after the game already ended, an off-board point, or an occupied/territory point.
            if(winner != EMPTY || !(m == PASSMOVE || (r >= 0 && r < rowSize && c >= 0 && c < colSize && game.is_legal(m)))){
                illegalPly = ply;
                break;
            }
            // Game::make_move doesn't end on two passes in a row (the engine only lets the side
            // ahead on score pass), but a human can; two passes end the game by score.
            if(m == PASSMOVE && prevPass){
                winner = game.score_winner();
                status = "score";
            }
            else{
                const auto [w, type] = game.make_move(m);
                if(w != EMPTY){
                    winner = w;
                    status = (type == CAPTURE) ? "capture" : "score";
                }
            }
            prevPass = (m == PASSMOVE);
            ply++;
        }
        if(illegalPly >= 0){
            std::cout << "rules " << id << " illegal " << illegalPly << std::endl;
            continue;
        }

        // board : b/w stone, n neutral, B/W territory, . empty (legal) point.
        std::string board(boardSize, '.');
        int bTerr = 0, wTerr = 0;
        for(int i=0; i<rowSize; ++i){
            for(int j=0; j<colSize; ++j){
                char& ch = board[i * colSize + j];
                switch(game.get_board({i, j})){
                    case BLACK: ch = 'b'; break;
                    case WHITE: ch = 'w'; break;
                    case NEUTRAL: ch = 'n'; break;
                    default:
                        if(game.get_score_board({i, j}) == BSCORE){ ch = 'B'; bTerr++; }
                        else if(game.get_score_board({i, j}) == WSCORE){ ch = 'W'; wTerr++; }
                        break;
                }
            }
        }
        // get_turn() is the side to move after the last move -- except after a capture, where
        // Game::make_move returns before switching turns, so it is still the side that moved.
        std::cout << "rules " << id << " ok " << status << " " << ((winner == EMPTY) ? 0 : static_cast<int>(winner)) << " "
                  << static_cast<int>(game.get_turn()) << " " << game.score_diff(BLACK) << " "
                  << bTerr << " " << wTerr << " " << board << std::endl;
    }
}

void Analysis::print_variation(MCTS& mcts){
    Node* node = mcts.root;

    while(node->N > 1){
        int maxv = -1;
        int maxi = -1;
        int m;

        if(node->forcedState == 0){
            for(int i=0; i<node->game.get_available_moves().size(); ++i){
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

            m = node->game.get_available_moves()[maxi];
        }
        else if(node->forcedState > 0){
            for(int i=0; i<node->child.size(); ++i){
                if(node->child[i]->forcedState < 0){
                    m = node->game.get_available_moves()[i];
                }
            }
            // std::cerr << "only move : " << static_cast<int>(m.first) << " " << static_cast<int>(m.second) << std::endl;

            if(node->forcedState == 2)
                break;

            for(int i=0; i<node->game.get_available_moves().size(); ++i){
                if(node->game.get_available_moves()[i] == m){
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
            for(int i=0; i<node->game.get_available_moves().size(); ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }

            if(maxi == -1)
                break;
            m = node->game.get_available_moves()[maxi];
        }

        int visit = node->edgeN[maxi];
        node = node->child[maxi];
        if(node == nullptr)
            return;
        std::cout << (int)m / colSize << " " << (int)m % colSize << " " << visit << " forced " << node->forcedState << " Q: " <<
            node->W/node->value_n() << " initQ : " << node->initQ << " Wp : " << node->wp/node->value_n()
            << " S : " << node->S / node->value_n() << std::endl;
    }
}

std::vector<Move> Analysis::follow_up_from(Node* node, int maxDepth){
    std::vector<Move> seq;

    while(node != nullptr && node->N > 1 && (int)seq.size() < maxDepth){
        int maxv = -1;
        int maxi = -1;
        int m;

        if(node->forcedState == 0){
            for(int i=0; i<node->game.get_available_moves().size(); ++i){
                if(node->edgeN[i] > maxv){
                    maxv = node->edgeN[i];
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;
            m = node->game.get_available_moves()[maxi];
        }
        else if(node->forcedState > 0){
            // A node's forcedState can be set at birth (Node::addChild, when the move that
            // created it already ended the game) without ever going through expand() filling
            // in `child` -- expand() deliberately skips that for any already-forced node, so
            // `child` can be shorter than game.getAvailableMoves() (even empty) here. Bound by
            // the smaller of the two so we never index past either array; an empty/partial
            // child list just means no continuation to show, which the maxi==-1 check below
            // already handles gracefully.
            int limit = (int)node->child.size() < (int)node->game.get_available_moves().size()
                ? (int)node->child.size() : (int)node->game.get_available_moves().size();
            for(int i=0; i<limit; ++i){
                if(node->child[i] != nullptr && node->child[i]->forcedState < 0){
                    m = node->game.get_available_moves()[i];
                    maxi = i;
                    break;
                }
            }
            if(maxi == -1 || node->forcedState == 2)
                break;
        }
        else{ // losing. Delay losing as much as possible.
            int limit = (int)node->child.size() < (int)node->game.get_available_moves().size()
                ? (int)node->child.size() : (int)node->game.get_available_moves().size();
            for(int i=0; i<limit; ++i){
                if((node->child[i] != nullptr) && node->child[i]->forcedState > maxv){
                    maxv = node->child[i]->forcedState;
                    maxi = i;
                }
            }
            if(maxi == -1)
                break;
            m = node->game.get_available_moves()[maxi];
        }

        seq.push_back({static_cast<uint8_t>(m / colSize), static_cast<uint8_t>(m % colSize)});
        node = node->child[maxi];
    }

    return seq;
}

void Analysis::print_playout_debug_line(const std::vector<Node*>& path, float winP, float scoreEXP, int forcedState){
    std::cout << "playout forced " << forcedState << " winp " << winP << " score " << scoreEXP << " path";
    for(size_t i=0; i+1<path.size(); ++i){
        const std::vector<Node*>& child = path[i]->child;
        for(size_t a=0; a<child.size(); ++a){
            if(child[a] == path[i+1]){
                int m = path[i]->game.get_available_moves()[a];
                std::cout << " " << (m / colSize) << " " << (m % colSize);
                break;
            }
        }
    }
    std::cout << std::endl;
}

void Analysis::print_analysis(MCTS& mcts){
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
        std::cout << "winrate : " << (-root->wp / root->value_n()) << std::endl;
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
        std::cout << "scoreSearch : " << (-root->S / root->value_n()) << std::endl;
    else
        std::cout << "scoreSearch : 0" << std::endl;

    // scoreExp / scoreMap / captureMap are NN outputs for the *current* root position, not per-child
    // search stats. scoreExp comes from the evaluator (a cache hit, since root was already evaluated
    // by the very first playout); the cache doesn't keep the maps, so boardMaps runs its own pass.
    // scoreExp: the score head's raw scalar output -- net (Black - White) territory/capture
    // margin, from whoever is to move at root's perspective (turn-relative, same convention as
    // winrate/scoreMap below), without komi applied (see NodeInfo in evaluator.h and the
    // training label in train.cpp, `-scoreDiff(startingTurn)`).
    // scoreMap: tanh output in [-1, 1], matching the training label convention in
    // Game::makeMove's end-of-game scoreMap (-1 = Black-owned point, +1 = White-owned point).
    {
        auto buf = std::make_shared<NNResultBuf>();
        mcts.evaluator->evaluate(buf, &root->game, root->hashValue);
        std::cout << "scoreExp : " << buf->result->S << std::endl;

        const BoardMaps maps = board_maps(mcts.evaluator->model(), root->game);
        std::cout << "scoreMap :";
        for(float v : maps.score) std::cout << " " << v;
        std::cout << std::endl;
        // captureMap: probability that the stone on each point is captured (game ends at the first capture).
        // Only meaningful on stones. Not printed for a model without a capture head.
        if(!maps.capture.empty()){
            std::cout << "captureMap :";
            for(float v : maps.capture) std::cout << " " << v;
            std::cout << std::endl;
        }
    }

    // per-move breakdown is only meaningful once the root has been expanded with real moves.
    if(root->forcedState == 0){
        const auto& moves = root->game.get_available_moves();
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
                winrate = root->child[i]->wp / root->child[i]->N;
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
                for(const Move& fm : follow_up_from(root->child[i], 6)){
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
        const auto& moves = root->game.get_available_moves();
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
            for(const Move& fm : follow_up_from(chosen, 6)){
                std::cout << " " << (int)fm.first << " " << (int)fm.second;
            }
            std::cout << std::endl;
        }
        // An immediate win found by setPolicyMask on root's first visit (game.getWin()) marks
        // root forced-won without ever creating the winning child, so the search above finds
        // nothing -- same case Node::selectMove handles via getWin(). Without this line the GUI
        // gets an empty move list for a won position and its auto-play has nothing to play.
        else if(root->forcedState > 0 && root->game.get_win() != RESIGNMOVE){
            Move win = root->game.get_win();
            std::cout << "move " << (int)win.first << " " << (int)win.second
                       << " visits 0 prior 0 winrate 1 q 1 forced -1 variation" << std::endl;
        }
    }

    // Debug mode's per-playout lines (see printPlayoutDebugLine) are NOT printed here -- they
    // stream individually, straight from playout()/updateEval(), the moment each one finishes,
    // rather than being batched up and dumped once per chunk. So they can appear interleaved
    // with, and outside of, any "analysis begin"/"analysis end" block; the GUI's reader just
    // recognizes "playout ..." lines wherever they show up and accumulates them itself.

    std::cout << "analysis end" << std::endl;
}
