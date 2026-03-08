#include "gamerules.h"
#include "MCTS.h"
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>

const static float eps = 0.0001f;

Node::Node(Game* g): game(g){
    turn = game->getTurn();
    winCount = 0;
    searchCount = 0;
    initEval = 0.0f; // add heuristic
    winmove = {rowSize, colSize};
}

void Node::expand(){
    Color clr;
    
    //game->getBoardStatus();

    for(int i=0; i<rowSize; ++i){
        for(int j=0; j<colSize; ++j){
            if(game->isLegal(i, j)){
                //std::cout << "legal move : " << i << " " << j << std::endl;
                Game* ng = new Game(*game);
                clr = ng->makeMove({i, j});

                if(clr == EMPTY){
                    child.push_back(new Node(ng));
                    searchPref.push_back(0.0f);
                    legal.push_back({i, j});
                }

                else if(clr == turn){
                    delete ng;
                    winmove = {i, j};
                    goto afterloop;
                }
            }
        }
    }

afterloop:
    if(searchCount == 0){
        //std::cout << "expand delete " << std::endl;
        delete game;
    }
}

void Node::deletetree(){
    //std::cout << "delete tree " << std::endl;
    for(Node* i : child){
        i->deletetree();
        delete i;
    }
}

Node::~Node(){
    //std::cout << searchCount << std::endl;
    if(searchCount == 0)
        delete game;
}

int Node::searchandPropagate(){
    //std::cout << "search and propagate : " << std::endl;
    if(searchCount == 0)
        expand();
    searchCount++;
    
    if(winmove.first < rowSize){
        return 1;
    }
    if(legal.empty()){
        winCount++;
        return 0;
    }

    int maxi;
    float maxval = -1.0f;
    for(int i=0; i<legal.size(); ++i){
        searchPref[i] = child[i]->winCount / (float)(child[i]->searchCount + eps) +
            1.4f * sqrt(log(searchCount) / (child[i]->searchCount + eps));
        
        if(maxval < searchPref[i]){
            maxval = searchPref[i];
            maxi = i;
        }
    }

    int r = child[maxi]->searchandPropagate();
    winCount += r;
    return 1 - r;
}

std::pair<int, int> Node::selectMove(){
    if(winmove.first < rowSize)
        return winmove;
    if(legal.empty()){
        return {-1, -1};
    }

    int maxi;
    int maxval = -1;
    for(int i=0; i<legal.size(); ++i){
        if(child[i]->searchCount > maxval){
            maxval = child[i]->searchCount;
            maxi = i;
        }
    }

    for(int i=0; i<legal.size(); ++i){
        std::cout << "move : " << legal[i].first << " " << legal[i].second << " sc: " << child[i]->searchCount << " wc: " << child[i]->winCount << std::endl;
    }
    return legal[maxi];
}

Node* Node::jump(std::pair<int, int> move){
    if(searchCount == 0){
        expand();
        searchCount++;
    }

    int idx = -1;
    Node* next;

    //std::cout << "# of possible moves " << legal.size() << std::endl;
    for(int i=0; i<legal.size(); ++i){
        if(legal[i] == move){
            idx = i;
            next = child[idx];
        }

        else{
            //std::cout << i << " ";
            child[i]->deletetree();
            delete child[i];
        }
    }

    child.clear();
    return next;
}

void MCTS::runSimulation(int simulCount){
    for(int i=0; i<simulCount; ++i){
        // if(i % (simulCount / 10) == 0)
        //     std::cout << i << std::endl;
        root->searchandPropagate();
    }
}

std::pair<int, int> MCTS::getMove(){
    return root->selectMove();
}

void MCTS::jump(std::pair<int, int> move){
    Node* nroot = root->jump(move);
    //std::cout << "deleting root " << std::endl;
    delete root;
    root = nroot;
}


MCTS::MCTS(Node* root){
    this->root = root;
}

MCTS::MCTS(){
    this->root = new Node(new Game());
}