#include <iostream>
#include "../consts.h"
#include "../hash.h"
#include "../gamerules.h"

int main(){
    Hash hash;
    HashValue h = hash.baseHash();
    Game g = Game();
    while(true){
        int r, c;
        std::cout << "Enter move (row col): ";
        std::cin >> r >> c;
        if(r == -1 && c == -1) break;
        if(!g.isLegal(r, c)){
            std::cout << "Illegal move. Try again." << std::endl;
            continue;
        }
        h = hash.computeHashAfterMove(g, {r, c}, h);
        g.makeMove(r, c);
        std::cout << "New hash: " << h << std::endl;
    }
}