#ifndef EVALCACHE_H
#define EVALCACHE_H

#include <vector>
#include <mutex>
#include <memory>
#include "consts.h"

template<typename T>
class Cache {
private:
    struct Entry {
        HashValue hash = 0;
        std::shared_ptr<T> ptr = nullptr;
    };

    size_t tableMask;

    Entry* table;

    size_t mutexPoolMask;
    std::vector<std::mutex> mutexPool;

    inline size_t indexOf(HashValue h) const {
        return (size_t)h & tableMask;
    }
    inline std::mutex& mutexOf(size_t idx) {
        return mutexPool[idx & mutexPoolMask];
    }

public:
    Cache() : mutexPool(mutexPoolSize), tableMask(tableSize - 1), mutexPoolMask(mutexPoolSize - 1){
        table = new Entry[tableSize];
    }

    ~Cache() {
        delete[] table;
    }

    bool get(HashValue h, std::shared_ptr<T>& out) {
        size_t idx = indexOf(h);
        std::mutex& m = mutexOf(idx);
        std::lock_guard<std::mutex> lock(m);

        Entry& e = table[idx];
        if (e.ptr == nullptr || e.hash != h)
            return false;

        out = e.ptr;
        return true;
    }

    void insert(HashValue h, std::shared_ptr<T>& val) {
        size_t idx = indexOf(h);
        std::mutex& m = mutexOf(idx);

        // Local copy before locking
        auto buf = val;

        {
            std::lock_guard<std::mutex> lock(m);
            Entry& e = table[idx];

            e.hash = h;
            // swap to avoid freeing old value under lock
            e.ptr.swap(buf);
        }
        // Old value in buf is freed outside the lock
    }

    void clear() {
        for(size_t i = 0; i < tableSize; i++) {
            std::mutex& m = mutexOf(i);
            std::lock_guard<std::mutex> lock(m);
            if(table[i].ptr != nullptr){
                table[i].ptr = nullptr;
            }
            table[i].hash = 0;
        }
    }
};


#endif