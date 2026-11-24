#ifndef EVALCACHE_H
#define EVALCACHE_H

#include <vector>
#include <mutex>
#include <memory>
#include "consts.h"
#include "config.h"

template<typename T>
class Cache {
private:
    struct Entry {
        HashValue hash = 0;
        std::shared_ptr<T> ptr = nullptr;
    };

    int tableSize;
    int tableMask;

    Entry* table;

    int mutexPoolMask;
    std::vector<std::mutex> mutexPool;

    inline int index_of(HashValue h) const {
        return (int)h & tableMask;
    }
    inline std::mutex& mutex_of(int idx) {
        return mutexPool[idx & mutexPoolMask];
    }

public:
    explicit Cache(const CacheConfig& cfg) : tableSize(cfg.tableSize), tableMask(cfg.tableSize - 1),
        mutexPoolMask(cfg.mutexPoolSize - 1), mutexPool(cfg.mutexPoolSize){
        table = new Entry[tableSize];
    }

    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    ~Cache() {
        delete[] table;
    }

    bool get(HashValue h, std::shared_ptr<T>& out) {
        int idx = index_of(h);
        std::mutex& m = mutex_of(idx);
        std::lock_guard<std::mutex> lock(m);

        Entry& e = table[idx];
        if (e.ptr == nullptr || e.hash != h)
            return false;

        out = e.ptr;
        return true;
    }

    void insert(HashValue h, std::shared_ptr<T>& val) {
        int idx = index_of(h);
        std::mutex& m = mutex_of(idx);

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
        for(int i = 0; i < tableSize; i++) {
            std::mutex& m = mutex_of(i);
            std::lock_guard<std::mutex> lock(m);
            if(table[i].ptr != nullptr){
                table[i].ptr = nullptr;
            }
            table[i].hash = 0;
        }
    }
};


#endif