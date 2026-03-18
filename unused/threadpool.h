#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <thread>
#include <functional>
#include <atomic>
#include "MPMCQueue.h"  // Include MPMCQueue

class ThreadPool {
public:
    explicit ThreadPool(int num_threads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);
    void waitForAll();

private:
    std::vector<std::thread> workers;
    rigtorp::MPMCQueue<std::function<void()>> tasks;  // MPMC queue for task storage
    std::atomic<bool> stop;
    std::atomic<int> total_tasks;  // Tracks remaining tasks
};

#endif // THREADPOOL_H
