#include "threadpool.h"
#include <iostream>

ThreadPool::ThreadPool(int num_threads) 
    : stop(false), total_tasks(0), tasks(1024) {  // Initialize MPMCQueue with capacity 1024
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                if (!tasks.try_pop(task)) {  // Try to fetch a task
                    if (stop && tasks.empty()) return;  // Exit if stopping and no tasks left
                    std::this_thread::yield();  // Yield CPU to avoid busy-wait
                    continue;
                }

                task();  // Execute the task
                total_tasks.fetch_sub(1, std::memory_order_release);  // Decrement task count
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    stop = true;
    for (std::thread& worker : workers) {
        worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    total_tasks.fetch_add(1, std::memory_order_relaxed);  // Increment task count
    tasks.push(std::move(task));  // Push task into MPMC queue
}

void ThreadPool::waitForAll() {
    while (total_tasks.load(std::memory_order_acquire) > 0) {
        std::this_thread::yield();  // Avoid blocking, let CPU handle other work
    }
}
