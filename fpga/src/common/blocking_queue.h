// Bounded FIFO used between the reader / compute / writer threads.
#ifndef ANTIUAV_BLOCKING_QUEUE_H
#define ANTIUAV_BLOCKING_QUEUE_H

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace antiuav {

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    // Returns false (item dropped) once close() was called.
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [&] { return items_.size() < capacity_ || closed_; });
        if (closed_) return false;
        items_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Returns std::nullopt once close() was called and the queue is drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [&] { return !items_.empty() || closed_; });
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return item;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    size_t capacity_;
    std::deque<T> items_;
    bool closed_ = false;
    std::mutex mu_;
    std::condition_variable not_empty_, not_full_;
};

}  // namespace antiuav

#endif  // ANTIUAV_BLOCKING_QUEUE_H
