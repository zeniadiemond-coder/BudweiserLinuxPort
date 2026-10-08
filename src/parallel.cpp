#include "openv6/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace openv6 {

namespace {

struct Job {
    Job(int count, int chunk, const std::function<void(int, int)>& body) : count(count), chunk(chunk), body(body) {}

    int count, chunk;
    const std::function<void(int, int)>& body;
    std::atomic<int> next{0};
    int running = 0;
    std::exception_ptr error;
};

class Pool {
public:
    static Pool& instance() {
        static Pool pool;
        return pool;
    }

    int threads() const { return static_cast<int>(workers_.size()) + 1; }

    void run(Job& job) {
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back(&job);
            job.running++;
        }
        wake_.notify_all();
        drain(job);
        std::unique_lock lock(mutex_);
        finished_.wait(lock, [&] { return job.running == 0; });
        std::erase(jobs_, &job);
    }

private:
    Pool() {
        unsigned count = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 1; i < count; i++) workers_.emplace_back([this] { work(); });
    }

    ~Pool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    Job* claimable() const {
        for (Job* job : jobs_)
            if (job->next.load() < job->count) return job;
        return nullptr;
    }

    void work() {
        std::unique_lock lock(mutex_);
        while (true) {
            Job* job = nullptr;
            wake_.wait(lock, [&] { return stopping_ || (job = claimable()); });
            if (stopping_) return;
            job->running++;
            lock.unlock();
            drain(*job);
            lock.lock();
        }
    }

    void drain(Job& job) {
        try {
            for (int begin; (begin = job.next.fetch_add(job.chunk)) < job.count;) job.body(begin, std::min(begin + job.chunk, job.count));
        } catch (...) {
            job.next = job.count;
            std::lock_guard lock(mutex_);
            if (!job.error) job.error = std::current_exception();
        }
        std::lock_guard lock(mutex_);
        if (--job.running == 0) finished_.notify_all();
    }

    std::mutex mutex_;
    std::condition_variable wake_, finished_;
    std::vector<Job*> jobs_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

}

void parallel_for(int count, const std::function<void(int, int)>& body) {
    Pool& pool = Pool::instance();
    int chunk = std::max(1, count / (pool.threads() * 8));
    if (count <= chunk || pool.threads() == 1) {
        body(0, count);
        return;
    }
    Job job(count, chunk, body);
    pool.run(job);
    if (job.error) std::rethrow_exception(job.error);
}

}
