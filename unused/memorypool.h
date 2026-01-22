// Source - https://stackoverflow.com/a
// Posted by sehe, modified by community. See post 'Timeline' for change history
// Retrieved 2025-11-12, License - CC BY-SA 4.0

#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

#include <boost/pool/pool.hpp>
#include <cassert>

using Pool = boost::pool<boost::default_user_allocator_malloc_free>;

template <typename T> struct my_pool_alloc {
    using value_type = T;

    my_pool_alloc(Pool& pool) : _pool(pool) {
        assert(pool_size() >= sizeof(T));
    }

    template <typename U>
    my_pool_alloc(my_pool_alloc<U> const& other) : _pool(other._pool) {
        assert(pool_size() >= sizeof(T));
    }

    T *allocate(const size_t n) {
        T* ret = static_cast<T*>(_pool.ordered_malloc(n));
        if (!ret && n) throw std::bad_alloc();
        return ret;
    }

    void deallocate(T* ptr, const size_t n) {
        if (ptr && n) _pool.ordered_free(ptr, n);
    }

    // for comparing
    size_t pool_size() const { return _pool.get_requested_size(); }

  private:
    Pool& _pool;
};

template <class T, class U> bool operator==(const my_pool_alloc<T> &a, const my_pool_alloc<U> &b) { return a.pool_size()==b.pool_size(); }
template <class T, class U> bool operator!=(const my_pool_alloc<T> &a, const my_pool_alloc<U> &b) { return a.pool_size()!=b.pool_size(); }

#endif
