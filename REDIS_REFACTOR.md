# Redis Refactor Investigation

## Overview

Investigation conducted on 2025-06-06 revealed that Redis support has been removed from the Poker Knight caching system, though API parameters and test infrastructure still reference it. This document captures all findings to facilitate future Redis implementation.

## Current State

### Architecture Reality
- **Actual Implementation**: Memory-only cache → SQLite persistence (optional)
- **No Redis Implementation**: Redis code paths have been completely removed
- **API Compatibility**: Redis parameters still accepted but ignored for backward compatibility
- **Fallback Chain**: The documented Redis → SQLite → Memory fallback doesn't exist; only SQLite → Memory

### Code Analysis

#### 1. Cache System Structure (`poker_knight/storage/`)

**cache.py**:
- `REDIS_AVAILABLE` flag only checks if redis library is imported (lines 16-20)
- `HandCache` class has `_redis_client = None` placeholder (line 45)
- `get_persistence_stats()` hardcoded to return `redis_connected: False` (line 201)
- Persistence type is always 'sqlite' or 'none', never 'redis' (line 204)

**unified_cache.py**:
- Redis parameters exist in function signatures but are unused:
  - `redis_host`, `redis_port`, `redis_db` in `get_unified_cache()` (lines 398-400)
  - Parameters are accepted but not passed to `ThreadSafeMonteCarloCache`

**CacheConfig dataclass** (cache.py):
- Still contains Redis configuration fields:
  ```python
  redis_host: Optional[str] = None
  redis_port: int = 6379
  redis_db: int = 0
  redis_ttl: int = 86400
  ```

#### 2. Test Infrastructure

**Test Files Referencing Redis**:
- `test_redis_integration.py` - Main Redis integration test suite
- `test_redis_simple.py` - Simple Redis functionality test
- `test_cache_with_redis_demo.py` - Redis cache demonstration
- `test_fallback_demo.py` - Fallback mechanism demonstration
- `test_sqlite_fallback_demo.py` - SQLite fallback test
- `test_integration_demo.py` - Contains Redis-related tests

**Test Patterns**:
```python
# Common pattern in tests
persistence_stats = cache.get_persistence_stats()
if not persistence_stats['redis_connected']:
    pytest.skip("Redis not available")
```

**conftest.py markers**:
- `@pytest.mark.redis_required` - Marker for tests requiring Redis
- `--redis` command line option for running Redis-specific tests

#### 3. Expected Redis Behavior (from tests)

**Performance Targets** (from test_redis_integration.py):
- Store operations: <50ms per operation
- Retrieval operations: <10ms per operation
- Batch operations support for pre-population

**Features Expected**:
- Persistent cache across application restarts
- Fast key-value storage for simulation results
- TTL support (86400 seconds default)
- Automatic fallback to SQLite when unavailable
- Docker Redis support on localhost:6379

## Implementation Clues

### 1. Key Generation
The system uses structured cache keys with normalization:
```python
# Format: "{hero_hand}_{num_opponents}_{board_cards}_{simulation_mode}"
# Example: "AA_2_preflop_default"
# Example: "AK_suited_3_As_Ks_Qs_precision"
```

### 2. Data Structure
Cache stores `CacheResult` objects containing:
- `win_probability`, `tie_probability`, `loss_probability`
- `simulations_run`, `execution_time_ms`
- `hand_categories` (hand category frequencies)
- `metadata` (additional metadata)
- Statistical convergence data

### 3. Integration Points

**Where Redis should integrate**:
1. `ThreadSafeMonteCarloCache.__init__()` - Initialize Redis client
2. `ThreadSafeMonteCarloCache.get()` - Check Redis before memory/SQLite
3. `ThreadSafeMonteCarloCache.put()` - Write-through to Redis
4. `ThreadSafeMonteCarloCache.clear()` - Clear Redis keys
5. `get_persistence_stats()` - Report actual Redis status

### 4. Persistence Hierarchy Design
Based on test expectations, the intended hierarchy should be:
1. **Memory Cache** (LRU, fastest)
2. **Redis** (if available, persistent, fast)
3. **SQLite** (fallback when Redis unavailable)
4. **Memory-only** (when persistence disabled)

## Future Implementation Considerations

### 1. Connection Management
- Lazy connection initialization
- Connection pooling for performance
- Graceful degradation on connection failure
- Health check mechanism

### 2. Serialization Strategy
- Current system uses dictionary serialization
- Consider msgpack or pickle for Redis storage
- Need to handle Unicode card symbols (♠♥♦♣)

### 3. Key Management
- Implement prefix for all Poker Knight keys (e.g., "pk:")
- TTL strategy for automatic cleanup
- Batch operations for warming/pre-population

### 4. Testing Requirements
- Mock Redis for unit tests
- Integration tests with real Redis (Docker)
- Performance benchmarks vs SQLite
- Failover scenario testing

### 5. Configuration Updates
```python
# Suggested configuration structure
class RedisConfig:
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl: int = 86400
    max_connections: int = 50
    connection_timeout: int = 5
    socket_timeout: int = 5
    retry_on_timeout: bool = True
```

### 6. Error Handling
- Connection errors → fallback to SQLite
- Serialization errors → log and skip Redis
- Network timeouts → configurable retry logic
- Full Redis → LRU eviction or error

## Migration Path

1. **Phase 1**: Implement Redis client in `ThreadSafeMonteCarloCache`
2. **Phase 2**: Add write-through caching (memory → Redis → SQLite)
3. **Phase 3**: Implement read hierarchy (memory ← Redis ← SQLite)
4. **Phase 4**: Add monitoring and metrics
5. **Phase 5**: Performance optimization and tuning

## Testing Strategy

### Unit Tests
- Mock redis-py client
- Test serialization/deserialization
- Test connection failure handling
- Test cache key generation

### Integration Tests
- Docker-based Redis instance
- Performance benchmarks
- Failover scenarios
- Concurrent access patterns

### Performance Tests
- Bulk operations (cache warming)
- Concurrent read/write
- Memory usage monitoring
- Network latency impact

## Dependencies

```python
# Required package
redis>=4.0.0  # Already checked in REDIS_AVAILABLE flag

# Optional for better performance
hiredis>=2.0.0  # C parser for better performance
msgpack>=1.0.0  # Efficient serialization
```

## Notes

- The codebase emphasizes "no external dependencies," but Redis was clearly planned
- Consider making Redis an optional feature installed with `pip install poker_knight[redis]`
- Document the performance/persistence tradeoff clearly
- Ensure Redis implementation doesn't break existing SQLite users