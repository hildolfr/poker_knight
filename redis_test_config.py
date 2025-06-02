
# Redis fallback configuration for tests
import os

def check_redis_available():
    """Check if Redis is available and running."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, socket_timeout=1)
        client.ping()
        return True
    except:
        return False

def get_redis_config():
    """Get Redis configuration with fallback."""
    if os.environ.get('POKER_KNIGHT_DISABLE_REDIS', '0') == '1':
        return None
    
    if not check_redis_available():
        print("Redis not available - using memory-only cache")
        return None
    
    try:
        import redis
        return {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
    except ImportError:
        return None

# Environment variable to disable Redis for tests
os.environ['POKER_KNIGHT_DISABLE_REDIS'] = '1'
