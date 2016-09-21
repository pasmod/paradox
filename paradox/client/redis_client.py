import config
import redis
import os


class RedisClient():
    def __init__(self, lang='de'):
        self.redis = redis.StrictRedis(
            host=os.environ['DB_PORT_6379_TCP_ADDR'],
            port=os.environ['DB_PORT_6379_TCP_PORT'],
            db=config.REDIS_DBS[lang])
        self.lang = lang

    def exists(self, key):
        """Check if the key exists in the database

        # Arguments
            key: key to be checked

        # Returns
            exists: True if the key exists in the database
        """
        return self.redis.exists(key)

    def keys(self):
        """Returns all keys in the database

        # Returns
            keys: all keys in the database
        """
        return self.redis.keys()

    def get(self, key):
        """Returns the values of a given keys as dictionary

        # Arguments
            key: key to be retrieved

        # Returns
            dictionary: value of the keys in form of a dictionary
        """
        return self.redis.hgetall(key)
