# jsoncache.py - Persistent store for JSON objects.
#
# There are two public methods: load() and store(). After loading from
# disk, JSON objects are cached locally, and written to disk every
# _CACHEWRITE_THRESHOLD writes, and when the process exits.
from atexit import register
from os import makedirs
from os.path import dirname

import json
import gitfs as fs

# Maximum number of writes to cache before writing to disk.
_CACHEWRITE_THRESHOLD = 10

# Maximum number of items to cache before evicting.
_MAX_CACHED_ITEMS = 50

_cachewrites = 0
_cachedirty = set()
_cache = {}

def _readjson(path):
    try:
        data = json.load(fs.markread(open(path)))
        return data
    except:
        return {}

def _writejson(path, data):
    try:
        makedirs(dirname(path))
    except:
        pass
    json.dump(data, fs.markwrite(open(path, 'w')),
              sort_keys=True, indent=2, separators=(',', ': '))

# Reduce total cache size by evicting elements.
def _squeezecache():
    # Random cache eviction strategy. Just iterate through the cache
    # keys until we've evicted enough entries. This could be improved
    # by a smarter caching strategy.
    i = 0
    for c in _cache.keys():
        evict(c)
        i += 1
        if i >= range(_MAX_CACHED_ITEMS / 2):
            return

def _loadcache(path):
    global _cache

    data = _readjson(path)
    _cache[path] = data

    if len(_cache) >= _MAX_CACHED_ITEMS:
        _squeezecache()

    return data

def _writedirty():
    global _cachedirty, _cachewrites

    for path in _cachedirty:
        _writejson(path, _cache[path])

    # Reset cache status.
    _cachewrites = 0
    _cachedirty = set()

def _dumpcache():
    # There may be nothing to dump
    if not _cachewrites:
        return
    _writedirty()

# Register exit handler.
register(_dumpcache)

def _flagdirty(path):
    global _cachedirty, _cachewrites

    _cachedirty.add(path)
    _cachewrites += 1
    if _cachewrites >= _CACHEWRITE_THRESHOLD:
        _dumpcache()

def load(path):
    return _cache[path] if path in _cache else _loadcache(path)

def store(path, data):
    _cache[path] = data
    _flagdirty(path)

#
def evict(path):
    global _cache,_cachedirty

    if path in _cachedirty:
        _writejson(path, _cache[path])
        _cachedirty.remove(path)

    del _cache[path]
