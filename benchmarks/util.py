# util.py - Static utility methods.
#
# Take 'em or leave 'em.
from hashlib import sha1
from os.path import abspath,basename,dirname,exists

# Return the checksum of file at "path".
def checksum(path):
    return sha1(open(path, 'rb').read()).hexdigest()

# Concatenate all components into a path.
def path(*components):
    return abspath('/'.join(components))
