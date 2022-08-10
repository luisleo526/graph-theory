def hash_function(x):
    if x >= 0:
        return str(x)
    else:
        return f"919{abs(x)}"


class Invariant:
    def __init__(self, data):
        self.data = data
        self.hash_val = None

    def __hash__(self):
        hashes = []
        for poly in self.data:
            _hashes = ""
            for x in poly:
                _hashes += hash_function(x)
            hashes.append(_hashes)
        return hash(("-".join(hashes)))

    def __eq__(self, other):
        return hash(self) == hash(other)
