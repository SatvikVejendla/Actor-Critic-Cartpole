import json

class Memory:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def clear(self):
        self.data.clear()

    def sum(self):
        return sum(self.data)

    def save(self, path):
        with open(path, "w") as handle:
            handle.write(json.dumps(self.data, indent=4))
            
