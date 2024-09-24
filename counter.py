
# counter.py

class Counter:
    def __init__(self):
        self.counter = 0

    def print_with_counter(self, message):
        print(f"[{self.counter}] {message}")
        self.counter += 1
