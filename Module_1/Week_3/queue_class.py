class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.lst = []
        
    def is_empty(self):
        if self.lst == []:
            return True
        else:
            return False
    
    def is_full(self):
        if len(self.lst) == self.capacity:
            return True
        else:
            return False
        
    def dequeue(self):
        return self.lst.pop(0)
    
    def enqueue(self, value):
        self.lst.append(value)
        
    def front(self):
        return self.lst[0]
    
#Testcases
queue1 = Queue(capacity=5)
queue1.enqueue(1)
queue1.enqueue(2)

print(queue1.is_full())
print(queue1.front())
print(queue1.dequeue())
print(queue1.front())
print(queue1.dequeue())
print(queue1.is_empty())
