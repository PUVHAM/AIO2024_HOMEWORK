
class Stack:
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
        
    def pop(self):
        return self.lst.pop(-1)
    
    def push(self, value):
        self.lst.append(value)
        
    def top(self):
        return self.lst[-1]
    
#Testcases
stack1 = Stack(capacity=5)
stack1.push(1)
stack1.push(2)

print(stack1.is_full())
print(stack1.top())
print(stack1.pop())
print(stack1.top())
print(stack1.pop())
print(stack1.is_empty())



    