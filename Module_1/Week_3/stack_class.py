
class Stack:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__lst = []
        
    def is_empty(self):
        if self.__lst == []:
            return True
        else:
            return False
        
    def is_full(self):
        if len(self.__lst) == self.__capacity:
            return True
        else:
            return False
        
    def pop(self):
        if self.is_empty():
            print("Underflow: Do nothing!")
        else:
            return self.__lst.pop(-1)
    
    def push(self, value):
        if self.is_full():
            print("Overflow: Do nothing!")
        else:
            self.__lst.append(value)
        
    def top(self):
        if self.is_empty():
            print("Stack is empty!")
            return
        return self.__lst[-1]
    
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



    