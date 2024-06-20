from abc import ABC, abstractmethod

# (a) 
class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob
        
    @abstractmethod
    def describe(self):
        pass 

class Student(Person):
    def __init__(self, name: str, yob: int, grade: str):
        super().__init__(name, yob)
        self.__grade = grade
        
    def describe(self):
        print(f"Student - Name: {self._name} - YoB: {self._yob} - Grade: {self.__grade}")
        
class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self.__specialist = specialist
    
    def describe(self):
        print(f"Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self.__specialist}")

class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)
        self.__subject = subject
        
    def describe(self):
        print(f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self.__subject}")

# (a) testcases   
student1 = Student(name="studentA", yob=2010, grade="7")
student1.describe()
teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher1.describe()
doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor1.describe()

# (b) & (c) & (d) & (e)
class Ward:
    def __init__(self, name):
        self.__name = name
        self.__lst = []
        
    def add_person(self, person):
        self.__lst.append(person)
        
    def describe(self):
        print(f"Ward Name: {self.__name}")
        for person in self.__lst:
            person.describe()
            
    def count_doctor(self):
        count = 0
        for person in self.__lst:
            if (isinstance(person, Doctor)):
                count += 1
        return count
    
    def sort_age(self):
        self.__lst.sort(key=lambda x: x._yob, reverse=True)
    
    def compute_average(self):
        count = sum(1 for person in self.__lst if isinstance(person, Teacher))
        total = sum(person._yob for person in self.__lst if isinstance(person, Teacher))
                
        return total / count
        
# (b) testcases   
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")

ward1 = Ward(name="Ward1")
ward1.add_person(student1)
ward1.add_person(teacher1)
ward1.add_person(teacher2)
ward1.add_person(doctor1)
ward1.add_person(doctor2)
ward1.describe()

# (c) testcases   
print(f"\nNumber of doctors: {ward1.count_doctor()}")

# (d) testcases 
print("\nAfter sorting Age of Ward1 people")
ward1.sort_age()
ward1.describe() 

# (e) testcases
print(f"\nAverage year of birth (teachers): {ward1.compute_average()}") 

