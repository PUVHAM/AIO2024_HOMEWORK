
# (a) 
class Student:
    def __init__(self, name, yob, grade):
        self.name = name
        self.yob = yob
        self.grade = grade
        
    def describe(self):
        print(f"Student - Name: {self.name} - YoB: {self.yob} - Grade: {self.grade}")
        
class Doctor:
    def __init__(self, name, yob, specialist):
        self.name = name
        self.yob = yob
        self.specialist = specialist
        
    def describe(self):
        print(f"Doctor - Name: {self.name} - YoB: {self.yob} - Specialist: {self.specialist}")

class Teacher:
    def __init__(self, name, yob, subject):
        self.name = name
        self.yob = yob
        self.subject = subject
        
    def describe(self):
        print(f"Teacher - Name: {self.name} - YoB: {self.yob} - Subject: {self.subject}")

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
        self.name = name
        self.lst = []
        
    def add_person(self, person):
        self.lst.append(person)
        
    def describe(self):
        print(f"Ward Name: {self.name}")
        for person in self.lst:
            person.describe()
            
    def count_doctor(self):
        count = 0
        for person in self.lst:
            if (isinstance(person, Doctor)):
                count += 1
        return count
    
    def sort_age(self):
        self.lst.sort(key=lambda x: x.yob, reverse=True)
    
    def compute_average(self):
        count = sum(1 for person in self.lst if isinstance(person, Teacher))
        total = sum(person.yob for person in self.lst if isinstance(person, Teacher))
                
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

