def exercise1(tp, fp, fn):
    #Check if tp, fp, fn is int or not
    check = {'tp': tp, 
             'fp': fp,
             'fn': fn}
    
    for key in check:
        if type(check[key]) != int:
            return f"{key} must be an int"
        if check[key] <= 0:
            return "tp and fp and fn must be greater than zero"
    
    #Calculate precision, recall, f1_score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f"precision is {precision}\nrecall is {recall}\nf1_score is {f1_score}"
    
#Testcases
print(exercise1(tp = 2, fp = 3, fn =4))
print(exercise1(tp = 2, fp = 'a', fn = 4))
print(exercise1(tp = 2, fp = 3, fn = 0))
print(exercise1(tp = 2.1, fp = 3, fn = 0))