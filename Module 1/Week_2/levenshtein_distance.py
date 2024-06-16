def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    
    d_matrix = [[0]*(n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        d_matrix[i][0] = i
    for j in range(n + 1):
        d_matrix[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            d_matrix[i][j] = min(d_matrix[i - 1][j] + 1,
                                 d_matrix[i][j - 1] + 1,
                                 d_matrix[i - 1][j - 1] + (0 if s1[i - 1] == s2[j - 1] else 1))
            
    return d_matrix[m][n]

print(levenshtein_distance('kitten', 'sitting'))
print(levenshtein_distance('yu','you'))
print(levenshtein_distance('flaw','lawn'))            
    
    