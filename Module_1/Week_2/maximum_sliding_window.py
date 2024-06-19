def max_sliding_window(lst, k):
    result = []
    for num in range(len(lst) - k + 1):
        max_num = max(lst[num:num + k])
        result.append(max_num)

    return result


num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
k = 3
print(max_sliding_window(num_list, k))
