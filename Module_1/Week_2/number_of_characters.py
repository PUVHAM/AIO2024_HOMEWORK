def count_chars(string):
    dict_char = {}

    for char in string:
        if char in dict_char:
            dict_char[char] += 1
        else:
            dict_char[char] = 1

    return dict_char


def sort_dict_by_value_and_key(d):
    sorted_dict = dict(sorted(d.items(), key=lambda item: (item[1], item[0])))
    return sorted_dict


print(sort_dict_by_value_and_key(count_chars('Happiness')))
print(sort_dict_by_value_and_key(count_chars('smiles')))
