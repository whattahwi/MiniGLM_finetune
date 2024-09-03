### TODO: Implement metrics Perplexity, Rouge-L, etc.

def lcs(str1, str2):
    # based on str1!
    numbers = []
    label = []
    for i in range(len(str1)):
        for j in range(len(str2)):
            if str2[j] ==str1[i]:
                numbers.append(j)
                label.append(i)
    
    max_lcs = [1 for _ in range(len(numbers))]
    for i in range(1, len(max_lcs)):
        for j in range(0, i):
            if numbers[j] < numbers[i] and max_lcs[j] + 1 > max_lcs[i] and label[i] != label[j]:
                max_lcs[i] = max_lcs[j] + 1
    
    # length = max(max_lcs)
    # index = max_lcs.index(length)
    # result = [str2[numbers[index]]]
    # for i in range(index-1, -1, -1):
    #     if max_lcs[i] == length - 1:
    #         result.append(str2[numbers[i]])
    #         length -= 1
    #         if length == 1:
    #             break
    # result = reversed(result)
    # print([i for i in result])
         
    return max(max_lcs)

def rouge_l(X, Y): # 模型生成的回答(Y)，参考答案(X)
    length = lcs(X, Y)
    beta = 10000
    Rlcs = length/len(X)
    Plcs = length/len(Y)
    return(((1 + beta**2)*Rlcs*Plcs) / (Rlcs + (beta**2)*Plcs))
    

###