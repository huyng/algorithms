def quicksort(L):
    if not L:
        return []
        
    pivot         = L[0]
    less_stack    = [i for i in L if i < pivot]
    greater_stack = [i for i in L if i >= pivot]
    return quicksort(less_stack) + [pivot] + quicksort(greater_stack)


def main():
    print quicksort([1, 4, 10, 11, 15, 3, 10 , 11, 2, 1, 5, 9, 1001, 1,1,10])

if __name__ == '__main__':
    main()
