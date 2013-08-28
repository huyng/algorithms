quicksort :: Ord a => [a] -> [a]
quicksort []         = []
quicksort (pivot:xs) = (quicksort lesser) ++ [pivot] ++ (quicksort greater)
    where
        lesser  = filter (< pivot) xs
        greater = filter (>= pivot) xs
        
    
main = 
    print (quicksort [1 ,  2 ,  4 ,  5 ,  10])
