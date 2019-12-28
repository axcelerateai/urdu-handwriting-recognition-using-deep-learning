import numpy as np

def Levenshtein(labels=None, logits=None):
    """Ref: https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/"""
    
    size_x = len(labels) + 1
    size_y = len(logits) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if labels[x-1] == logits[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
                
    lev_dist = matrix[size_x - 1, size_y - 1]
    
    #return lev_dist
    return max(0, 1-lev_dist/size_x)*100
