def find_longest_diag(array2D, patternlength):
    res=[]
    for k in range(-array2D.shape[0]+1, array2D.shape[1]):
        diag=np.diag(array2D, k=k)
        if(len(diag)>=patternlength):
            for i in range(len(diag)-patternlength+1):
                if(all(diag[i:i+patternlength]==1)):
                    res.append((i+abs(k), i) if k<0 else (i, i+abs(k)))
    return res