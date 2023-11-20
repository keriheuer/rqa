import numpy as np

def find_longest_diag(self, array2D, patternlength):
    res=[]
    for k in range(-array2D.shape[0]+1, array2D.shape[1]):
        diag=np.diag(array2D, k=k)
        if(len(diag)>=patternlength):
            for i in range(len(diag)-patternlength+1):
                if(all(diag[i:i+patternlength]==1)):
                    res.append((i+abs(k), i) if k<0 else (i, i+abs(k)))
    return res

def find_vmax(self, distances, i, vmax):
  # find start positions of all vertical lines with length = V MAXß
  col = distances[i, :]
  idx_pairs = np.where(np.diff(np.hstack(([False],col==1,[False]))))[0].reshape(-1,2)
  lengths = np.diff(idx_pairs,axis=1) # island lengths
  if vmax in lengths:
    return (i, idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]) # longest island start position


def find_lmax(self, array2D, lmax):
  # find start position of diagonal with length = L MAX
    res = []
    for k in range(-array2D.shape[0]+1, array2D.shape[1]):
        diag = np.diag(array2D, k=k)
        if(len(diag) >= lmax):
            for i in range(len(diag) - lmax + 1):
                if(all(diag[i:i + lmax] == 1)):
                    res.append((i + abs(k), i) if k<0 else (i, i + abs(k)))
    return res

def transform_to_area(self, x, y1, y2):

    """ Helper function for filling between lines in bqplot figure."""

    return np.append(x, x[::-1]), np.append(y1, y2[::-1])

def exclude_diagonals(self, rp, theiler):

  # copy recurrence matrix for plotting
  rp_matrix = rp.recurrence_matrix()

  # clear pyunicorn cache for RQA
  rp.clear_cache(irreversible=True)

  # set width of Theiler window (n=1 for main diagonal only)
  n = theiler

  # create mask to access corresponding entries of recurrence matrix
  mask = np.zeros_like(rp.R, dtype=bool)

  for i in range(len(rp_matrix)):
    mask[i:i+n, i:i+n] = True

  # set diagonal (and subdiagonals) of recurrence matrix to zero
  # by directly accessing the recurrence matrix rp.R

  rp.R[mask] = 0

# check if cell generating fig2 was run, i.e. rp_plot is not empty
def validate(self):
    if self.rp_plot != None:
        return True
    else:
        return False