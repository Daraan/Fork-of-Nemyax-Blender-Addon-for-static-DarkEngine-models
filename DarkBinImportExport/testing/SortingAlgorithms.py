# ====================
import numpy as np
from itertools import product as product_iter, chain, permutations, compress, tee
REL_OPT_IMPROVEMENT = 0.08 # THIS IS SET IN SORTING HELPER

def _calc_delta(N):
    from math import log, sqrt
    """
    Calculates the range of range(i-delta, i+delta)
    This is very experimental.
    
    For N < 300 the time is kinda neglible and everything should be covered.
    
    Some results:
    ('50= ', 432.86033742509426, 865.7206748501885, -432.86033742509426)
    ('100= ', 424.1518488382719, 424.1518488382719, 0.0)
    ('200= ', 397.0003917874827, 198.50019589374136, 198.50019589374136)
    ('300= ', 375.6602794332463, 125.22009314441542, 250.44018628883086)
    ('400= ', 358.97647419251854, 89.74411854812962, 269.2323556443889)
    ('600= ', 334.1161899889282, 55.68603166482137, 278.43015832410686)
    ('850= ', 312.11516905585603, 36.71943165363012, 275.3957374022259)
    ('1500= ', 276.18601811198664, 18.412401207465777, 257.77361690452085)
    
    """
    if N <= 200:
        return N, N
    intervall = log(N)**2/sqrt(N) * 200
    j_d = intervall * 100  / N
    return int(intervall - j_d), int(j_d)

def _OptCore(mat, order, access, i, j, l, cost, cur_i_cost):
    """
    Main function that is used in every step.
    
    Compares current to possible new cost. Swaps if new cost is less.
    """
    # need to know which i,j is higher, if equal contine
    if i < j:
        h = i
        low = j
    elif i == j:
        return cur_i_cost, cost
    else:
        low = i
        h = j
    
    # access[0][j] is right side of j-th row
    # access[1][i] is above side of i-th column
    cur_j_cost = access[0][j].sum() + access[1][j].sum()
    
    # The intersection is counted twice need to remove.
    # Somewhere is still a bug that the costs are wrong by 1.
    old_cost = cur_i_cost + cur_j_cost - access[1][low][h]

    # Sum of row and column in the upper triangle parts after a potential swap
    new_cost_j_int = mat[j,i:].sum()
    new_cost_j_ext = mat[:i,j].sum()
    new_cost_i_int = mat[i,j:].sum() 
    new_cost_i_ext = mat[:j,i].sum()
    new_cost = new_cost_i_ext + new_cost_i_int + new_cost_j_ext + new_cost_j_int - access[1][low][h] 

    if new_cost < old_cost:
        Swaps1.append((i,j))
        cost -= (old_cost-new_cost)
        order[i], order[j] = order[j], order[i]
        
        # Adjust i 
        cur_i_cost = new_cost_j_int + new_cost_j_ext - (access[1][low][h] if j < i else -mat[j][i])
        
        mat[:, i], mat[:, j] = mat[:, j], mat[:, i].copy()
        mat[i], mat[j] = mat[j], mat[i].copy()
        
        print(cur_i_cost, sum(access[0][i].tolist()) + sum(access[1][i].tolist()))
        
        if i > j:
            cur_i_cost += access[1][low][h] # old cross, not favored in j if i is higher
    
    # Could only return j with cost value and at the end of i, choose best j.
    return cur_i_cost, cost

    
def kOptAutoKernel(mat, mesh, order, access):
    l = len(order)
    upper_delta , lower_delta = _calc_delta(l)
    cost = start_cost = sum(a.sum() for a in access[0])
    for i in range(l):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        for j in range(max(0,i-lower_delta),min(l,i+upper_delta)): # This could be changed to some range i-delta, i+delta, but bit dependant on object
            cur_i_cost, cost = _OptCore(mat, order, access, i, j, l, cost, cur_i_cost)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)
        
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return (start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT) if cost != 0 else True, mat
    return start_cost == cost, mat

# 20,5 better than auto for high poly
def kOptKernel(mat, mesh, order, access, upper=20, lower=5):
    """
    Most swaps are performed near the original location. 
    
    In some situations not but 
    generally j>i-delta is a good restriction for j
    """
    print("Max", REL_OPT_IMPROVEMENT)
    l = len(order)
    upper_delta = int(l * upper / 100) # 20% 
    lower_delta = int(l * lower / 100) # 5% margin
    cost = start_cost = sum(a.sum() for a in access[0])
    for i in range(l):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        for j in range(max(0,i-lower_delta),min(l,i+upper_delta)): # This could be changed to some range i-delta, i+delta, but bit dependant on object
            cur_i_cost, cost = _OptCore(mat, order, access, i, j, l, cost, cur_i_cost)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l, " / " , l)
        
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return (start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT) if cost != 0 else True, mat
    return start_cost == cost, mat



def _OptCore4(mat, order, access, i, j, allcosts, old_i_cost):
    """
    Main function that is used in every step.
    
    Compares current to possible new cost. Swaps if new cost is less.
    """
    # need to know which i,j is higher, if equal contine
    if i < j:
        h = i
        low = j
        # Sum of row and column in the upper triangle parts after a potential swap
        new_cost_j = allcosts[j] + mat[j,i:j].sum() - mat[i:j,j].sum()
    elif i == j:
        return 0, 0 # 2nd 0 is not relevant
    else:
        low = i
        h = j
        new_cost_j = allcosts[j] - mat[j,j:i].sum() + mat[j:i,j].sum()
   
    # The intersection is counted twice need to remove.
    old_cost = allcosts[i] + allcosts[j] - access[1][low][h]
       
    new_cost =  old_i_cost + new_cost_j - access[1][low][h] 
    
    return new_cost - old_cost, new_cost_j



# 20,5 better than auto for high poly
def kOptKernel4(mat, mesh, order, access, upper=20, lower=5):
    """
    Most swaps are performed near the original location. 
    
    In some situations not but generally 
    j > i - delta is a good restriction for j
    
    Needs less than N^2 / 4 loops.
    
    ---
    
    Using steepest descent, best possible swap for i.
    
    """
    print("Max Rel.", REL_OPT_IMPROVEMENT)
    l = len(order)
    upper_delta = int(l * upper / 100) # 20% 
    lower_delta = int(l * lower / 100) # 5% margin
    cost = start_cost = sum(a.sum() for a in access[0]) # = np.sum(np.triu(mat))
    
    allcosts = [access[0][j].sum() + access[1][j].sum() for j in range(l)]
    
    for i in range(l):
    #for i in range(l):
        
        # These are to calculate the new i cost step by step depending on j.
        # On a full matrix this would be the complete ith row
        # And starting from 0 the ith column
        # when working with 'kernels' already reduce the i-th row until j
        # and add the first j values in the ith column.
        # Adjustment is done at the start of the loop  +- Start offset for i==j==0
        i_r_val = mat[i, max(0, i - lower_delta):].sum() + mat[i, max(0, i - lower_delta)-1]
        i_c_val = mat[:max(0, i - lower_delta), i].sum() - mat[max(-1, i - lower_delta-1), i]
        
        best = 0
        bestj = None
        for j in range(max(0, i - lower_delta), min(l, i + upper_delta)): # This could be changed to some range i-delta, i+delta, but bit dependant on object
            i_r_val -= mat[i, j-1]
            i_c_val += mat[j-1, i]
            old_i_cost = i_c_val + i_r_val
            dif, new_j_cost = _OptCore4(mat, order, access, i, j, allcosts, old_i_cost)
            if dif < best:
                best = dif
                bestj = j
                new_bestj_cost = old_i_cost
                new_besti_cost = new_j_cost

        #print("Optimizing", i,"/", l, end='\r')
        if bestj is not None:            
            mat[:, i], mat[:, bestj] = mat[:, bestj], mat[:, i].copy()
            mat[i], mat[bestj] = mat[bestj], mat[i].copy()
            
            # Need to cope with intersection when adjusting
            if i > bestj:
                high = i
                low = bestj
                allcosts[i] = new_besti_cost -mat[i,bestj] + mat[bestj,i]
                allcosts[bestj] = new_bestj_cost
            else:
                high = bestj
                low = i
                allcosts[i] = new_besti_cost
                allcosts[bestj] = new_bestj_cost + mat[i, bestj] - mat[bestj,i]
            
            # Update costs for all others
            for k in range(low+1, high):
                allcosts[k] += mat[low,k] + mat[k,high] - mat[high,k] - mat[k, low]           
        print("Optimizing", i,"/", l, end='\r')                        
    print("Optimized", l, " / " , l)
    print("End cost", np.sum(np.triu(mat)))
    # More efficient:
    return (cost == 0 or (start_cost/cost > 1.0 + REL_OPT_IMPROVEMENT)), mat # 0 division



# =====================
def _OptCoreGradient(mat, order, access, i, jmin, jmax, l, cost):
    """
    This one is not good
    """
    cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
    i_r_val = np.sum(mat[i]) + mat[i, -1]     # Start offset for i==j==0
    i_c_val = 0              - int(mat[-1, i])
    # Look for best improvement
    cost_change = 0
    choosen_j = None
    for j in range(max(0,jmin), min(l,jmax)): 
        if i < j:
            h = i
            g = j
        elif i == j:
            i_r_val -= mat[i, j-1]
            i_c_val += mat[j-1, i]
            continue
        else:
            g = i
            h = j
        # The intersection is counted twice need to remove.
        cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
        old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
        # These are bit faster than the internal interesting.
        new_cost_j_int = sum(mat[j,i:].tolist())
        new_cost_j_ext = sum(mat[:i,j].tolist())
        i_r_val -= mat[i, j-1]
        i_c_val += mat[j-1, i]
        new_cost = i_c_val + i_r_val +new_cost_j_ext+new_cost_j_int-access[1][g][h]
        if new_cost - old_cost < cost_change:
            cost_change = new_cost - old_cost
            choosen_j = j

    if choosen_j != None:
        Swaps.append((i,choosen_j))
        cost += cost_change
        order[i], order[choosen_j] = order[choosen_j], order[i]
        mat[:, i], mat[:, choosen_j] = mat[:, choosen_j], mat[:, i].copy()
        mat[i], mat[choosen_j] = mat[choosen_j], mat[i].copy()
        #itered.append(choosen_j)
    return cost
def kOptKernelGradient(mat, mesh, order, access, upper=20, lower=5):
    """
    This one is not good. Need to use high margins => loose speed
    """
    l = len(order)
    upper_delta = int(l * upper / 100)
    lower_delta = int(l * lower / 100)
    cost = start_cost = sum(a.sum() for a in access[0])
    for i in range(l):
        cost = _OptCoreGradient(mat, order, access, i, jmin=i-lower_delta, jmax=i+upper_delta, l=l, cost=cost)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return (start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT) if cost != 0 else True, mat
    return start_cost == cost, mat

# ====================
SwapsK3 = []
def kOpt3(mat, mesh, order, access):
    l = len(order)
    cost = start_cost = sum(a.sum() for a in access[0])  
    for i in range(l // 2):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        vals = [i,0]
        for j in range(l // 2):
            k = j + l//2
            cur_k_cost = sum(access[0][k].tolist()) + sum(access[1][k].tolist())
            old_ik_cost = cur_i_cost + cur_k_cost - access[1][k][i]
            new_cost_ik_int = sum(mat[i,k:].tolist())
            new_cost_ik_ext = sum(mat[:k,i].tolist())
            new_cost_ki_int = sum(mat[k,i:].tolist()) 
            new_cost_ki_ext = sum(mat[:i,k].tolist())
            new_ik_cost = new_cost_ik_ext + new_cost_ik_int+new_cost_ki_ext+new_cost_ki_int-access[1][k][i]
            dif_ik = old_ik_cost - new_ik_cost
            if i == j:
                if dif_ik <= 0:
                    continue # New cost not better
                do = 1
            else:
                vals[0] = i
                vals[1] = j
                vals.sort()
                g = vals[1]
                h = vals[0]
                cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
                old_ij_cost = cur_i_cost + cur_j_cost - access[1][g][h]
                new_cost_jk_int = sum(mat[j,k:].tolist())
                new_cost_jk_ext = sum(mat[:k,j].tolist())
                new_cost_kj_int = sum(mat[k,j:].tolist()) 
                new_cost_kj_ext = sum(mat[:j,k].tolist())
                new_cost_ji_int = sum(mat[j,i:].tolist())
                new_cost_ji_ext = sum(mat[:i,j].tolist())
                new_cost_ij_int = sum(mat[i,j:].tolist()) 
                new_cost_ij_ext = sum(mat[:j,i].tolist())
                old_jk_cost = cur_j_cost + cur_k_cost - access[1][k][j]
                new_ij_cost = new_cost_ij_ext + new_cost_ij_int+new_cost_ji_ext+new_cost_ji_int-access[1][g][h]
                new_jk_cost = new_cost_jk_ext + new_cost_jk_int+new_cost_kj_ext+new_cost_kj_int-access[1][k][j]
                dif_ij = old_ij_cost - new_ij_cost
                dif_jk = old_jk_cost - new_jk_cost
                # New costs must be less than old
                res = (dif_ij > 0, dif_ik > 0, dif_jk > 0)
                s = res.count(True)
            
                if s == 0:
                    continue # Nothing better
                if s == 1:
                    if res[1]:
                        do = 1
                    elif res[2]:
                        do = 2
                    else:
                        do = 0
                # Now two methods s == 2
                else:
                    # We can swap one or all
                    diffs = [dif_ij, dif_ik, dif_jk]
                    diffs.sort()
                    most = diffs[2]
                    if most == diffs[1]:
                        do = 1
                    elif most == diffs[2]:
                        do = 2
                    else:
                        do = 0
                if False and s == 3:
                    print("All 3 good")
                    if False:
                        # All 3 are good
                        # Now swap two or all three
                        all = dif_ij + dif_ik + dif_jk
                        # i->k->j or i->j->k OR i<->k or i<->j or j<->k
                        # now: IJK vs KIJ vs JKI
                        diffs = [dif_ij, dif_ik, dif_jk]
                        diffs.sort()
                        most = diffs[2]
                        # i->j->k
                        dif_ik + dif_jk
                        # this gets complicated

            if do == 1:
                s1 = i
                s2 = k
                cost -= (old_ik_cost-new_ik_cost)
            elif do == 2:
                s1 = j
                s2 = k
                cost -= (old_jk_cost-new_jk_cost)
            else:
                s1 = i
                s2 = j
                cost -= (old_ij_cost-new_ij_cost)
            SwapsK3.append((s1,s2))
            order[s1], order[s2] = order[s2], order[s1]
            mat[:, s1], mat[:, s2] = mat[:, s2], mat[:, s1].copy()
            mat[s1], mat[s2] = mat[s2], mat[s1].copy()
            if s1 == i:
                cur_i_cost = sum(access[0][s2].tolist()) + sum(access[1][s2].tolist())
            
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)     
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return (start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT) if cost != 0 else True, mat
    return start_cost == cost, mat

# ====================
Swaps = [] 
Swaps1 = []
def fo4(mat, mesh, order, access):
    start_cost = sum(a.sum() for a in access[0])
    l = len(order)
    cost = start_cost        
    for i in range(l):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        for j in range(l):
            cur_i_cost, cost = _OptCore(mat, order, access, i, j, l, cost, cur_i_cost)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)
        
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return (start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT) if cost != 0 else True, mat
    return start_cost == cost, mat

# ====================

class _compressiter(list):
    def __init__(self):
        self.i = -1
    
    def __next__(self):
        #Stops as compress stops when iter is done
        self.i += 1
        return self.i not in self
    
    def __iter__(self):
        return self


def fo5(mat, mesh, order, access):
    """
    Similar to steepest gradient method
    """
    start_cost = cost = sum(a.sum() for a in access[0])
    l = len(order)
    itered = _compressiter()
    for i in range(l):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        i_r_val = np.sum(mat[i]) + mat[i, -1]     # Start offset for i==j==0
        i_c_val = 0              - int(mat[-1, i])
        
        # Look for best improvement
        cost_change = 0
        choosen_j = None
        for j in range(l):
            if i < j:
                h = i
                g = j
            elif i == j:
                i_r_val -= mat[i, j-1]
                i_c_val += mat[j-1, i]
                continue
            else:
                g = i
                h = j
            # The intersection is counted twice need to remove.
            cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
            old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
            # These are bit faster than the internal interesting.
            new_cost_j_int = sum(mat[j,i:].tolist())
            new_cost_j_ext = sum(mat[:i,j].tolist())
            i_r_val -= mat[i, j-1]
            i_c_val += mat[j-1, i]
            
            new_cost = i_c_val + i_r_val +new_cost_j_ext+new_cost_j_int-access[1][g][h]
            if new_cost - old_cost < cost_change:
                cost_change = new_cost - old_cost
                choosen_j = j
             
        if choosen_j != None:
            Swaps.append((i,choosen_j))
            cost += cost_change
            order[i], order[choosen_j] = order[choosen_j], order[i]
            mat[:, i], mat[:, choosen_j] = mat[:, choosen_j], mat[:, i].copy()
            mat[i], mat[choosen_j] = mat[choosen_j], mat[i].copy()
            itered.append(choosen_j)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return (start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT) if cost != 0 else True, mat
    return start_cost == cost, mat

Swaps2 = []
def fo52(mat, mesh, order, access):
    """
    Similar to steepest gradient method
    """
    start_cost = cost = sum(a.sum() for a in access[0])
    l = len(order)
    itered = _compressiter()
    for i in range(l):#compress(range(l), itered):
        cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
        i_r_val = np.sum(mat[i])
        i_c_val = 0
        
        # Look for best improvement
        cost_change = 0
        choosen_j = None
        for j in range(l):
            if i < j:
                h = i
                g = j
            elif i == j:
                continue
            else:
                g = i
                h = j
            # The intersection is counted twice need to remove.
            cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
            old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
            # These are bit faster than the internal interesting.
            new_cost_j_int = sum(mat[j,i:].tolist())
            new_cost_j_ext = sum(mat[:i,j].tolist())
            new_cost_i_int = sum(mat[i,j:].tolist()) 
            new_cost_i_ext = sum(mat[:j,i].tolist())
            new_cost = new_cost_i_ext+new_cost_i_int+new_cost_j_ext+new_cost_j_int-access[1][g][h]
            if new_cost - old_cost < cost_change:
                cost_change = new_cost - old_cost
                choosen_j = j
             
        if choosen_j != None:
            Swaps2.append((i,choosen_j))
            cost += cost_change
            order[i], order[choosen_j] = order[choosen_j], order[i]
            mat[:, i], mat[:, choosen_j] = mat[:, choosen_j], mat[:, i].copy()
            mat[i], mat[choosen_j] = mat[choosen_j], mat[i].copy()
            itered.append(choosen_j)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT if cost != 0 else True, mat
    return start_cost == cost, mat

# ====================

class _compressiter2(list):
    def __init__(self, l):
        self.i = -1
        self.len = l-1
    
    def __next__(self):
        self.i += 1
        return self.i//self.len not in self
    
    def __iter__(self):
        return self

def fo6(mat, mesh, order, access):
    """
    Similar to steepest gradient method
    """
    cost = start_cost = sum(a.sum() for a in access[0])
    # (0,0) is not done
    cur_i_cost = sum(access[0][0].tolist()) + sum(access[1][0].tolist())
    cost_change = 0
    choosen_j = None
    l = len(order)
    compressor = _compressiter2(l)
    for i,j in compress(permutations(range(l),2), compressor):
        if j == 0:
            cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
            # Look for best improvement
            cost_change = 0
            choosen_j = None
            g = i
            h = 0
        elif i < j:
            h = i
            g = j
        else:
            h = j
        # The intersection is counted twice need to remove.
        cur_j_cost = sum(access[0][j].tolist()) + sum(access[1][j].tolist())
        old_cost = cur_i_cost + cur_j_cost - access[1][g][h]
        # These are bit faster than the internal interesting.
        new_cost_j_int = sum(mat[j,i:].tolist())
        new_cost_j_ext = sum(mat[:i,j].tolist())
        new_cost_i_int = sum(mat[i,j:].tolist()) 
        new_cost_i_ext = sum(mat[:j,i].tolist())
        new_cost = new_cost_i_ext+new_cost_i_int+new_cost_j_ext+new_cost_j_int-access[1][g][h]
        if new_cost - old_cost < cost_change:
            cost_change = new_cost - old_cost
            choosen_j = j
        # At end of one cycle  
        if j == l-1 and choosen_j != None:
            Swaps.append((i, choosen_j))
            cost += cost_change
            order[i], order[choosen_j] = order[choosen_j], order[i]
            mat[:, i], mat[:, choosen_j] = mat[:, choosen_j], mat[:, i].copy()
            mat[i], mat[choosen_j] = mat[choosen_j], mat[i].copy()
            cur_i_cost = sum(access[0][i].tolist()) + sum(access[1][i].tolist())
            compressor.append(choosen_j)
        print("Optimizing", i,"/", l, end='\r')
    print("Optimizing", l,"/", l)
    print("End cost", cost, cost == np.sum(np.triu(mat)), np.sum(np.triu(mat)))
    # More efficient:
    return start_cost/cost <= 1.0 + REL_OPT_IMPROVEMENT if cost != 0 else True, mat
    return start_cost == cost, mat