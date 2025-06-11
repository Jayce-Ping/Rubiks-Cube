# The code is based on https://www.kaggle.com/code/vicensgaitan/minkwitz-10x-faster-with-pypy/output
from __future__ import annotations
from typing import List, Dict, Optional, Iterable, Union
import numpy as np
from itertools import chain, product, combinations
from sympy.combinatorics.perm_groups import Permutation, PermutationGroup
from sympy.combinatorics.perm_groups import _distribute_gens_by_base, _orbits_transversals_from_bsgs




# A class for maintainig permutations and factorization over a SGS
class PermWord:
    def __init__(self, perms=[], words=[]):
        self.words = words
        self.permutation = perms
        self.flag = True

    def inverse(self,geninvs):
        inv_perm = ~self.permutation
        inv_word = invwords(self.words, geninvs)
        return PermWord(inv_perm,inv_word)

    def __mul__(self, other : PermWord):
        result_perm = self.permutation * other.permutation
        result_words = simplify(self.words + other.words)
        return PermWord(result_perm, result_words)


# A class generating factorization of permutations over a Strong Generating Set (SGS)
# The SGS is obtained using the sympy implementation for Schreier–Sims algorith
# The Minkwith algorithm (https://www.sciencedirect.com/science/article/pii/S0747717198902024)
# is used for mantainig a short word representation 



class SGSPermutationGroup:
    def __init__(
            self,
            generators_dict : Dict[str, Permutation | List[int]],
            deterministic : bool = True,
            params : Dict[str, int] = {'n': 10000, 's': 2000, 'w': 20}
        ):
        self.N = max([max(generators_dict[g]) for g in generators_dict]) + 1

        generators_dict = {
            g : Permutation(generators_dict[g], size = self.N) if not isinstance(generators_dict[g], Permutation) else generators_dict[g]
            for g in generators_dict
        }
        generators = generators_dict.copy()
        generators_list= [generators_dict[p] for p in generators]
        
        geninvs = {}
        for s in list(generators.keys()):
            if s[0] != '-': 
                s1 = ~generators[s] # Inverse permutation
                generators["-" + s] = s1
                geninvs[s] = '-' + s
                geninvs['-' + s] = s

        self.gens = generators
        self.geninvs = geninvs
        # Create the permutation group
        #gen0= [gens[p] for p in gens]
        G = PermutationGroup(generators_list)
        self.G = G
        # obtain the strong generating set
        if deterministic:
            G.schreier_sims()
            self.base = G.base
            self.basic_orbits = G.basic_orbits
            self.basic_transversals = G.basic_transversals
        else:
            base,trans, orbits = schreier_sims_random(G)
            self.base = base
            self.basic_orbits = orbits
            self.basic_transversals = trans
        
     
        self.orbit_lens = [len(x) for x in self.basic_orbits]
        self.oribit_len_sum = np.sum(self.orbit_lens)
        self.getShortWords(**params)  # Initialize with default parameters

    #   n: max number of rounds
    #   s: reset each s rounds
    #   w: limit for word size
    
    def getShortWords(self, n : int =10000, s : int = 2000, w : int = 20):
        self.nu = buildShortWordsSGS(self.N, self.gens, self.geninvs, self.base, n, s, w, self.oribit_len_sum)

    def FactorPermutation(self, target : Permutation) -> List[str]:
        if self.nu == None:
            raise RuntimeError('Execute getShortWords first') 
           
        return factorizeM(self.N, self.gens, self.geninvs, self.base, self.nu, target)

    def CheckQuality(self):
        test = test_SGS(self.N, self.nu,self.base)
        qual = quality(self.N, self.nu, self.base)
        return test,qual

    def swapBase(self,i):
        S = self.G
        base, gens = S.baseswap(S.base, S.strong_gens, i, randomized=False)
        self.base = base

        
        
def schreier_sims_random(G):
    base, strong_gens = G.schreier_sims_random(consec_succ=5)
    strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    basic_orbits, transversals, slps = _orbits_transversals_from_bsgs(base,\
                strong_gens_distr, slp=True)

    # rewrite the indices stored in slps in terms of strong_gens
    for i, slp in enumerate(slps):
        gens = strong_gens_distr[i]
        for k in slp:
            slp[k] = [strong_gens.index(gens[s]) for s in slp[k]]

    transversals = transversals
    basic_orbits = [sorted(x) for x in basic_orbits]
    return base, transversals,basic_orbits
        
def applyPerm(sol, PG):
    if sol == []:
        return Permutation(size = PG.N)
    target = PG.gens[sol[0]]
    for m in sol[1:]:
        target = target * PG.gens[m]
    return target


def oneStep(N, gens, geninvs, base, i, t, nu):
    j = t.permutation.array_form[base[i]]  # b_i ^ t
    t1 = t.inverse(geninvs)
    if nu[i][j] is not None:
        result = t * nu[i][j]
        result.words = simplify(result.words)
        if len(t.words) < len(nu[i][j].words):
            nu[i][j] = t1
            oneStep(N, gens,geninvs, base, i, t1, nu)
    else:
        nu[i][j] = t1
        oneStep(N, gens, geninvs, base, i, t1, nu)
        result =  PermWord(Permutation(N),[])
    return result

def oneRound(N, gens,geninvs, base, lim, c, nu, t):
    i = c
    while i < len(base) and len(t.words)>0 and len(t.words) < lim:
        t = oneStep(N, gens, geninvs, base, i, t, nu)
        i += 1

def oneImprove(N, gens,geninvs, base, lim, nu):
    for j in range(len(base)):
        for x in nu[j]:
            for y in nu[j]:
                if x != None and y != None  and (x.flag or y.flag):
                    t = x * y
                    oneRound(N, gens, geninvs, base, lim, j, nu, t)

        for x in nu[j]:
            if x is not None:
                pw = x
                x.flag = False

def fillOrbits(N, gens,geninvs, base, lim, nu):
    for i in range(len(base)):
        orbit = []  # partial orbit already found
        for y in nu[i]:
            if y is not None:
                j = y.permutation.array_form[base[i]]
                if j not in orbit:
                    orbit.append(j)
        for j in range(i + 1, len(base)):
            for x in nu[j]:
                if x is not None:
                    x1 = x.inverse(geninvs)
                    orbit_x = [x.permutation.array_form[it] for it in orbit]
                    new_pts = [p for p in orbit_x if p not in orbit]

                    for p in new_pts:
                        t1 = x1 * (nu[i][x1.permutation.array_form[p]])
                        t1.words = simplify(t1.words)
                        if len(t1.words) < lim:
                            nu[i][p] = t1

# Options:
#   n: max number of rounds
#   s: reset each s rounds
#   w: limit for word size
#   so: sum  orbits size

#
def buildShortWordsSGS(
        N : int,
        gens : Dict[str, Permutation],
        geninvs : Dict[str, str],
        base : List[int], n : int, s : int, w : int, sum_orbit_len : int) -> List[List[Optional[PermWord]]]:
    nu : List[List[Optional[PermWord]]] = [[None for _ in range(N)] for _ in range(len(base))]
    for i in range(len(base)):
        nu[i][base[i]] = PermWord(Permutation(N),[])
    
    number_of_words = 0 # number of words found
    maximum = n # maximum number of iterations
    lim = float(w) # limit for word size
    cnt = 0 # counter for iterations
    iter_gen = chain.from_iterable(product(list(gens), repeat=i) for i in range(1, 12))
    for gen in iter_gen:
        cnt += 1
        if cnt >= maximum or number_of_words == sum_orbit_len :
            break

        perm = gen_perm_from_word(gens, gen) # generate permutation from word
        pw = PermWord(perm, list(gen)) # Create PermWord object
        oneRound(N, gens, geninvs, base, lim, 0, nu, pw) # Update nu with the new word
        nw0 = number_of_words
        number_of_words =  np.sum([np.sum([x != None for x in nu_i]) for nu_i in nu]) # Count number of words
        
        if cnt % s == 0:
            oneImprove(N, gens, geninvs, base, lim, nu) # Improve the current set of words
            if number_of_words < sum_orbit_len: # If not enough words found,
                fillOrbits(N, gens, geninvs, base, lim, nu) # Fill the orbits with new words
            
            lim *= 5 / 4 # Increase the limit for word size
                
    return nu

def factorizeM(N : int,
               gens : Dict[str, Permutation],
               geninvs : Dict[str, str],
               base : List[int], nu : List[List[Optional[PermWord]]],
               target : Permutation) -> List[str]:
    result_list = []
    perm = target
    for i in range(len(base)):
        omega = perm.array_form[base[i]]
        if nu[i][omega] is None:
            raise RuntimeError(f'Unexpected error: No word found for {omega} in orbit {i}, base {base[i]}')
        
        perm *= nu[i][omega].permutation
        result_list = result_list + nu[i][omega].words

    if not perm == Permutation(size = N+1):
        print("failed to reach identity, permutation not in group")

    return simplify(invwords(result_list, geninvs), geninvs)

def gen_perm_from_word(gens : Dict[str, Permutation], words : Iterable[str]) -> Permutation:
    res = Permutation()
    for w in words:
        res = res * gens[w]

    return res

def invwords(ws : List[str], geninvs : Dict[str, str]) -> List[str]:
    inv_ws = [geninvs[g] for g in ws]
    inv_ws.reverse() 
    return inv_ws


#remove invers generators in concatenation
def simplify(word_sequence : list[str], geninvs : dict[str, str] | None = None) -> list[str]:
    if not word_sequence:
        return word_sequence
    
    if geninvs is None:
        return word_sequence # Lazy simplification, just return the original list

    # find adjacent inverse generators
    zero_sum_indices = [(i, i + 1) for i in range(len(word_sequence) - 1) if word_sequence[i] == geninvs[word_sequence[i + 1]] ]

    # If there is no more simplications
    if len(zero_sum_indices) == 0:
        return word_sequence

    zero_sum_indices = list(chain.from_iterable(zero_sum_indices))
    word_sequence = [word_sequence[i] for i in range(len(word_sequence)) if i not in zero_sum_indices]

    # Recursively simplify the list
    word_sequence = simplify(word_sequence, geninvs)
    return word_sequence



# Test Minwictz algorithm
    
def test_SGS(N : int, nu : List[List[Optional[PermWord]]], base : List[int]) -> bool:
    result = True
    for i in range(len(base)):
        # diagonal identities
        p = nu[i][base[i]].words
        if p != []:
            result = False
            print('fail identity')
            
        for j in range(N):
            if j in nu[i]:
                p =nu[i][j].permutation.array_form 
                # stabilizes points upto i
                for k in range(i):
                    p2 = p[base[k]]
                    if p2 != base[k]:
                        result = False
                        print('fail stabilizer',i,j,k)

                
                # correct transversal at i
                if p[j] != base[i]:
                    result = False  
                    print('fail traversal ',i,j)
    return result

def quality(N : int, nu : List[List[Optional[PermWord]]], base : List[int]) -> int:
    result = 0
    for i in range(len(base)):
        maxlen = 0
        for j in range(N):
            if nu[i][j] is not None:
                wordlen = len(nu[i][j].words)
                if wordlen > maxlen:
                    maxlen = wordlen
        result += maxlen
    return result



            
