from typing import Any


class Permutation:
    """置换类"""
    
    def __init__(self, cycles=None, perm=None, size=None):
        """
        Initialize a permutation.
        Args:
            perm: array form [0,2,1]: 0→0, 1→2, 2→1
            cycles: cycles form [(0,1,2), (3,4)]
            n: the size of the permutation, if None, it will be inferred from cycles or perm
        """
        if cycles is not None:
            if not self._cycles_disjointQ(cycles):
                raise ValueError("Cycles must be disjoint.")
            self.perm = self._cycles_to_array(cycles, size)
        elif perm is None:
            self.perm = []
        else:
            self.perm = list(perm)
        self._shrink()
    
    def _cycles_disjointQ(self, cycles):
        """
            Check if all cycles are disjoint.
        """
        seen = set()
        for cycle in cycles:
            for element in cycle:
                if element in seen:
                    return False
                seen.add(element)
        
        return True

    def _cycles_to_array(self, cycles, n=None):
        """
            Convert cycles to array form.
        """
        if not cycles:
            return []
        
        max_element = 0
        for cycle in cycles:
            if len(cycle) > 0:
                max_element = max(max_element, max(cycle))
        
        if n is not None:
            max_element = max(max_element, n - 1)
        
        array_size = max_element + 1
        result = list(range(array_size))
        
        for cycle in cycles:
            if len(cycle) <= 1:
                continue
            
            # [0,1,2,3,4,5,6,7]
            # [0->0, 1->1, 2->3, 3->7, 4->4, 5->5, 6->6, 7->2]

            # (a, b, c, ...): a→b, b→c, ..., last→a
            for i in range(len(cycle)):
                next_i = (i + 1) % len(cycle)
                result[cycle[i]] = cycle[next_i]
        
        return result
    
    def to_cycles(self):
        """
        Convert the permutation to cycle notation.
        """
        if not self.perm:
            return []
        
        visited = [False] * len(self.perm)
        cycles = []
            
        for i in range(len(self.perm)):
            if not visited[i]:
                cycle = []
                current = i
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    current = self.perm[current]
                
                if len(cycle) > 1:
                    cycles.append(cycle)
        
        return cycles
    
    def __call__(self, lis : list):
        """
        Call the permutation on a list.
        Args:
            lis: list to be permuted
        Returns:
            A new list with the permutation applied.
        """
        return self.apply(lis)
    
    def apply(self, lis : list):
        """
        Apply the permutation to a list.
        Args:
            lis: list to be permuted
        Returns:
            A new list with the permutation applied.
        """
        if not self.perm:
            return lis.copy()
        

        res = [0] * len(self.perm)
        for i in range(len(self.perm)):
            res[self.perm[i]] = lis[i]

        return res
            
    
    def __repr__(self):
        """
        Return a string representation of the permutation.
        """
        cycles = self.to_cycles()
        if not cycles:
            return "Permutation()"  # 恒等置换
        cycle_strs = [f"({','.join(map(str, cycle))})" for cycle in cycles]
        return f"Permutation({' '.join(cycle_strs)})"
    
    def _shrink(self):
        """
        Shrink the permutation by removing trailing elements that map to themselves.
        """
        m = len(self.perm)
        while m > 0 and self.perm[m - 1] == m - 1:
            m -= 1
        self.perm = self.perm[:m]
    
    def __getitem__(self, i):
        """
        Get the image of i under the permutation.
        """
        return self.perm[i] if i < len(self.perm) else i
    
    def is_empty(self):
        """
        Check if the permutation is empty (i.e., it is the identity permutation).
        """
        return len(self.perm) == 0
    
    def is_identity(self):
        """
        Check if the permutation is the identity permutation.
        """
        return all(self[i] == i for i in range(len(self.perm))) or len(self.perm) == 0
    
    def __mul__(self, rhs):
        """
            multiplication operation: self * rhs
            (self * rhs)[i] := rhs[self[i]]
        """
        res = Permutation()
        size = max(len(self.perm), len(rhs.perm))
        res.perm = [0] * size
        
        for i in range(size):
            res.perm[i] = rhs[self[i]]
        
        res._shrink()
        return res
    
    def __truediv__(self, rhs):
        """
            division operation: self / rhs
            (self / rhs) := (self * rhs.inv())
        """
        res = Permutation()
        size = max(len(self.perm), len(rhs.perm))
        
        inv_rhs = rhs.inv()
        res = self * inv_rhs
        
        res._shrink()
        return res
    
    def __eq__(self, other):
        """
            Test for equality
            Returns True if self and other permute the same elements in the same way.
        """
        if not isinstance(other, Permutation):
            return False
        
        if not self.perm and not other.perm:
            return True
        
        max_len = max(len(self.perm) if self.perm else 0, 
                     len(other.perm) if other.perm else 0)
        
        for i in range(max_len):
            if self[i] != other[i]:
                return False
        
        return True
    
    def inv(self):
        """
            Compute the inverse of the permutation
        """
        res = Permutation()
        res.perm = [0] * len(self.perm)
        
        for i in range(len(self.perm)):
            res.perm[self[i]] = i
        
        return res


class PermutationGroup:
    """
        Permutation Group class
        Use Schreier-Sims algorithm to construct a permutation group
    """
    
    def __init__(self, generators=None, n=None):
        """
        Initialize a permutation group.
        Args:
            generators: list of Permutation objects that generate the group
            n: size of the permutation group, if None, it will be inferred from generators
        """
        if generators is None:
            generators = []
        
        if n is None:
            n = 0
            for gen in generators:
                if gen.perm:
                    n = max(n, len(gen.perm))

        self.n = n
        self.k = 1
        self.orbit : list[bool] = [False] * n if n > 0 else []
        self.generators : list[Permutation] = []
        self.original_generators : list[Permutation] = generators.copy()  # Store original generators
        self.transversal : list[None | Permutation] = [None] * n if n > 0 else []
        self.next : None | PermutationGroup = None
        
        if n == 0:
            return
        
        # Initialize the orbit and transversal
        self.base_point = n - 1 # Base point - defaultly the last element

        self.orbit[self.base_point] = True # Set orbit for the base point
        # Set the transversal for the base point to the identity permutation
        identity = Permutation()
        self.transversal[self.base_point] = identity
        if n > 0:
            self.next = PermutationGroup([], n - 1)
        
        # Add the generators to the group
        for i, gen in enumerate(generators):
            self.extend(gen)
    
    def __del__(self):
        """Clean up the group by deleting the next group if it exists."""
        if self.next:
            del self.next

    def __repr__(self):
        """Return a string representation of the permutation group."""
        if not self.generators:
            return "PermutationGroup()"
        
        gen_strs = [repr(gen) for gen in self.generators]
        return f"PermutationGroup({', '.join(gen_strs)})"
    
    def _sift(self, h : Permutation):
        """Sift the permutation h to find its position in the group."""
        if self.n == 0 or h.is_empty():
            return h

        i = h[self.base_point]
        if not self.orbit[i]:
            return h

        h = h * self.transversal[i].inv()

        return self.next._sift(h)
        

    def decompose(self, h : Permutation):
        """
        Decompose the permutation h into a product of representatives of the transversal.
        Args:
            h: Permutation to be decomposed
        Returns:
            A list `decomposition` is a list of permutations that represent the decomposition.
        """
        # Test if the permutation h is in the group
        if not self.contains(h):
            raise ValueError(f"The permutation {h} is not in the group {self}")
        
        return self._decompose(h)[1]

    def _decompose(self, h : Permutation):
        """
            Decompose the permutation h into a product of representatives of the transversal.
        """
        if self.n == 0 or h.is_empty() or self.next is None:
            return h, []
        
        i = h[self.base_point]
        if not self.orbit[i]:
            return h, []
        
        h = h * self.transversal[i].inv()
        decomposition = [self.transversal[i]]
        next_h, next_decomp = self.next._decompose(h)
        decomposition.extend(next_decomp)
        return next_h, decomposition

    def _extend_transversal(self, t : Permutation):
        """
        Attempt to extend the transversal with the permutation t.
        """
        assert self.next is not None, f"The next stabilizer is None for group {self}"

        i = t[self.base_point]
        if not self.orbit[i]:
            self.k += 1
            self.orbit[i] = True
            self.transversal[i] = t
            for s in self.generators:
                self._extend_transversal(t * s)
        else:
            self.next.extend(t * self.transversal[i].inv())
    
    def extend(self, g):
        """
        Add a generator g to the group.
        """
        h = self._sift(g)
        
        if h.is_empty():
            return
                
        self.generators.append(h)
        for i in range(self.n):
            if self.orbit[i]:
                self._extend_transversal(self.transversal[i] * h)
    
    def contains(self, h : Permutation):
        """
        Check if the group contains the permutation h.
        """
        # Test the length of the permutation
        if len(h.perm) > self.n:
            return False
        
        test_h = Permutation(perm=h.perm)
        test_h = self._sift(test_h)
        return test_h.is_empty()
    
    def order(self):
        """
        Calculate the order of the permutation group.
        """
        if self.n == 0:
            return 1
        return self.next.order() * self.k
    


def permutation_product(*perms : Permutation):
    """
        Permutation product of multiple permutations.
        (perm_1 * perm_2 * ... * perm_n)[i] := perm_n[perm_(n-1)[...perm_1[i]...]]
    """
    if not perms:
        return Permutation()

    result = Permutation(perm=perms[0].perm.copy())
    
    for perm in perms[1:]:
        result = result * perm
    
    return result