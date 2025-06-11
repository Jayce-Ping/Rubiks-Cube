from typing import Dict, List, Iterable
from itertools import permutations, chain
from sympy.core.random import seed as sympy_seed
from sympy.combinatorics.permutations import Permutation
from minkwitz import SGSPermutationGroup

class RubiksCubeSolver:
    def __init__(self):
        '''
            Initializes the RubiksCubeSolver
        '''
        size = 54
        operations = {
            "T": [[0, 2, 8, 6], [1, 5, 7, 3], [9, 36, 27, 18], [10, 37, 28, 19], [11, 38, 29, 20]],
            "L": [[0, 18, 45, 44], [3, 21, 48, 41], [6, 24, 51, 38], [9, 11, 17, 15], [10, 14, 16, 12]],
            "F": [[6, 27, 47, 17], [7, 30, 46, 14], [8, 33, 45, 11], [18, 20, 26, 24], [19, 23, 25, 21]],
            "R": [[2, 42, 47, 20], [5, 39, 50, 23], [8, 36, 53, 26], [27, 29, 35, 33], [28, 32, 34, 30]],
            "B": [[0, 15, 53, 29], [1, 12, 52, 32], [2, 9, 51, 35], [36, 38, 44, 42], [37, 41, 43, 39]],
            "D": [[15, 24, 33, 42], [16, 25, 34, 43], [17, 26, 35, 44], [45, 47, 53, 51], [46, 50, 52, 48]]
        }
        self.operations : Dict[str, Permutation] = {
            key: Permutation(value, size=size)
            for key, value in operations.items()
        }
        self.rubiksCubeGroup = SGSPermutationGroup(self.operations, deterministic=True)
        self.group = self.rubiksCubeGroup.G
        if self.rubiksCubeGroup.nu is None:
            self.rubiksCubeGroup.getShortWords(n=10000, s=2000, w=20)

        self.faceIndices = self.operations.keys()

        self.edgeBlocks = [
            [('T', 2), ('B', 2)],
            [('T', 4), ('L', 2)],
            [('T', 6), ('R', 2)],
            [('T', 8), ('F', 2)],
            [('L', 4), ('B', 6)],
            [('L', 6), ('F', 4)],
            [('L', 8), ('D', 4)],
            [('F', 6), ('R', 4)],
            [('F', 8), ('D', 2)],
            [('R', 6), ('B', 4)],
            [('R', 8), ('D', 6)],
            [('B', 8), ('D', 8)]
        ]

        self.cornerBlocks = [
            [('T', 1), ('L', 1), ('B', 3)], 
            [('T', 3), ('R', 3), ('B', 1)], 
            [('T', 7), ('L', 3), ('F', 1)], 
            [('T', 9), ('F', 3), ('R', 1)], 
            [('L', 7), ('D', 7), ('B', 9)], 
            [('L', 9), ('F', 7), ('D', 1)], 
            [('F', 9), ('D', 3), ('R', 7)], 
            [('R', 9), ('B', 7), ('D', 9)]
        ]
        
        self.centerBlocks = [[('T', 5)], [('L', 5)], [('F', 5)], [('R', 5)], [('B', 5)], [('D', 5)]]

    @property
    def init_state(self):
        return {
            k : [i] * 9 for i, k in enumerate("TLFRBD")
        }
    
    def random_state(self, seed=None) -> dict:
        """
            Generates a random state of the rubik's cube.
            Args:
                seed (int): The seed for the random number generator
            Returns:
                dict: The state of the rubik's cube
        """ 
        if seed is not None:
            sympy_seed(seed)

        state_permutation_array = self.group.random(af=True)
        init_state = self.init_state
        state_seq = state_dict_to_sequence(init_state)
        permuted_seq = [state_seq[i] for i in state_permutation_array]
        state = state_sequence_to_dict(permuted_seq, face_ids=list(self.faceIndices))
        return state

    def getCellNumber(self, cell : tuple):
        faceIdMap = {
            'T': 0,
            'L': 1,
            'F': 2,
            'R': 3,
            'B': 4,
            'D': 5
        }
        return faceIdMap[cell[0]] * 9 + cell[1] - 1


    def matchCellPermutation(self, state : dict, ref_blocks: list, colorIdToFaceId : dict):
        
        # Make a deep copy of ref_blocks to avoid modifying the original list
        ref_blocks = [list(block) for block in ref_blocks]

        state_blocks = [
            [(faceId, cellId, state[faceId][cellId - 1]) for faceId, cellId in ref_block]
            for ref_block in ref_blocks
        ]
        cellPermutationMap = {}

        for block in state_blocks:
            faceIds = tuple(q[0] for q in block)
            cellIds = tuple(q[1] for q in block)
            colors = tuple(colorIdToFaceId[q[2]] for q in block)
            
            matched = False
            for idx, ref_block in enumerate(ref_blocks):
                perm = permutations(ref_block)
                
                for p in perm:
                    ref_faceIds = tuple(q[0] for q in p)
                    ref_cellIds = tuple(q[1] for q in p)

                    if colors == ref_faceIds:
                        for faceId, cellId, refFace, refCell in zip(faceIds, cellIds, ref_faceIds, ref_cellIds): 
                            cellPermutationMap[(faceId, cellId)] = (refFace, refCell)
                        matched = True
                        break

                if matched:
                    ref_blocks.pop(idx)
                    break
            
            if not matched:
                print("Invalid state")
                return None

        
        return cellPermutationMap
    

    def get_state_permutation(self, state : dict) -> Permutation:
        """
            Find a permutation that takes the initial state of the rubik's cube to the given state.
        """
        # Get block color map according to the center blocks
        colorIdToFaceId = {colorId: faceId for faceId, colorId in zip(self.faceIndices, [blocks[4] for blocks in state.values()])}
        
        # Get the permutation of the rubik's cube to obtain the current state from the solved state
        cellPermutation = {}
        for ref_blocks in [self.cornerBlocks, self.edgeBlocks, self.centerBlocks]:
            res = self.matchCellPermutation(state, ref_blocks, colorIdToFaceId)
            if res is None:
                print("Match failed - possibly invalid state!")
                return None
            else:
                cellPermutation.update(res)

        
        encoded_CellPermutation = sorted([tuple([self.getCellNumber(state_cell), self.getCellNumber(ref_cell)]) for state_cell, ref_cell in cellPermutation.items()], key=lambda cell_pair: cell_pair[0])
        # The permutation of the rubik's cube to obtain the current state
        state_permutation = Permutation.from_sequence([cell_pair[1] for cell_pair in encoded_CellPermutation])

        return state_permutation


    def solve(self, state : dict):
        '''
            Solves the rubik's cube given a state
            Args:
                state (dict): The state of the rubik's cube
                state is in a form of
                {
                    "T": [xt, ..., yt],
                    "L": [xl, ..., yl],
                    "F": [xf, ..., yf],
                    "R": [xr, ..., yr],
                    "B": [xb, ..., yb],
                    "D": [xd, ..., yd]
                }
                where xt, ..., yt are the colors (integers) from 1 to 6 of the top face of the rubik's cube.
        '''
        state_permutation = self.get_state_permutation(state)

        if not self.rubiksCubeGroup.G.contains(state_permutation):
            # If the current state is a valid state of the rubik's cube, return None
            raise ValueError("Invalid state: The given state is not a valid Rubik's Cube configuration.")

        perm_words = self.rubiksCubeGroup.FactorPermutation(state_permutation)
        # perm_words = [w1, w2, ...], where each w_i is a word in the generators of the Rubik's Cube group
        # each w_i is represented as a list of generator names or with a '-' sign for inverses, ['T', 'L', '-F', 'R', ...]
        # we have w1w2... = word(state_permutation)

        # The operations takes given state back to the initial state is the reverse of the operations that take the initial state to the given state
        operations = [s[1] if s[0] == '-' else '-' + s[0] for s in perm_words]
        # Reverse the sign of each operation to get inverse operations
        # Apply operations from left to right on the given state to get the initial state

        return operations
    
    def apply_operations(self, state : dict, operations : Iterable[str], process=True) -> List[Dict] | Dict:
        """
            Applies the given operations on the state of the rubik's cube.
            Args:
                state (dict): The state of the rubik's cube
                operations (List[str]): The operations to apply on the state
                process (bool): Whether to provide a process of applying the operations
            Returns:
                dict: The new state of the rubik's cube after applying the operations
        """
        applying_process = []
        for op in operations:
            if op[0] == '-':
                op = ~self.operations[op[1]]
            else:
                op = self.operations[op]

            state_seq = state_dict_to_sequence(state)
            # permuted_seq : List[int] = op(state_seq)
            permuted_seq = [state_seq[i] for i in op.array_form]
            state = state_sequence_to_dict(permuted_seq, face_ids=list(self.faceIndices))
            if process:
                applying_process.append(state.copy())

        if process:
            return applying_process
        else:
            return state
        

    

            
def state_dict_to_sequence(state: Dict[str, List[int]]) -> List[int]:
    """
        Converts the state dictionary to a sequence of colors.
        Args:
            state (dict): The state of the rubik's cube
        Returns:
            List[int]: The sequence of colors
    """
    return list(chain.from_iterable(state.values()))


def state_sequence_to_dict(sequence: List[int], face_ids : List[str] = ['T', 'L', 'F', 'R', 'B', 'D']) -> Dict[str, List[int]]:
    """
        Converts the sequence of colors to a state dictionary.
        Args:
            sequence (List[int]): The sequence of colors
        Returns:
            dict: The state of the rubik's cube
    """
    return {k: sequence[i:i+9] for i, k in zip(range(0, 54, 9), face_ids)}
    