import itertools
from permutation import Permutation, PermutationGroup

class RubiksCubeSolver:
    def __init__(self):
        '''
            Initializes the RubiksCubeSolver
        '''

        self.operations = {
            "T": Permutation([[0, 2, 8, 6], [1, 5, 7, 3], [9, 36, 27, 18], [10, 37, 28, 19], [11, 38, 29, 20]]),
            "L": Permutation([[0, 18, 45, 44], [3, 21, 48, 41], [6, 24, 51, 38], [9, 11, 17, 15], [10, 14, 16, 12]]),
            "F": Permutation([[6, 27, 47, 17], [7, 30, 46, 14], [8, 33, 45, 11], [18, 20, 26, 24], [19, 23, 25, 21]]),
            "R": Permutation([[2, 42, 47, 20], [5, 39, 50, 23], [8, 36, 53, 26], [27, 29, 35, 33], [28, 32, 34, 30]]),
            "B": Permutation([[0, 15, 53, 29], [1, 12, 52, 32], [2, 9, 51, 35], [36, 38, 44, 42], [37, 41, 43, 39]]),
            "D": Permutation([[15, 24, 33, 42], [16, 25, 34, 43], [17, 26, 35, 44], [45, 47, 53, 51], [46, 50, 52, 48]])
        }

        self.rubiksCubeGroup = PermutationGroup([generator for generator in self.operations.values()])

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
                perm = itertools.permutations(ref_block)
                
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
                
            
    

    def solve(self, state : dict, permutation=False):
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

        if not self.rubiksCubeGroup.contains(state_permutation):
            # If the current state is a valid state of the rubik's cube, return None
            return None

        # The state is a valid state, return the permutation to take the rubik's cube to the solved state
        if permutation is True:
            # It requires to return the permutation itself
            return state_permutation
        else:
            # It requires to return the sequence of operations to take the rubik's cube to the solved state
            generator_sequence = reversed(self.rubiksCubeGroup.generator_product(state_permutation**-1,  original=True))
            operations = []
            for gen in generator_sequence:
                for operationId, operation in self.operations.items():
                    if gen == operation:
                        operations.append(operationId)
                    elif gen**(-1) == operation:
                        operations.append("-" + operationId)

            return operations
         
