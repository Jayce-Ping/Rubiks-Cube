import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from rubiksCubeSolver import RubiksCubeSolver, state_dict_to_sequence, state_sequence_to_dict
import argparse

class RubiksCubeVisualizer:
    def __init__(self):
        """Initialize the enhanced Rubik's Cube visualizer"""
        self.solver = RubiksCubeSolver()
        
        # Enhanced color scheme with better visual appeal
        self.colors = {
            0: '#FFFFFF',  # T (Top) - White
            1: '#00FF00',  # L (Left) - Green  
            2: '#0000FF',  # F (Front) - Blue
            3: '#FF0000',  # R (Right) - Red
            4: '#FFA500',  # B (Back) - Orange
            5: '#FFFF00'   # D (Down) - Yellow
        }
        
        # Face names
        self.face_names = ['T', 'L', 'F', 'R', 'B', 'D']
        
        self.cube_size = 1
        self.gap_size = 0.0
        
        # Generate cube positions with proper spacing
        self.cube_positions = self._generate_cube_positions()
        
    def _generate_cube_positions(self):
        """Generate cube positions with proper spacing"""
        positions = {}
        spacing = 1.0 + self.gap_size
        
        for face in self.face_names:
            positions[face] = []
            
        # T face (Top) - z=1.5
        for i in range(3):
            for j in range(3):
                x = (j - 1) * spacing
                y = (i - 1) * spacing
                positions['T'].append(np.array([x, y, 1.5]))
        
        # D face (Down) - z=-1.5
        for i in range(3):
            for j in range(3):
                x = (j - 1) * spacing
                y = (1 - i) * spacing
                positions['D'].append(np.array([x, y, -1.5]))
        
        # F face (Front) - y=1.5
        for i in range(3):
            for j in range(3):
                x = (j - 1) * spacing
                z = (1 - i) * spacing
                positions['F'].append(np.array([x, 1.5, z]))
        
        # B face (Back) - y=-1.5
        for i in range(3):
            for j in range(3):
                x = (1 - j) * spacing
                z = (1 - i) * spacing
                positions['B'].append(np.array([x, -1.5, z]))
        
        # L face (Left) - x=-1.5
        for i in range(3):
            for j in range(3):
                y = (j - 1) * spacing
                z = (1 - i) * spacing
                positions['L'].append(np.array([-1.5, y, z]))
        
        # R face (Right) - x=1.5
        for i in range(3):
            for j in range(3):
                y = (1 - j) * spacing
                z = (1 - i) * spacing
                positions['R'].append(np.array([1.5, y, z]))
        
        return positions
    
    def _create_cube_face(self, center, face_name, size=None, rotation_axis=None, rotation_angle=0):
        """Create cube face vertices"""
        if size is None:
            size = self.cube_size
        
        half_size = size / 2
        offset = 0.02  # Small offset to prevent z-fighting
        
        if face_name == 'T':  # Top face
            vertices = np.array([
                [center[0] - half_size, center[1] - half_size, center[2] + offset],
                [center[0] + half_size, center[1] - half_size, center[2] + offset],
                [center[0] + half_size, center[1] + half_size, center[2] + offset],
                [center[0] - half_size, center[1] + half_size, center[2] + offset]
            ])
        elif face_name == 'D':  # Down face
            vertices = np.array([
                [center[0] - half_size, center[1] - half_size, center[2] - offset],
                [center[0] + half_size, center[1] - half_size, center[2] - offset],
                [center[0] + half_size, center[1] + half_size, center[2] - offset],
                [center[0] - half_size, center[1] + half_size, center[2] - offset]
            ])
        elif face_name == 'F':  # Front face
            vertices = np.array([
                [center[0] - half_size, center[1] + offset, center[2] - half_size],
                [center[0] + half_size, center[1] + offset, center[2] - half_size],
                [center[0] + half_size, center[1] + offset, center[2] + half_size],
                [center[0] - half_size, center[1] + offset, center[2] + half_size]
            ])
        elif face_name == 'B':  # Back face
            vertices = np.array([
                [center[0] + half_size, center[1] - offset, center[2] - half_size],
                [center[0] - half_size, center[1] - offset, center[2] - half_size],
                [center[0] - half_size, center[1] - offset, center[2] + half_size],
                [center[0] + half_size, center[1] - offset, center[2] + half_size]
            ])
        elif face_name == 'L':  # Left face
            vertices = np.array([
                [center[0] - offset, center[1] + half_size, center[2] - half_size],
                [center[0] - offset, center[1] - half_size, center[2] - half_size],
                [center[0] - offset, center[1] - half_size, center[2] + half_size],
                [center[0] - offset, center[1] + half_size, center[2] + half_size]
            ])
        elif face_name == 'R':  # Right face
            vertices = np.array([
                [center[0] + offset, center[1] - half_size, center[2] - half_size],
                [center[0] + offset, center[1] + half_size, center[2] - half_size],
                [center[0] + offset, center[1] + half_size, center[2] + half_size],
                [center[0] + offset, center[1] - half_size, center[2] + half_size]
            ])

        if rotation_axis is not None and rotation_angle != 0:
            # Apply rotation around the specified axis
            rotation_matrix = self._rotation_matrix(rotation_axis, rotation_angle)
            vertices = np.dot(vertices - center, rotation_matrix) + center
        
        return vertices

    def _rotation_matrix(self, axis, angle):
        """Create a rotation matrix for a given axis and angle"""
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = axis / np.linalg.norm(axis)
        return np.array([
            [cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
            [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
            [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]
        ])

    def _setup_plot(self, ax : Axes3D, title="Rubik's Cube Animation"):
        """Setup plot with enhanced visual properties"""
        ax.clear()
        
        # Set limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        
        # 设置背景为深色
        fig = ax.figure
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2d2d2d')

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Clean appearance
        ax.set_axis_off()
        
        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color='white')
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)

    def _draw_static_cube(self, ax, state):
        """Draw cube in static state"""
        for face_name in self.face_names:
            face_colors = state[face_name]
            positions = self.cube_positions[face_name]
            
            for i, (pos, color_idx) in enumerate(zip(positions, face_colors)):
                color = self.colors[color_idx]
                vertices = self._create_cube_face(pos, face_name)
                
                poly = [[tuple(vertex) for vertex in vertices]]
                collection = Poly3DCollection(poly,
                                            facecolors=color,
                                            edgecolors='black',
                                            linewidths=1.5,
                                            alpha=0.8)
                ax.add_collection3d(collection)

    def _draw_rotating_cube(self, ax, state, rotation_axis, affected_blocks, angle):
        """Draw cube with rotation applied to affected blocks"""
        for face_name in self.face_names:
            face_colors = state[face_name]
            positions = self.cube_positions[face_name]
            
            for i, (pos, color_idx) in enumerate(zip(positions, face_colors)):
                color = self.colors[color_idx]
                
                # Apply rotation if affected
                if (face_name, i) in affected_blocks:
                    rotated_pos = self._rotate_position(pos, rotation_axis, angle)
                    vertices = self._create_cube_face(rotated_pos, face_name, 
                                                    rotation_axis=rotation_axis, 
                                                    rotation_angle=-angle)
                else:
                    vertices = self._create_cube_face(pos, face_name)
                
                poly = [[tuple(vertex) for vertex in vertices]]
                collection = Poly3DCollection(poly,
                                            facecolors=color,
                                            edgecolors='black',
                                            linewidths=1.5,
                                            alpha=0.8)
                ax.add_collection3d(collection)

    def create_sequence_animation(self, initial_state, operations, pause_frames=3, speed=1, save_path=None):
        """Create animation for a sequence of operations"""
        # Auto-adjust frames per operation to make sure the animation is smooth
        # Assuming each operation takes 10 frames for rotation and 3 frames for pause
        frames_per_op = max(5, int(10 / speed))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_plot(ax, "Rubik's Cube Animation")
        
        # Calculate total frames
        total_frames = len(operations) * (frames_per_op + pause_frames)
        
        # Precompute all states
        states = [initial_state]
        current_state = initial_state.copy()
        for op in operations:
            current_state = self.solver.apply_operations(current_state, [op], process=False)
            states.append(current_state.copy())
    
        def animate(frame):
            self._setup_plot(ax, f"Rubik's Cube - Operation {frame // (frames_per_op + pause_frames) + 1} / {len(operations)}")
            
            # Determine current operation and progress
            op_index = frame // (frames_per_op + pause_frames)
            frame_in_op = frame % (frames_per_op + pause_frames)
            
            if op_index >= len(operations):
                # Animation finished, show final state
                self._draw_static_cube(ax, states[-1])
                return
            
            operation = operations[op_index]
            
            if frame_in_op < frames_per_op:
                # In rotation phase
                t = frame_in_op / (frames_per_op - 1) if frames_per_op > 1 else 1
                angle = t * (np.pi / 2)  # 90 degree rotation
                
                # Handle reverse operations
                if operation.startswith('-'):
                    angle = -angle
                    op_name = operation[1:]
                else:
                    op_name = operation
                
                # Get rotation info
                rotation_axis, affected_blocks = self._get_rotation_info(op_name)
                
                # Draw rotating cube
                self._draw_rotating_cube(ax, states[op_index], rotation_axis, affected_blocks, -angle)
            else:
                # In pause phase, show completed state
                self._draw_static_cube(ax, states[op_index + 1])
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=0.01, repeat=False)
        
        if save_path:
            try:
                anim.save(save_path, writer='pillow', dpi=100, fps=120)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Error saving animation: {e}")
        
        else:
            plt.show()
            
        return anim

    
    def _get_rotation_info(self, operation):
        """Get rotation information for operations"""
        rotation_axes = {
            'T': np.array([0, 0, 1]),
            'D': np.array([0, 0, -1]),
            'F': np.array([0, 1, 0]),
            'B': np.array([0, -1, 0]),
            'L': np.array([-1, 0, 0]),
            'R': np.array([1, 0, 0])
        }
        
        # Simplified affected blocks
        affected_blocks_map = {
            'T': [('T', i) for i in range(9)] + 
                 [('L', i) for i in range(3)] + 
                 [('F', i) for i in range(3)] + 
                 [('R', i) for i in range(3)] + 
                 [('B', i) for i in range(3)],
            
            'L': [('L', i) for i in range(9)] + 
                 [('T', i) for i in [0, 3, 6]] + 
                 [('F', i) for i in [0, 3, 6]] + 
                 [('D', i) for i in [0, 3, 6]] + 
                 [('B', i) for i in [2, 5, 8]],
            
            'F': [('F', i) for i in range(9)] + 
                 [('T', i) for i in range(6, 9)] + 
                 [('R', i) for i in [0, 3, 6]] + 
                 [('D', i) for i in range(3)] + 
                 [('L', i) for i in [2, 5, 8]],
            
            'R': [('R', i) for i in range(9)] + 
                 [('T', i) for i in [2, 5, 8]] + 
                 [('B', i) for i in [0, 3, 6]] + 
                 [('D', i) for i in [2, 5, 8]] + 
                 [('F', i) for i in [2, 5, 8]],
            
            'B': [('B', i) for i in range(9)] + 
                 [('T', i) for i in range(3)] + 
                 [('L', i) for i in [0, 3, 6]] + 
                 [('D', i) for i in range(6, 9)] + 
                 [('R', i) for i in [2, 5, 8]],
            
            'D': [('D', i) for i in range(9)] + 
                 [('F', i) for i in range(6, 9)] + 
                 [('R', i) for i in range(6, 9)] + 
                 [('B', i) for i in range(6, 9)] + 
                 [('L', i) for i in range(6, 9)]
        }
        
        op_name = operation[0] if operation[0] != '-' else operation[1]
        return rotation_axes.get(op_name, np.array([0, 0, 1])), affected_blocks_map.get(op_name, [])
    
    def _rotate_position(self, pos, axis, angle):
        """Rotate position using Rodrigues' formula"""
        try:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            return (pos * cos_angle + 
                    np.cross(axis, pos) * sin_angle + 
                    axis * np.dot(axis, pos) * (1 - cos_angle))
        except Exception:
            return pos  # Return original position if rotation fails
    
    def plot_cube(self, state, title="Rubik's Cube"):
        """Plot the Rubik's Cube in a static state"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_plot(ax, title)
        
        self._draw_static_cube(ax, state)
        
        plt.show()
        return fig, ax


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Rubik's Cube Visualizer")
    parser.add_argument('--save', type=str, default=None, help="Path to save the animation (e.g., 'animation.gif')")
    parser.add_argument('--speed', type=int, default=1, help="Speed of the animation (default: 1)")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for generating initial state (default: 0)")
    return parser.parse_args()

def main():
    """Main function with better error handling"""
    print("🎯 Rubik's Cube Visualizer")
    print("=" * 30)
    args = parse_args()
    print(f"Arguments: {args}")
    for arg, value in vars(args).items():
        print(f"{arg:>20}={value}")

    seed = args.seed
    save_path = args.save
    speed = args.speed
    
    try:
        visualizer = RubiksCubeVisualizer()
        
        # Example state
        init_state = visualizer.solver.random_state(seed=seed)

        # Get the sequence of operations
        sol = visualizer.solver.solve(init_state)

        # Visualize the solving process
        visualizer.create_sequence_animation(init_state, sol, speed=speed, save_path=save_path)
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()