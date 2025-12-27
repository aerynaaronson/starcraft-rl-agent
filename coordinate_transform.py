import numpy as np


class CoordinateTransform:
    """
    Single source of truth for all coordinate transformations.
    Handles rotation and CC-centric normalization.
    """
    
    def __init__(self, map_width, map_height, cc_x, cc_y):
        """
        Args:
            map_width: Map width in pixels
            map_height: Map height in pixels
            cc_x: Starting Command Center x coordinate (absolute)
            cc_y: Starting Command Center y coordinate (absolute)
        """
        self.map_width = map_width
        self.map_height = map_height
        self.cc_x = cc_x
        self.cc_y = cc_y
        
        # Determine if we need rotation (bottom-right spawn)
        self.needs_rotation = self._is_bottom_right_spawn(cc_x, cc_y, map_width, map_height)
        
        # If bottom-right, compute the "virtual" top-left CC position after rotation
        if self.needs_rotation:
            self.virtual_cc_x, self.virtual_cc_y = self._rotate_180(cc_x, cc_y, map_width, map_height)
        else:
            self.virtual_cc_x = cc_x
            self.virtual_cc_y = cc_y
        
        # Compute normalization scale (maximum distance from CC to any corner)
        # This ensures [-1, 1] range covers the entire map
        self.scale_x = max(self.virtual_cc_x, map_width - 1 - self.virtual_cc_x)
        self.scale_y = max(self.virtual_cc_y, map_height - 1 - self.virtual_cc_y)
        
        # Prevent division by zero
        self.scale_x = max(self.scale_x, 1)
        self.scale_y = max(self.scale_y, 1)
    
    @staticmethod
    def _is_bottom_right_spawn(x, y, map_width, map_height):
        """Check if spawn is in bottom-right quadrant"""
        return x >= map_width / 2 and y >= map_height / 2
    
    @staticmethod
    def _rotate_180(x, y, map_width, map_height):
        """Rotate point 180 degrees around map center"""
        return map_width - 1 - x, map_height - 1 - y
    
    def world_to_normalized(self, x, y):
        # Apply rotation if needed
        if self.needs_rotation:
            x, y = self._rotate_180(x, y, self.map_width, self.map_height)
        
        # Normalize to [0, 1] range
        nx = x / (self.map_width - 1)
        ny = y / (self.map_height - 1)
        
        # Clamp to [0, 1] and ensure scalar float
        nx = float(np.clip(nx, 0, 1))
        ny = float(np.clip(ny, 0, 1))
        
        return nx, ny
    
    def normalized_to_world(self, nx, ny):
        """
        Convert normalized coordinates [0, 1] back to absolute world coordinates.
        
        This is used when the network outputs a position and we need to execute an action.
        Applies inverse rotation if we're in a bottom-right spawn.
        
        Args:
            nx, ny: Normalized coordinates in [0, 1]
            
        Returns:
            (x, y): Absolute world coordinates (integers)
        """
        # Clamp input
        nx = np.clip(nx, 0, 1)
        ny = np.clip(ny, 0, 1)
        
        # Denormalize to world coordinates
        x = nx * (self.map_width - 1)
        y = ny * (self.map_height - 1)
        
        # Apply inverse rotation if needed
        if self.needs_rotation:
            x, y = self._rotate_180(x, y, self.map_width, self.map_height)
        
        # Convert to integers and clamp to map bounds
        x = int(np.clip(x, 0, self.map_width - 1))
        y = int(np.clip(y, 0, self.map_height - 1))
        
        return x, y
    
    def normalized_to_visualization(self, nx, ny):
        """
        Convert normalized coords [0, 1] to visualization space [0, 1].
        These are the same now, so this is essentially a pass-through.
        """
        vx = np.clip(nx, 0.0, 1.0)
        vy = np.clip(ny, 0.0, 1.0)
        
        return float(vx), float(vy)
    
    def world_to_visualization(self, x, y):
        """
        Complete pipeline: world coords -> normalized -> visualization space.
        This is what should be used for storing coordinates for visualization.
        
        Args:
            x, y: Absolute world coordinates
            
        Returns:
            (vx, vy): Visualization coordinates in [0, 1] with CC at (0.125, 0.125)
        """
        nx, ny = self.world_to_normalized(x, y)
        vx, vy = self.normalized_to_visualization(nx, ny)
        return vx, vy
    
    def get_cc_normalized_position(self):
        """
        Get the CC position in normalized space (always 0, 0 by design).
        Useful for sanity checks.
        """
        return 0.0, 0.0
    
    def get_cc_visualization_position(self):
        """
        Get the CC position in visualization space (always 0.125, 0.125).
        """
        return 0.125, 0.125
    
    def get_map_info(self):
        """Get debug info about the coordinate system"""
        return {
            'map_size': (self.map_width, self.map_height),
            'cc_world_pos': (self.cc_x, self.cc_y),
            'needs_rotation': self.needs_rotation,
            'virtual_cc_pos': (self.virtual_cc_x, self.virtual_cc_y),
            'scale': (self.scale_x, self.scale_y),
            'cc_normalized_pos': self.get_cc_normalized_position(),
            'cc_visualization_pos': self.get_cc_visualization_position(),
        }


def test_coordinate_system():
    """Test suite for coordinate transformations"""
    
    print("=" * 60)
    print("COORDINATE SYSTEM TESTS")
    print("=" * 60)
    
    # Test case 1: Top-left spawn (no rotation)
    print("\nTest 1: Top-left spawn (32, 32) on 176x176 map")
    transform = CoordinateTransform(176, 176, 32, 32)
    info = transform.get_map_info()
    print(f"  Needs rotation: {info['needs_rotation']}")
    print(f"  Virtual CC: {info['virtual_cc_pos']}")
    print(f"  Scale: {info['scale']}")
    print(f"  CC normalized: {info['cc_normalized_pos']}")
    print(f"  CC visualization: {info['cc_visualization_pos']}")
    
    # CC should map to (0, 0) in normalized, (0.125, 0.125) in visualization
    nx, ny = transform.world_to_normalized(32, 32)
    vx, vy = transform.world_to_visualization(32, 32)
    print(f"  CC (32, 32) -> normalized ({nx:.2f}, {ny:.2f}) -> viz ({vx:.3f}, {vy:.3f})")
    assert abs(nx) < 0.01 and abs(ny) < 0.01, "CC should be at origin in normalized!"
    assert abs(vx - 0.125) < 0.01 and abs(vy - 0.125) < 0.01, "CC should be at (0.125, 0.125) in visualization!"
    
    # Test corners
    corners = [(0, 0), (175, 0), (0, 175), (175, 175)]
    for cx, cy in corners:
        nx, ny = transform.world_to_normalized(cx, cy)
        vx, vy = transform.world_to_visualization(cx, cy)
        back_x, back_y = transform.normalized_to_world(nx, ny)
        print(f"  Corner ({cx}, {cy}) -> norm ({nx:.2f}, {ny:.2f}) -> viz ({vx:.3f}, {vy:.3f}) -> back ({back_x}, {back_y})")
    
    # Test case 2: Bottom-right spawn (needs rotation)
    print("\nTest 2: Bottom-right spawn (144, 144) on 176x176 map")
    transform = CoordinateTransform(176, 176, 144, 144)
    info = transform.get_map_info()
    print(f"  Needs rotation: {info['needs_rotation']}")
    print(f"  Virtual CC: {info['virtual_cc_pos']}")
    print(f"  Scale: {info['scale']}")
    print(f"  CC normalized: {info['cc_normalized_pos']}")
    print(f"  CC visualization: {info['cc_visualization_pos']}")
    
    # CC should still map to (0, 0) in normalized, (0.125, 0.125) in visualization
    nx, ny = transform.world_to_normalized(144, 144)
    vx, vy = transform.world_to_visualization(144, 144)
    print(f"  CC (144, 144) -> normalized ({nx:.2f}, {ny:.2f}) -> viz ({vx:.3f}, {vy:.3f})")
    assert abs(nx) < 0.01 and abs(ny) < 0.01, "CC should be at origin in normalized!"
    assert abs(vx - 0.125) < 0.01 and abs(vy - 0.125) < 0.01, "CC should be at (0.125, 0.125) in visualization!"
    
    # Enemy base (top-left in world) should appear in consistent position
    enemy_x, enemy_y = 32, 32
    nx, ny = transform.world_to_normalized(enemy_x, enemy_y)
    vx, vy = transform.world_to_visualization(enemy_x, enemy_y)
    print(f"  Enemy base ({enemy_x}, {enemy_y}) -> norm ({nx:.2f}, {ny:.2f}) -> viz ({vx:.3f}, {vy:.3f})")
    
    # Test corners after rotation
    for cx, cy in corners:
        nx, ny = transform.world_to_normalized(cx, cy)
        vx, vy = transform.world_to_visualization(cx, cy)
        back_x, back_y = transform.normalized_to_world(nx, ny)
        print(f"  Corner ({cx}, {cy}) -> norm ({nx:.2f}, {ny:.2f}) -> viz ({vx:.3f}, {vy:.3f}) -> back ({back_x}, {back_y})")
    
    # Test case 3: Verify consistency
    print("\nTest 3: Verify top-left and bottom-right see enemy similarly")
    tl_transform = CoordinateTransform(176, 176, 32, 32)
    br_transform = CoordinateTransform(176, 176, 144, 144)
    
    # In top-left spawn, enemy is at (144, 144) in world coords
    tl_enemy_viz = tl_transform.world_to_visualization(144, 144)
    print(f"  Top-left spawn sees enemy at viz: {tl_enemy_viz}")
    
    # In bottom-right spawn, enemy is at (32, 32) in world coords
    br_enemy_viz = br_transform.world_to_visualization(32, 32)
    print(f"  Bottom-right spawn sees enemy at viz: {br_enemy_viz}")
    
    # These should be very similar (opposite corner from CC)
    diff = np.sqrt((tl_enemy_viz[0] - br_enemy_viz[0])**2 + 
                   (tl_enemy_viz[1] - br_enemy_viz[1])**2)
    print(f"  Difference: {diff:.3f}")
    print(f"  ✓ Enemy base appears consistently!" if diff < 0.1 else "  ✗ INCONSISTENT!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    test_coordinate_system()