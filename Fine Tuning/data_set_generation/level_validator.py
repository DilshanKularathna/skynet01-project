import json
from dataclasses import dataclass
from typing import List
from collections import deque

@dataclass
class Position:
    x: int
    y: int

@dataclass
class MapTiles:
    width: int
    height: int
    walls: list[Position]
    enemies: list[Position]
    player_pos: Position

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
class LevelValidator:
    def __init__(self):
        self.required_width = 20
        self.required_height = 15
    
    def validate_dimensions(self, maptiles: MapTiles) -> List[str]:
        """Validate map dimensions are 20x15"""
        errors = []
        if maptiles.width != self.required_width:
            errors.append(f"Invalid width: {maptiles.width}, expected {self.required_width}")
        if maptiles.height != self.required_height:
            errors.append(f"Invalid height: {maptiles.height}, expected {self.required_height}")
        return errors
    
    def validate_single_player(self, maptiles: MapTiles) -> List[str]:
        """Validate exactly one player exists"""
        errors = []
        if not maptiles.player_pos:
            errors.append("No player position found")
        elif (maptiles.player_pos.x < 0 or maptiles.player_pos.x >= maptiles.width or
              maptiles.player_pos.y < 0 or maptiles.player_pos.y >= maptiles.height):
            errors.append(f"Player position out of bounds: ({maptiles.player_pos.x}, {maptiles.player_pos.y})")
        return errors
    
    def validate_border_walls(self, maptiles: MapTiles) -> List[str]:
        """Validate all edges are walls"""
        errors = []
        wall_positions = {(w.x, w.y) for w in maptiles.walls}
        
        # Check all border positions
        for x in range(maptiles.width):
            if (x, 0) not in wall_positions:
                errors.append(f"Missing wall at top border: ({x}, 0)")
            if (x, maptiles.height - 1) not in wall_positions:
                errors.append(f"Missing wall at bottom border: ({x}, {maptiles.height - 1})")
        
        for y in range(maptiles.height):
            if (0, y) not in wall_positions:
                errors.append(f"Missing wall at left border: (0, {y})")
            if (maptiles.width - 1, y) not in wall_positions:
                errors.append(f"Missing wall at right border: ({maptiles.width - 1}, {y})")
        
        return errors
    
    def validate_no_overlaps(self, maptiles: MapTiles) -> List[str]:
        """Validate no entities occupy same position"""
        errors = []
        
        wall_positions = {(w.x, w.y) for w in maptiles.walls}
        enemy_positions = {(e.x, e.y) for e in maptiles.enemies}
        player_position = {(maptiles.player_pos.x, maptiles.player_pos.y)}
        
        # Check overlaps
        if wall_positions & player_position:
            errors.append("Player overlaps with wall")
        
        if wall_positions & enemy_positions:
            errors.append("Enemy overlaps with wall")
        
        if player_position & enemy_positions:
            errors.append("Player overlaps with enemy")
        
        return errors
    
    def validate_reachability(self, maptiles: MapTiles) -> List[str]:
        """Validate player can reach all enemies"""
        errors = []
        
        wall_positions = {(w.x, w.y) for w in maptiles.walls}
        start = (maptiles.player_pos.x, maptiles.player_pos.y)
        
        # BFS pathfinding
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            x, y = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < maptiles.width and 0 <= ny < maptiles.height and
                    (nx, ny) not in wall_positions and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        
        # Check if all enemies are reachable
        unreachable_enemies = []
        for enemy in maptiles.enemies:
            if (enemy.x, enemy.y) not in visited:
                unreachable_enemies.append((enemy.x, enemy.y))
        
        if unreachable_enemies:
            errors.append(f"Unreachable enemies at: {unreachable_enemies}")
        
        return errors
    
    def validate_level(self, maptiles: MapTiles) -> ValidationResult:
        """Complete level validation"""
        errors = []
        warnings = []
        
        errors.extend(self.validate_dimensions(maptiles))
        errors.extend(self.validate_single_player(maptiles))
        errors.extend(self.validate_border_walls(maptiles))
        errors.extend(self.validate_no_overlaps(maptiles))
        errors.extend(self.validate_reachability(maptiles))
        
        # Warnings
        if len(maptiles.enemies) < 1:
            warnings.append("No enemies found")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

def validate_dataset(dataset_file: str, output_file: str = "validation_results.json"):
    """Validate entire dataset"""
    validator = LevelValidator()
    results = []
    
    print(f"Validating dataset: {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                response_data = data["response"]
                
                # Convert to MapTiles
                walls = [Position(x=pos["x"], y=pos["y"]) for pos in response_data["walls"]]
                enemies = [Position(x=pos["x"], y=pos["y"]) for pos in response_data["enemies"]]
                player_pos = Position(x=response_data["player_pos"]["x"], y=response_data["player_pos"]["y"])
                
                maptiles = MapTiles(
                    width=response_data["width"],
                    height=response_data["height"],
                    walls=walls,
                    enemies=enemies,
                    player_pos=player_pos
                )
                
                validation_result = validator.validate_level(maptiles)
                
                result = {
                    "line_number": line_num,
                    "generation_id": data.get("generation_id", line_num),
                    "is_valid": validation_result.is_valid,
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
                
                results.append(result)
                
                if line_num % 10 == 0:
                    print(f"Validated {line_num} samples...")
                
            except Exception as e:
                print(f"Error validating line {line_num}: {e}")
                results.append({
                    "line_number": line_num,
                    "generation_id": None,
                    "is_valid": False,
                    "errors": [f"Parsing error: {str(e)}"],
                    "warnings": []
                })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    valid_count = sum(1 for r in results if r["is_valid"])
    total_count = len(results)
    
    print(f"\nValidation Summary:")
    print(f"Total samples: {total_count}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {total_count - valid_count}")
    print(f"Validation rate: {valid_count/total_count*100:.2f}%")
    print(f"Results saved to: {output_file}")