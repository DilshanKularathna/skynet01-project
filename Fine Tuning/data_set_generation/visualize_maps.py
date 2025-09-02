import json
from dataclasses import dataclass
from typing import List

@dataclass
class Position:
    x: int
    y: int

@dataclass
class MapTiles:
    width: int
    height: int
    walls: List[Position]
    enemies: List[Position]
    player_pos: Position

def _serialize_maptile(maptile: MapTiles) -> list[str]:
    """Convert MapTiles back to visual representation (same as map.py)"""
    ret = [
        ['.'] * maptile.width for _ in range(maptile.height)
    ]
    for wall in maptile.walls:
        ret[wall.y][wall.x] = 'B'
    for enemy in maptile.enemies:
        ret[enemy.y][enemy.x] = 'E'
    ret[maptile.player_pos.y][maptile.player_pos.x] = 'P'
    return [
        ''.join(row) for row in ret
    ]

def json_to_maptiles(json_data: dict) -> MapTiles:
    """Convert JSON response back to MapTiles object"""
    walls = [Position(x=pos["x"], y=pos["y"]) for pos in json_data["walls"]]
    enemies = [Position(x=pos["x"], y=pos["y"]) for pos in json_data["enemies"]]
    player_pos = Position(x=json_data["player_pos"]["x"], y=json_data["player_pos"]["y"])
    
    return MapTiles(
        width=json_data["width"],
        height=json_data["height"],
        walls=walls,
        enemies=enemies,
        player_pos=player_pos
    )

def load_validation_results(validation_file: str = "validation_results.json") -> dict:
    """Load validation results to show which maps are valid/invalid"""
    try:
        with open(validation_file, 'r') as f:
            return {result["generation_id"]: result for result in json.load(f)}
    except FileNotFoundError:
        print(f"Warning: {validation_file} not found. Will show maps without validation info.")
        return {}

def visualize_all_maps(dataset_file: str = "rpg_training_dataset.jsonl", 
                      output_file: str = "visual_maps.txt"):
    """Convert all generated maps to visual format"""
    
    print(f"Loading maps from: {dataset_file}")
    validation_results = load_validation_results()
    
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as out_f:  # Added encoding
        out_f.write("=" * 80 + "\n")
        out_f.write("GENERATED RPG MAPS VISUALIZATION\n")
        out_f.write("=" * 80 + "\n\n")
        
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
                generation_id = data.get("generation_id", line_num)
                
                # Get validation info
                validation_info = validation_results.get(generation_id, {})
                is_valid = validation_info.get("is_valid", "Unknown")
                errors = validation_info.get("errors", [])
                warnings = validation_info.get("warnings", [])
                
                # Convert to MapTiles
                response_data = data["response"]
                maptiles = json_to_maptiles(response_data)
                
                # Generate visual map
                visual_map = _serialize_maptile(maptiles)
                
                # Write to file - REMOVED EMOJIS
                out_f.write(f"MAP #{generation_id} (Line {line_num})\n")
                if is_valid:
                    out_f.write("Status: VALID\n")
                elif is_valid == False:
                    out_f.write("Status: INVALID\n")
                else:
                    out_f.write("Status: UNKNOWN\n")
                
                if errors:
                    out_f.write(f"Errors: {'; '.join(errors)}\n")
                if warnings:
                    out_f.write(f"Warnings: {'; '.join(warnings)}\n")
                
                out_f.write(f"Size: {maptiles.width}x{maptiles.height}\n")
                out_f.write(f"Enemies: {len(maptiles.enemies)}\n")
                out_f.write(f"Player: ({maptiles.player_pos.x}, {maptiles.player_pos.y})\n")
                out_f.write("-" * 40 + "\n")
                
                # Write the visual map
                for row in visual_map:
                    out_f.write(row + "\n")
                
                out_f.write("-" * 40 + "\n\n")
                
                print(f"Processed map {generation_id} ({'Valid' if is_valid else 'Invalid' if is_valid == False else 'Unknown'})")
                
            except Exception as e:
                out_f.write(f"ERROR processing line {line_num}: {str(e)}\n\n")
                print(f"Error processing line {line_num}: {e}")
    
    print(f"\nVisualization complete! Check: {output_file}")

def visualize_invalid_maps_only(dataset_file: str = "rpg_training_dataset.jsonl", 
                               output_file: str = "invalid_maps_only.txt"):
    """Show only the invalid maps for easier debugging"""
    
    print(f"Loading maps from: {dataset_file}")
    validation_results = load_validation_results()
    
    with open(dataset_file, 'r') as f:
        lines = f.readlines()
    
    invalid_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:  # Added encoding
        out_f.write("=" * 80 + "\n")
        out_f.write("INVALID RPG MAPS ONLY\n")
        out_f.write("=" * 80 + "\n\n")
        
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
                generation_id = data.get("generation_id", line_num)
                
                # Get validation info
                validation_info = validation_results.get(generation_id, {})
                is_valid = validation_info.get("is_valid", True)  # Default to valid if no validation
                
                # Skip valid maps
                if is_valid:
                    continue
                
                invalid_count += 1
                errors = validation_info.get("errors", [])
                warnings = validation_info.get("warnings", [])
                
                # Convert to MapTiles
                response_data = data["response"]
                maptiles = json_to_maptiles(response_data)
                
                # Generate visual map
                visual_map = _serialize_maptile(maptiles)
                
                # Write to file - REMOVED EMOJIS
                out_f.write(f"INVALID MAP #{generation_id} (Line {line_num})\n")
                out_f.write(f"ERRORS: {'; '.join(errors)}\n")
                if warnings:
                    out_f.write(f"WARNINGS: {'; '.join(warnings)}\n")
                
                out_f.write(f"Size: {maptiles.width}x{maptiles.height}\n")
                out_f.write(f"Enemies: {len(maptiles.enemies)}\n")
                out_f.write(f"Player: ({maptiles.player_pos.x}, {maptiles.player_pos.y})\n")
                out_f.write("-" * 40 + "\n")
                
                # Write the visual map
                for row in visual_map:
                    out_f.write(row + "\n")
                
                out_f.write("-" * 40 + "\n\n")
                
            except Exception as e:
                out_f.write(f"ERROR processing line {line_num}: {str(e)}\n\n")
                invalid_count += 1
    
    print(f"\nFound {invalid_count} invalid maps. Check: {output_file}")

if __name__ == "__main__":
    # Visualize all maps
    visualize_all_maps(
        dataset_file="rpg_training_dataset.jsonl",
        output_file="all_maps_visual.txt"
    )
    
    # Show only invalid maps for debugging
    visualize_invalid_maps_only(
        dataset_file="rpg_training_dataset.jsonl", 
        output_file="invalid_maps_debug.txt"
    )
    
    print("\nFiles created:")
    print("- all_maps_visual.txt (All maps with validation status)")
    print("- invalid_maps_debug.txt (Only invalid maps for debugging)")