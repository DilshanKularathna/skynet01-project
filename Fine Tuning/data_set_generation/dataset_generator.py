from xml.parsers.expat import model
import openai
import json
import time
from dataclasses import dataclass, asdict
from typing import List
import os

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

class DatasetGenerator:
    def __init__(self, api_key: str, model: str = "gpt-4.1"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        
    def generate_level_prompt(self) -> str:
        """Use the EXACT prompt from map.py"""
        return """Generate a tilemap for a game level, where all the edges should be walls
there should only be *ONE* player and multiple enemies, all enemies should be placed
randomly and the player should be able to reach all enemies. Make sure to place some
walls inside the level and place the player near the center of the level. (w*h should be 20*15)"""

    def call_openai_gpt4(self, prompt: str) -> dict:
        """Call OpenAI GPT-4 to generate map data"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates game level data. Return your response as a JSON object with the structure: {\"width\": 20, \"height\": 15, \"walls\": [{\"x\": 0, \"y\": 0}, ...], \"enemies\": [{\"x\": 3, \"y\": 2}, ...], \"player_pos\": {\"x\": 10, \"y\": 7}}"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
            
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None

    def convert_to_maptiles(self, json_data: dict) -> MapTiles:
        """Convert JSON response to MapTiles object"""
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

    def generate_dataset(self, num_samples: int = 100, output_file: str = "rpg_dataset.jsonl"):
        """Generate dataset of RPG levels"""
        print(f"Generating {num_samples} RPG levels...")
        
        successful_generations = 0
        failed_generations = 0
        
        with open(output_file, 'w') as f:
            for i in range(num_samples):
                print(f"Generating sample {i+1}/{num_samples}...")
                
                prompt = self.generate_level_prompt()
                response = self.call_openai_gpt4(prompt)
                
                if response:
                    try:
                        maptiles = self.convert_to_maptiles(response)
                        
                        # Create training example
                        training_example = {
                            "prompt": prompt,
                            "response": asdict(maptiles),
                            "generation_id": i+1,
                            "timestamp": time.time()
                        }
                        
                        # Write to JSONL file
                        f.write(json.dumps(training_example) + "\n")
                        f.flush()
                        
                        successful_generations += 1
                        
                    except Exception as e:
                        print(f"Error processing response for sample {i+1}: {e}")
                        failed_generations += 1
                else:
                    failed_generations += 1
                
                # Rate limiting
                time.sleep(1)
        
        print(f"Dataset generation complete!")
        print(f"Successful: {successful_generations}")
        print(f"Failed: {failed_generations}")
        print(f"Dataset saved to: {output_file}")