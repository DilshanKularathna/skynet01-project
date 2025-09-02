import json
from typing import Dict, List, Any

def filter_valid_maps(dataset_file: str, validation_file: str, output_file: str) -> None:
    """
    Filter out invalid maps from the dataset based on validation results.
    
    Args:
        dataset_file: Path to the original dataset JSONL file
        validation_file: Path to the validation results JSON file
        output_file: Path to save the filtered dataset
    """
    
    # Load validation results
    with open(validation_file, 'r') as f:
        validation_results = json.load(f)
    
    # Create a mapping of line numbers to validation status
    valid_lines = set()
    invalid_count_by_error = {}
    
    for result in validation_results:
        line_number = result['line_number']
        is_valid = result['is_valid']
        errors = result.get('errors', [])
        
        if is_valid:
            valid_lines.add(line_number)
        else:
            # Count different types of errors for statistics
            for error in errors:
                if error not in invalid_count_by_error:
                    invalid_count_by_error[error] = 0
                invalid_count_by_error[error] += 1
    
    # Filter the dataset
    valid_maps = []
    total_maps = 0
    
    with open(dataset_file, 'r') as f:
        for line_number, line in enumerate(f, 1):
            total_maps += 1
            if line_number in valid_lines:
                try:
                    map_data = json.loads(line.strip())
                    valid_maps.append(map_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {line_number}")
                    continue
    
    # Save filtered dataset
    with open(output_file, 'w') as f:
        for map_data in valid_maps:
            f.write(json.dumps(map_data) + '\n')
    
    # Print statistics
    print(f"Dataset Filtering Results:")
    print(f"=" * 50)
    print(f"Total maps in original dataset: {total_maps}")
    print(f"Valid maps: {len(valid_maps)}")
    print(f"Invalid maps removed: {total_maps - len(valid_maps)}")
    print(f"Success rate: {len(valid_maps)/total_maps*100:.2f}%")
    print()
    
    if invalid_count_by_error:
        print("Error breakdown:")
        for error_type, count in sorted(invalid_count_by_error.items()):
            print(f"  - {error_type}: {count} maps")
    
    print(f"\nFiltered dataset saved to: {output_file}")

def analyze_dataset_quality(validation_file: str) -> Dict[str, Any]:
    """
    Analyze the quality of the generated dataset.
    
    Args:
        validation_file: Path to the validation results JSON file
    
    Returns:
        Dictionary with analysis results
    """
    with open(validation_file, 'r') as f:
        validation_results = json.load(f)
    
    total_maps = len(validation_results)
    valid_maps = sum(1 for result in validation_results if result['is_valid'])
    invalid_maps = total_maps - valid_maps
    
    # Count error types
    error_counts = {}
    for result in validation_results:
        if not result['is_valid']:
            for error in result.get('errors', []):
                if error not in error_counts:
                    error_counts[error] = 0
                error_counts[error] += 1
    
    # Calculate success rate
    success_rate = (valid_maps / total_maps) * 100 if total_maps > 0 else 0
    
    analysis = {
        'total_maps': total_maps,
        'valid_maps': valid_maps,
        'invalid_maps': invalid_maps,
        'success_rate': success_rate,
        'error_breakdown': error_counts
    }
    
    return analysis

def main():
    """Main function to filter the dataset."""
    
    # File paths
    dataset_file = "rpg_training_dataset_gpt4_1.jsonl"
    validation_file = "validation_results_gpt4_1.json"
    output_file = "rpg_training_dataset_gpt4_1_filtered.jsonl"
    
    print("Starting dataset filtering process...")
    print()
    
    # Analyze dataset quality first
    analysis = analyze_dataset_quality(validation_file)
    
    print("Dataset Quality Analysis:")
    print(f"=" * 50)
    print(f"Total generated maps: {analysis['total_maps']}")
    print(f"Valid maps: {analysis['valid_maps']}")
    print(f"Invalid maps: {analysis['invalid_maps']}")
    print(f"Overall success rate: {analysis['success_rate']:.2f}%")
    print()
    
    if analysis['error_breakdown']:
        print("Common error types:")
        sorted_errors = sorted(analysis['error_breakdown'].items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors:
            percentage = (count / analysis['total_maps']) * 100
            print(f"  - {error_type}: {count} ({percentage:.1f}%)")
    
    print()
    print("Filtering dataset to keep only valid maps...")
    print()
    
    # Filter the dataset
    filter_valid_maps(dataset_file, validation_file, output_file)
    
    print("\nDataset filtering completed successfully!")

if __name__ == "__main__":
    main()