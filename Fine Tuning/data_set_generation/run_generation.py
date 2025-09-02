from dataset_generator import DatasetGenerator
from level_validator import validate_dataset

def main():
    # Set your OpenAI API key here
    API_KEY = "API_KEY"  # Replace with your actual key
    
    # Generate dataset
    print("Starting dataset generation...")
    generator = DatasetGenerator(API_KEY)
    generator.generate_dataset(
        num_samples=50,  # Start with 50 samples for testing
        output_file="rpg_training_dataset.jsonl"
    )
    
    # Validate dataset
    print("\nStarting validation...")
    validate_dataset(
        dataset_file="rpg_training_dataset.jsonl",
        output_file="validation_results.json"
    )

if __name__ == "__main__":
    main()