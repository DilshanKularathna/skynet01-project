from dataset_generator import DatasetGenerator

def main():
    # Set your OpenAI API key here
    API_KEY = "API_KEY"
    
    # Generate dataset only
    print("Starting dataset generation...")
    generator = DatasetGenerator(API_KEY)
    generator.generate_dataset(
        num_samples=500,  # Change this number as needed
        output_file="rpg_training_dataset_gpt4_1.jsonl"
    )
    print("Generation complete! Run validate_only.py to validate.")

if __name__ == "__main__":
    main()