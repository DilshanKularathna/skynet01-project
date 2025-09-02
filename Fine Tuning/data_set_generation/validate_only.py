from level_validator import validate_dataset

def main():
    print("Starting validation...")
    validate_dataset(
        dataset_file="rpg_training_dataset_gpt4_1.jsonl",
        output_file="validation_results_gpt4_1.json"
    )

if __name__ == "__main__":
    main()