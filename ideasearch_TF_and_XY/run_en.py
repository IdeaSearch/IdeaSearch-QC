# run_en.py - Main IdeaSearch execution script for TF_and_XY quantum circuit optimization
# English Version

import yaml
from pathlib import Path
from IdeaSearch import IdeaSearcher
from user_code.prompt import prologue_section, epilogue_section
from user_code.evaluation import evaluate
from user_code.initial_ideas import initial_ideas


def load_config(config_path: str = "config_en.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration from YAML file
    print("Loading configuration from config_en.yaml...")
    config = load_config()
    
    # Extract configuration parameters
    model_type = config['model']['type']
    project_name = config['project']['name']
    api_keys_path = config['paths']['api_keys']
    database_base = config['paths']['database']
    database_path = f"{database_base}_{model_type}"
    language = config['language']['interface']
    
    # LLM configuration
    models = config['llm']['models']
    temperatures = config['llm']['temperatures']
    examples_num = config['llm']['examples_num']
    sample_temperature = config['llm']['sample_temperature']
    
    # Search parameters
    island_num = config['search']['island_num']
    cycle_num = config['search']['cycle_num']
    unit_interaction_num = config['search']['unit_interaction_num']
    samplers_num = config['search']['samplers_num']
    evaluators_num = config['search']['evaluators_num']
    generate_num = config['search']['generate_num']
    
    # Scoring parameters
    hand_over_threshold = config['scoring']['hand_over_threshold']
    score_range = (config['scoring']['score_range']['min'], 
                   config['scoring']['score_range']['max'])
    
    # Logging
    record_prompt = config['logging']['record_prompt_in_diary']
    
    # Display configuration
    print(f"\n{'='*60}")
    print(f"IdeaSearch Configuration")
    print(f"{'='*60}")
    print(f"Project: {project_name}")
    print(f"Model Type: {model_type}")
    print(f"Database: {database_path}")
    print(f"Islands: {island_num}")
    print(f"Cycles: {cycle_num}")
    print(f"Rounds per cycle: {unit_interaction_num}")
    print(f"LLM Models: {models}")
    print(f"{'='*60}\n")
    
    # Create evaluation function with model_type
    def evaluate_type(idea_code: str) -> tuple:
        return evaluate(idea_code, model_type=model_type)
    
    # 1. Initialize IdeaSearcher
    ideasearcher = IdeaSearcher()
    
    # 2. Set language
    ideasearcher.set_language(language)
    
    # 3. Load models - set API keys path
    ideasearcher.set_api_keys_path(api_keys_path)
    
    # 4. Set program name and database path
    ideasearcher.set_program_name(project_name)
    ideasearcher.set_database_path(database_path)
    
    # 5. Set evaluation function
    ideasearcher.set_evaluate_func(evaluate_type)
    
    # 6. Set prompt sections
    ideasearcher.set_prologue_section(prologue_section)
    ideasearcher.set_epilogue_section(epilogue_section)
    
    # 7. Configure models to use
    ideasearcher.set_models(models)
    ideasearcher.set_model_temperatures(temperatures)
    
    # 8. Configure additional parameters
    ideasearcher.set_examples_num(examples_num)
    ideasearcher.set_samplers_num(samplers_num)
    ideasearcher.set_evaluators_num(evaluators_num)
    ideasearcher.set_generate_num(generate_num)
    ideasearcher.set_sample_temperature(sample_temperature)
    ideasearcher.set_hand_over_threshold(hand_over_threshold)
    ideasearcher.set_score_range(score_range)
    ideasearcher.set_record_prompt_in_diary(record_prompt)
    
    # 9. Add initial ideas programmatically
    if config['initial_ideas']['enabled']:
        ideasearcher.add_initial_ideas(initial_ideas)
    
    # 10. Create islands
    print(f"Creating {island_num} islands...")
    for _ in range(island_num):
        ideasearcher.add_island()
    
    # 11. Run the evolutionary search
    print(f"\nStarting evolutionary search: {cycle_num} cycles, {unit_interaction_num} rounds per cycle\n")
    
    for cycle in range(1, cycle_num + 1):
        print(f"\n{'='*60}")
        print(f"Cycle {cycle}/{cycle_num}")
        print(f"{'='*60}")
        
        # Perform island migration (except for the first cycle)
        if cycle > 1:
            print("Performing island migration...")
            ideasearcher.repopulate_islands()
        
        # Run evolution for specified number of rounds
        ideasearcher.run(unit_interaction_num)
        
        # Get and display best result so far
        best_idea = ideasearcher.get_best_idea()
        best_score = ideasearcher.get_best_score()
        
        print(f"\n{'='*60}")
        print(f"[Cycle {cycle} Complete]")
        print(f"Current Best Score: {best_score:.2f}")
        print(f"Best Idea Code:")
        print(f"{'-'*60}")
        print(best_idea)
        print(f"{'='*60}\n")
    
    # 12. Final results
    print("\n" + "="*60)
    print("Search Complete!")
    print("="*60)
    
    final_best_idea = ideasearcher.get_best_idea()
    final_best_score = ideasearcher.get_best_score()
    
    print(f"\nFinal Best Score: {final_best_score:.2f}")
    print(f"\nFinal Best Idea:")
    print("-"*60)
    print(final_best_idea)
    print("-"*60)
    
    print("\nProgram execution completed!")
    print(f"Results saved to: {database_path}")


if __name__ == "__main__":
    main()
