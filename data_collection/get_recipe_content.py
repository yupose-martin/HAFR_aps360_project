import argparse
import json
import os

from api import AllRecipes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get recipe content")
    parser.add_argument(
        "path_to_recipe_list",
        type=str,
        help="Path to the JSON file containing a list of recipes",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory where the recipe content will be saved",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing recipe content",
    )

    args = parser.parse_args()

    with open(args.path_to_recipe_list, "r", encoding="utf-8") as f:
        recipe_name_to_url = json.load(f)

    if args.overwrite:
        remaining_recipes_file_path = args.path_to_recipe_list
    else:
        remaining_recipes_file_path = os.path.join(
            args.output_dir, "remaining_recipes.json"
        )

    # Get recipe content for each recipe
    num_recipes_done = 0
    recipes_remaining = set(recipe_name_to_url.keys())

    try:
        for recipe_name, recipe_url in recipe_name_to_url.items():
            print(f"Processing recipe {num_recipes_done + 1}/{len(recipe_name_to_url)}")
            detailed_recipe = AllRecipes.get(
                recipe_url,
                args.output_dir,
                download_images=False,
                get_comments=False,
                get_descriptions=False,
                get_steps=False,
                get_ingredients=False,
            )
            num_recipes_done += 1
            recipes_remaining.remove(recipe_name)
    except KeyboardInterrupt:
        print("Interrupted by user")
        remaining_recipes_to_url = {
            recipe: recipe_name_to_url[recipe] for recipe in recipes_remaining
        }
        with open(remaining_recipes_file_path, "w", encoding="utf-8") as f:
            json.dump(remaining_recipes_to_url, f, ensure_ascii=False, indent=4)
        print(f"Remaining recipes saved to {remaining_recipes_file_path}")

    print(f"Processed {num_recipes_done} recipes")
    print("Process complete")
