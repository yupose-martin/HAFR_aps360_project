import json
import os

from data_collection.api import AllRecipes

# Set directory to save results
save_dir = os.path.join(os.getcwd(), "data")
os.makedirs(save_dir, exist_ok=True)

# Search all broad categories
recipe_categories = AllRecipes.get_recipe_categories()

print(f"Got {len(recipe_categories)} categories")
# print(recipe_categories)

# Search for specific recipes in each category
all_recipe_names = set()
recipe_name_to_url = dict()
for category in recipe_categories:
    search_string = category  # Query
    query_result = AllRecipes.search(search_string)

    print(f"Got {len(query_result)} results for {category}")
    found_recipe_names = [result["name"] for result in query_result]
    all_recipe_names.update(found_recipe_names)
    recipe_name_to_url.update(
        {result["name"]: result["url"] for result in query_result}
    )
    current_num_recipes = len(all_recipe_names)

print(f"Got {len(all_recipe_names)} unique recipes")

# Save all recipe names and URLs in a JSON file
with open(os.path.join(save_dir, "all_recipe_names.json"), "w", encoding="utf-8") as f:
    json.dump(recipe_name_to_url, f, ensure_ascii=False, indent=4)


# # Get:
# main_recipe_url = query_result[0]["url"]

# print(f"Got {len(query_result)} results")
# print(f"First result: {main_recipe_url}")

# detailed_recipe = AllRecipes.get(
#     main_recipe_url,
#     save_dir,
#     download_images=True,
#     get_comments=True,
# )  # Get the details of the first returned recipe (most relevant in our case)

# print(detailed_recipe)

# allrecipes_wrapper = AllRecipes()

# comments = allrecipes_wrapper.get_comments(main_recipe_url)

# print(f"Got {len(comments)} comments")

# for i in range(5):
#     print(f"Comment {i}: {comments[i]}")
