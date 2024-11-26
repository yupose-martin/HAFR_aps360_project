# -*- coding: utf-8 -*-

import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import TypedDict

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class CommentData(TypedDict, total=False):
    username: str
    user_link: str
    rating: int
    date: str
    comment: str
    helpful_count: int


class AllRecipes(object):
    """
    A class to fetch recipes from AllRecipes.com using web scraping.
    """

    @staticmethod
    def get_recipe_categories():
        # URL of the AllRecipes A-Z page
        url = "https://www.allrecipes.com/recipes-a-z-6735880"

        try:
            # Set up the request
            req = urllib.request.Request(url)
            req.add_header(
                "Cookie", "euConsent=true"
            )  # Add a cookie header if required

            # Create an HTTPS handler with an unverified context
            handler = urllib.request.HTTPSHandler(
                context=ssl._create_unverified_context()
            )
            opener = urllib.request.build_opener(handler)

            # Open the URL and read the response
            response = opener.open(req)
            html_content = response.read()

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # List to store recipe names
            recipe_names = []

            # Find all recipe elements
            recipe_elements = soup.find_all("a", {"class": "mntl-link-list__link"})
            for element in recipe_elements:
                recipe_name = element.get_text(
                    strip=True
                )  # Extract and clean the recipe name
                recipe_names.append(recipe_name)

            return recipe_names

        except Exception as e:
            print(f"Error fetching recipes: {e}")
            return []

    @staticmethod
    def search(search_string):
        """
        Search recipes parsing the returned html data.
        """
        base_url = "https://allrecipes.com/search?"
        query_url = urllib.parse.urlencode({"q": search_string})

        url = base_url + query_url

        req = urllib.request.Request(url)
        req.add_header("Cookie", "euConsent=true")

        handler = urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
        opener = urllib.request.build_opener(handler)
        response = opener.open(req)
        html_content = response.read()

        soup = BeautifulSoup(html_content, "html.parser")

        search_data = []
        articles = soup.findAll("a", {"class": "mntl-document-card"})
        articles = [
            a
            for a in articles
            if a["href"].startswith("https://www.allrecipes.com/recipe/")
        ]

        for article in articles:
            data = {}
            try:
                data["name"] = (
                    article.find("span", {"class": "card__title"})
                    .get_text()
                    .strip(" \t\n\r")
                )
                data["url"] = article["href"]
                try:
                    data["rate"] = len(article.find_all("svg", {"class": "icon-star"}))
                    try:
                        if len(article.find_all("svg", {"class": "icon-star-half"})):
                            data["rate"] += 0.5
                    except Exception:
                        pass
                except Exception as e0:
                    data["rate"] = None
                try:
                    data["image"] = article.find("img")["data-src"]
                except Exception as e1:
                    try:
                        data["image"] = article.find("img")["src"]
                    except Exception as e1:
                        pass
                    if "image" not in data:
                        data["image"] = None
            except Exception as e2:
                pass
            if data:
                search_data.append(data)

        return search_data

    @staticmethod
    def sanitize_directory_name(name: str):
        """
        Sanitizes a string to create a valid directory name by replacing spaces with underscores
        and removing invalid characters.

        Args:
            name (str): The original name.

        Returns:
            str: The sanitized name.
        """
        # Replace spaces with underscores
        sanitized_name = name.replace(" ", "_")
        # Remove characters that are not alphanumeric, dashes, or underscores
        sanitized_name = re.sub(r"[^\w\-_]", "", sanitized_name)
        return sanitized_name

    @staticmethod
    def _get_name(soup):
        return soup.find("h1", {"class": "article-heading"}).get_text().strip(" \t\n\r")

    @staticmethod
    def _get_rating(soup):
        return float(
            soup.find("div", {"class": "mm-recipes-review-bar__rating"})
            .get_text()
            .strip(" \t\n\r")
        )

    # @staticmethod
    # def _get_ingredients(soup):
    #     return [
    #         li.get_text().strip(" \t\n\r")
    #         for li in soup.find(
    #             "div", {"id": "mntl-structured-ingredients_1-0"}
    #         ).find_all("li")
    #     ]

    # @staticmethod
    # def _get_steps(soup):
    #     return [
    #         li.get_text().strip(" \t\n\r")
    #         for li in soup.find("div", {"id": "recipe__steps_1-0"}).find_all("li")
    #     ]

    @staticmethod
    def _get_times_data(soup, text):
        return (
            soup.find("div", {"class": "mm-recipes-details__content"})
            .find("div", text=text)
            .parent.find("div", {"class": "mm-recipes-details__value"})
            .get_text()
            .strip(" \t\n\r")
        )

    @staticmethod
    def get_images(soup: BeautifulSoup, save_path: str, max_number: int = 100):
        """
        Retrieves all <img> tags under a <div> with the class `photo-dialog__page`
        and downloads the images to a specified directory.

        Args:
            soup (BeautifulSoup): BeautifulSoup object parsed from the HTML content.
            save_path (str): Path to the directory where images will be saved.
            max_number (int): Maximum number of images to download. Default is 10.

        Returns:
            list: A list of file paths for the downloaded images.
        """
        # Create the save directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Find the div with the class `photo-dialog__content `
        gallery_div = soup.find("div", {"class": "photo-dialog__content"})
        if not gallery_div:
            print("No gallery div found")
            return []

        # Find all <img> tags within the gallery div
        images = gallery_div.find_all("img")
        if not images:
            print("No images found in the gallery div")
            return []

        if len(images) > max_number:
            print(
                f"Found {len(images)} images. Downloading the first {max_number} images."
            )
            images = images[:max_number]
        else:
            print(f"Found {len(images)} images. Downloading all images.")

        # Download each image
        downloaded_files = []
        for i, img in enumerate(images):
            print(f"Downloading image {i + 1}...")
            # Get the src attribute of the img tag
            img_src = img.get("data-src")
            if not img_src:
                print(f"Image {i + 1} has no src attribute. Skipping...")
                continue

            # Generate the file name for saving
            file_name = f"image_{i + 1}.jpg"
            file_path = os.path.join(save_path, file_name)

            try:
                # Download the image
                urllib.request.urlretrieve(img_src, file_path)
                downloaded_files.append(file_path)
                print(f"Downloaded: {file_path}")
            except Exception as e:
                print(f"Failed to download {img_src}: {e}")

        return downloaded_files

    @staticmethod
    def parse_comment(div: BeautifulSoup) -> CommentData:
        """
        Parse a single comment from its HTML element.

        Args:
            div (bs4.element.Tag): The HTML element containing the comment data.

        Returns:
            CommentData: A dictionary with the parsed comment data.
        """
        comment_data: CommentData = {
            "username": None,
            "user_link": None,
            "rating": 0,  # Default rating is 0
            "date": None,
            "comment": None,
            "helpful_count": 0,  # Default helpful count is 0
        }

        # Extract user name
        username_tag = div.find("span", {"class": "feedback__display-name"})
        if username_tag:
            comment_data["username"] = username_tag.get_text(strip=True)
            # Extract user profile link if available
            link_tag = username_tag.find("a")
            comment_data["user_link"] = link_tag["href"] if link_tag else None

        # Extract rating
        stars = div.find_all("svg", {"class": "ugc-icon-star"})
        comment_data["rating"] = len(stars)  # Count filled stars

        # Extract date
        date_tag = div.find("span", {"class": "feedback__meta-date"})
        if date_tag:
            comment_data["date"] = date_tag.get_text(strip=True)

        # Extract comment text
        text_tag = div.find("div", {"class": "feedback__text"})
        if text_tag:
            comment_data["comment"] = text_tag.get_text(strip=True)

        # Extract "helpful" count (if available)
        helpful_tag = div.find("span", {"class": "feedback__helpful-count"})
        if helpful_tag:
            try:
                comment_data["helpful_count"] = int(helpful_tag.get_text(strip=True))
            except ValueError:
                comment_data["helpful_count"] = 0

        return comment_data

    @staticmethod
    def get_comments(
        url: str, save_dir: str, wait_time: float = 0.1
    ) -> list[CommentData]:
        """
        Get all user comments by fully loading them first.

        Args:
            url (str): The URL of the recipe page.
            wait_time (float): Time to wait after clicking "Load More" (in seconds).

        Returns:
            list: A list of comments.
        """

        # Configure Selenium to run efficiently
        options = Options()
        options.add_argument("--headless")  # Run in headless mode for performance
        options.add_argument("--disable-gpu")  # Disable GPU to save resources
        service = Service(
            "/opt/homebrew/bin/chromedriver"
        )  # Set path to your ChromeDriver

        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        driver.implicitly_wait(10)

        try:
            while True:
                # Try to click the "Load More" button
                try:
                    print("Clicking 'Load More' button...")
                    load_more_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable(
                            (By.CLASS_NAME, "feedback-list__load-more-button")
                        )
                    )
                    load_more_button.click()
                    time.sleep(wait_time)  # Wait for new comments to load
                except Exception:
                    print("No more 'Load More' button found. All comments are loaded.")
                    break

            # After all comments are loaded, parse the full page
            soup = BeautifulSoup(driver.page_source, "html.parser")
            comments = [
                AllRecipes.parse_comment(div)
                for div in soup.find_all("div", {"class": "feedback-list__item"})
            ]

        except Exception as e:
            print(f"Error occurred: {e}")
            comments = []

        finally:
            driver.quit()

        # Save comments as a JSON file
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        save_path = os.path.join(save_dir, "comments.json")
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(comments, json_file, ensure_ascii=False, indent=4)

        print(f"Comments saved to {save_path}")

        # Return all comments as a list
        return comments

    @staticmethod
    def get_description(soup: BeautifulSoup, save_path: str) -> str:
        """
        Get the recipe description from the BeautifulSoup object.

        Args:
            soup (BeautifulSoup): BeautifulSoup object parsed from the HTML content.
            save_path (str): Path to the directory where the description will be saved.

        Returns:
            str: The recipe description.
        """
        # Find the meta tag with the name "description"
        description_tag = soup.find("meta", {"name": "description"})
        if description_tag and description_tag.get("content"):
            descriptions = description_tag["content"]
        else:
            descriptions = "No description found."

        # Save the description to a file
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it doesn't exist

        save_path = os.path.join(save_path, "descriptions.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(descriptions)

        print(f"Description saved to {save_path}")

        return descriptions 
    
    @staticmethod
    def get_steps(soup: BeautifulSoup, save_path: str) -> str:
        # Step 1: Find all <script> tags
        script_tags = soup.find_all("script", type="application/ld+json")
        steps = ''
        # Step 2: Loop through script tags to find relevant JSON data
        for script in script_tags:
            try:
                # Attempt to parse JSON from script contents
                json_data = json.loads(script.string)
                
                
                
                # Check if the JSON contains the desired key (e.g., "recipeInstructions")
                if isinstance(json_data, list):  # Sometimes JSON starts with a list
                    for entry in json_data:
                        if "recipeInstructions" in entry:
                            instructions = entry["recipeInstructions"]
                            
                            # Extract the "text" from instructions
                            for step in instructions:
                                # print(step.get("text"))
                                steps += step.get("text") + '\n'
            except (json.JSONDecodeError, TypeError):
                continue  # Skip non-JSON or incompatible <script> tags
            
        # Save the description to a file
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it doesn't exist
            
        save_path = os.path.join(save_path, "steps.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(steps)
        print(f"Steps saved to {save_path}")
        return steps
    
    @staticmethod
    def get_ingredients(soup: BeautifulSoup, save_path: str) -> str:
        # Step 1: Find all <script> tags
        script_tags = soup.find_all("script", type="application/ld+json")
        ingredients = ''
        # Step 2: Loop through script tags to find relevant JSON data
        for script in script_tags:
            try:
                # Attempt to parse JSON from script contents
                json_data = json.loads(script.string)
                
                # Check if the JSON contains the desired key (e.g., "recipeInstructions")
                if isinstance(json_data, list):  # Sometimes JSON starts with a list
                    for entry in json_data:
                        if "recipeIngredient" in entry:
                            recipeIngredients = entry["recipeIngredient"]
                            for ingredient in recipeIngredients:
                                ingredients += ingredient + '\n'
            except (json.JSONDecodeError, TypeError):
                continue  # Skip non-JSON or incompatible <script> tags
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, "ingredients.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(ingredients)
        print(f"Ingredients saved to {save_path}")
        return ingredients
    
    @staticmethod
    def split_ingredient(soup: BeautifulSoup, save_path: str) -> str:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saved_path = os.path.join(save_path, "ingredients.txt")
        
        units = [
        # Common volume measurements
        "teaspoon", "teaspoons", "tsp",
        "tablespoon", "tablespoons", "tbsp",
        "cup", "cups",
        "pint", "pints",
        "quart", "quarts",
        "gallon", "gallons",
        "ml", "milliliter", "milliliters",
        "l", "liter", "liters",
        
        # Common weight measurements
        "ounce", "ounces", "oz",
        "pound", "pounds", "lb", "lbs",
        "gram", "grams", "g",
        "kilogram", "kilograms", "kg",
        
        # Length or size measurements
        "inch", "inches",
        "cm", "centimeter", "centimeters",
        "mm", "millimeter", "millimeters",

        # Descriptive amounts
        "pinch", "pinches",
        "dash", "dashes",
        "handful", "handfuls",
        "stick", "sticks",
        "slice", "slices",
        "clove", "cloves",
        "head", "heads",
        "piece", "pieces",
        "bunch", "bunches",
        "can", "cans",
        "jar", "jars",
        "package", "packages",
        "container", "containers",
        "bag", "bags",
        "block", "blocks",
        "sprig", "sprigs",
        "stalk", "stalks",
        
        # Unusual or less common descriptors
        "drop", "drops",
        "sheet", "sheets",
        "fillet", "fillets",
        "filet", "filets",
        "patty", "patties",
        "loaf", "loaves",
        "roll", "rolls",
        "ball", "balls",
        "cube", "cubes",
        "ring", "rings",
        "strip", "strips",
        "bar", "bars",
        "square", "squares",

        # Time-based descriptors (for rare cases)
        "hour", "hours",
        "minute", "minutes",
        "second", "seconds"
    ]
        # Read the ingredients from the file
        with open(saved_path, "r", encoding="utf-8") as file:
            recipes_list = file.read()

        # Transform the ingredients into a list
        ingredients = recipes_list.strip().split('\n')
        # print(f"ingredients: {ingredients}")
        # Step 1: Remove everything after the first commaa nd inside the parentheses
        cleaned_ingredients = [re.sub(r",.*", "", item).strip() for item in ingredients]
        # Step 1: Remove everything inside parentheses
        cleaned_ingredients = [re.sub(r"\([^)]*\)", "", item).strip() for item in cleaned_ingredients]

        print(f"cleaned_ingredients: {cleaned_ingredients}")
        # Step 3: Function to parse the ingredients
        def parse_ingredient(ingredient):
            #1. match the of like "plenty of salt" "handful of agurula"
            if " of " in ingredient:
                amount, name = ingredient.split(" of ", 1)  # Split on "of"
                return amount.strip(), name.strip()
            
            # 2. Match unit first
            for unit in units:
                pattern = rf"\b(\d*\.?\d+\s*[\w/+-]*)\s*{unit}\b"
                match = re.search(pattern, ingredient)
                if match:
                    amount = match.group(0)
                    name = ingredient[match.end():].strip()
                    return amount.strip(), name
            
            # 3. If no unit, match the number
            number_pattern = r"^\d*\.?\d+[\w/+-]*"
            number_match = re.match(number_pattern, ingredient)
            if number_match:
                amount = number_match.group(0)
                name = ingredient[number_match.end():].strip()
                return amount.strip(), name
            
            # 3. Default: No amount, treat entire string as name
            return None, ingredient

        # Step 4: Process each ingredient
        parsed_ingredients = [parse_ingredient(ingredient) for ingredient in cleaned_ingredients]

        # Step 5: Split the results into two lists
        amounts, names = zip(*parsed_ingredients)

        # Display the results
        print("Amounts:")
        print(amounts)
        print("\nIngredients:")
        print(names)
        
        # Save the amounts and names to a file
        save_path = os.path.join(save_path, "parsed_ingredients.txt")
        with open(save_path, "w", encoding="utf-8") as file:
            for amount, name in zip(amounts, names):
                file.write(f"{amount}|{name}\n")
            print(f"Parsed ingredients saved to {save_path}")
        
    
    @classmethod
    def _get_prep_time(cls, soup):
        return cls._get_times_data(soup, "Prep Time:")

    @classmethod
    def _get_cook_time(cls, soup):
        return cls._get_times_data(soup, "Cook Time:")

    @classmethod
    def _get_total_time(cls, soup):
        return cls._get_times_data(soup, "Total Time:")

    @classmethod
    def _get_nb_servings(cls, soup):
        return cls._get_times_data(soup, "Servings:")

    @classmethod
    def get(cls, url, save_path, download_images=False, get_comments=False, get_descriptions=False, get_steps=False, get_ingredients=False):
        """
        'url' from 'search' method.
        ex. "/recipe/106349/beef-and-spinach-curry/"
        """
        # base_url = "https://allrecipes.com/"
        # url = base_url + uri

        req = urllib.request.Request(url)
        req.add_header("Cookie", "euConsent=true")

        handler = urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
        opener = urllib.request.build_opener(handler)
        response = opener.open(req)
        html_content = response.read()

        soup = BeautifulSoup(html_content, "html.parser")

        elements = [
            {"name": "name", "default_value": ""},
            # {"name": "ingredients", "default_value": []},
            # {"name": "steps", "default_value": []},
            {"name": "rating", "default_value": None},
            {"name": "prep_time", "default_value": ""},
            {"name": "cook_time", "default_value": ""},
            {"name": "total_time", "default_value": ""},
            {"name": "nb_servings", "default_value": ""},
        ]

        data = {"url": url}
        for element in elements:
            try:
                data[element["name"]] = getattr(cls, "_get_" + element["name"])(soup)
            except Exception as e:
                data[element["name"]] = element["default_value"]

            recipe_directory = os.path.join(
                save_path, cls.sanitize_directory_name(data["name"])
            )

        if download_images:
            try:
                recipe_image_directory = os.path.join(recipe_directory, "images")
                image_paths = cls.get_images(soup, recipe_image_directory)
                data["images_path"] = image_paths
            except Exception as e:
                print(f"Failed to download images: {e}")
                data["images_path"] = None

        if get_comments:
            try:
                recipe_comment_directory = os.path.join(recipe_directory, "comments")
                comments = cls.get_comments(url, recipe_comment_directory)
                data["comments_path"] = recipe_comment_directory
            except Exception as e:
                print(f"Failed to get comments: {e}")
                data["comments_path"] = None
                
        if get_descriptions:
            try:
                recipe_description_directory = os.path.join(recipe_directory, "descriptions")
                description = cls.get_description(soup, recipe_description_directory)
                data["descriptions_path"] = recipe_description_directory
            except Exception as e:
                print(f"Failed to get descriptions: {e}")
                data["descriptions_path"] = None
        
        if get_steps:
            try:
                recipe_steps_directory = os.path.join(recipe_directory, "steps")
                steps = cls.get_steps(soup,recipe_steps_directory)
                data["steps"] = recipe_steps_directory
            except Exception as e:
                print(f"Failed to get steps: {e}")
                data["steps"] = None

        if get_ingredients:
            try:
                recipe_ingredients_directory = os.path.join(recipe_directory, "ingredients")
                ingredients = cls.get_ingredients(soup, recipe_ingredients_directory)
                data["ingredients"] = recipe_ingredients_directory
            except Exception as e:
                print(f"Failed to get ingredients: {e}")
                data["ingredients"] = None
                
        #split the downloaded ingredient into amount and ingredient
        # split_ingredients = False
        # if split_ingredients:
        #     try:
        #         recipe_ingredients_directory = os.path.join(recipe_directory, "ingredients")
        #         splited_ingredients = cls.split_ingredient(soup, recipe_ingredients_directory)
        #         data["ingredients"] = recipe_ingredients_directory
        #     except Exception as e:
        #         print(f"Failed to get ingredients: {e}")
        #         data["ingredients"] = None
        
        return data
