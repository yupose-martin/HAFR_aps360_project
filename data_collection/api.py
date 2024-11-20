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
    def get(cls, url, save_path, download_images=False, get_comments=False):
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

        return data
