import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin


# Function to extract all HTML pages from a given URL
def extract_html_pages(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the webpage
        soup = BeautifulSoup(response.text, "html.parser")

        # Create a directory to store HTML pages if it doesn't exist
        if not os.path.exists("html_pages"):
            os.makedirs("html_pages")

        # Extract and save all HTML pages
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".html") and not href.startswith("#"):
                html_page_url = urljoin(url, href)
                html_page_response = requests.get(html_page_url)
                if html_page_response.status_code == 200:
                    # Save the HTML content to a file
                    filename = href.split("/")[-1]  # Extract filename from the URL
                    with open(
                        f"html_pages/{filename}", "w", encoding="utf-8"
                    ) as html_file:
                        html_file.write(html_page_response.text)
                        print(f"HTML page '{filename}' saved successfully.")
                else:
                    print(f"Failed to retrieve HTML page: {html_page_url}")
    else:
        print(f"Failed to retrieve webpage: {url}")


def fetch_wikipedia_page(url):
    # Send a GET request to fetch the page content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # Return the parsed soup object
        return soup
    else:
        print("Failed to fetch Wikipedia page. Status code:", response.status_code)
        return None


def save_html_to_file(html_content, file_path):
    # Write the HTML content to a file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(html_content))


# URL of the Wikipedia page you want to fetch
wiki_url = "https://en.wikipedia.org/wiki/India"
# Call the fetch_wikipedia_page function to fetch the page content
page_soup = fetch_wikipedia_page(wiki_url)

if page_soup:
    # Save the entire Wikipedia page as an HTML file
    save_html_to_file(page_soup, "html_pages-tmp/wikipedia_page.html")
    print("Wikipedia page saved successfully as 'wikipedia_page.html'.")
else:
    print("Failed to fetch Wikipedia page.")
