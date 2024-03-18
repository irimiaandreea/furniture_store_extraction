import csv
import sys
import requests
from bs4 import BeautifulSoup
import re


class CustomError(Exception):
    def __init__(self, message="An error occurred"):
        self.message = message
        super().__init__(self.message)


def process_input_data(input_file, start_valid_url, limit, training_data):
    count_current_page = 0
    if training_data:
        output_file = "crawled_train_data.txt"
    else:
        output_file = "crawled_test_data.txt"

    with open(output_file, "w") as output_file:
        # read the input_csv_file line by line
        with open(input_file, "r") as input_file:
            csv_reader = csv.reader(input_file, delimiter="\t")

            # skip the header
            next(csv_reader)

            count_successful_pages = 0
            for url in csv_reader:
                crawled_data_response = crawl_data_from_page(url[0])
                if crawled_data_response is not None:
                    count_current_page += 1

                    if count_current_page >= start_valid_url:
                        output_file.write(crawled_data_response + '\n')
                        count_successful_pages += 1

                if limit == count_successful_pages:
                    break

            input_file.close()

            try:
                if limit > count_successful_pages:
                    raise CustomError("The limit is bigger than the length of lines of URLs in .csv file! \n")
            except CustomError as custom_exception:
                print("An error occurred: ", custom_exception)
                sys.exit()

        output_file.close()


def crawl_data_from_page(current_url):
    page_response = requests.Response()

    try:
        # the variable page_response will contain the entire HTML of the page
        page_response = requests.get(current_url)
        content_type_response = page_response.headers.get('content-type')
        status_code_response = page_response.status_code

        if status_code_response == 200:
            if 'text/html' in content_type_response:
                soup_response = BeautifulSoup(page_response.content,
                                              "html.parser")  # page_response.text is the content of the response in Unicode, and page_response.content is the content of the response in bytes
                crawled_data_response = ' '.join(
                    [element.get_text() for element in
                     soup_response.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])])  # include <span>, <a> togs

                if len(crawled_data_response) != 0:
                    # remove special characters, extra-spaces
                    crawled_data_response = re.sub(r"[^a-zA-Z0-9\s]", "", crawled_data_response)
                    crawled_data_response = crawled_data_response.lower().strip()
                    crawled_data_response = " ".join(crawled_data_response.split())

                    return crawled_data_response

        return None

    except requests.exceptions.RequestException as exception:
        page_response.status_code = exception
        return None


if __name__ == '__main__':
    input_file = "../../datasets/furniture_stores_pages.csv"
    start_valid_url = 100
    limit = 100
    training_data = False
    process_input_data(input_file, start_valid_url, limit, training_data)
