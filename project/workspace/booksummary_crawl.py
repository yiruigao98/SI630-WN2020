import requests
from bs4 import BeautifulSoup
import json
import csv

link_file = "novel_links.txt"
output_file_base = "summary_"
CACHE_FNAME = 'novelsummaries.json'

# Caching:
def get_unique_key(url):
    return url

def make_request_using_cache(url):
    unique_ident = get_unique_key(url)
    if unique_ident in CACHE_DICTION:
        print("Getting cached data...")
        return CACHE_DICTION[unique_ident]
    else:
        print("Making a request for new data...")
        resp = requests.get(url)
        CACHE_DICTION[unique_ident] = resp.text
        dumped_json_cache = json.dumps(CACHE_DICTION)
        fw = open(CACHE_FNAME,"w", encoding='utf-8')
        fw.write(dumped_json_cache)
        fw.close() 
    return CACHE_DICTION[unique_ident]

def get_summary(page_soup):
    summary_soup = page_soup.find_all("p")
    summary_text = [s.text.strip() for s in summary_soup[:-2]]
    summary_text = " ".join(summary_text)
    print(summary_text)
    return summary_text

def read_file(filename):
    handle = open(filename, "r", encoding="utf-8")
    fullconts = handle.read().split("\n")
    handle.close()
    return fullconts


def get_bookname(link):
    bookname = link.split('/')[-2]
    return bookname


try:
    cache_file = open(CACHE_FNAME, 'r', encoding='utf-8')
    cache_contents = cache_file.read()
    CACHE_DICTION = json.loads(cache_contents)
    cache_file.close()
except:
    CACHE_DICTION = {}

book_links = read_file(link_file)
book_links = [l.replace("characters","summary") for l in book_links]
print(book_links)

for link in book_links:
    book_name = get_bookname(link)
    output_file = "summaries/{}{}.txt".format(output_file_base, book_name)
    page_text = make_request_using_cache(link)
    page_soup = BeautifulSoup(page_text, 'html.parser')
    summary_text = get_summary(page_soup)
    with open(output_file,'w', encoding="utf-8") as f:
        f.write(summary_text)
# brands = get_brand(page_soup)
