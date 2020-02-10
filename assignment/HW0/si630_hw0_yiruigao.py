import re
import csv

input_file = "W20_webpages.txt"
output_file = "email-outputs.csv"

with open(input_file, 'r') as file:
    rows = file.readlines()

pattern = re.compile(r'[^@\s<>]+\s*@+\s*[a-zA-Z0-9\-]+\s*\.+\s*(com|net|org|gov)[^a-zA-Z]')
pattern_dot = re.compile(r'[^@\s<>]+\s+[\[\/]?at[\]\/]?\s+[a-zA-Z0-9\-]+\s+[\[\/]?dot[\]\/]?\s+(com|net|org|gov)')

output_lists = []

for i, text in enumerate(rows):
    result = pattern.search(text)
    if result:
        result = re.sub(r"\s*@+\s*", "@", result.group())
        result = re.sub(r"\s*\.+\s*", ".", result)
        output_lists.append((i, result[:-1]))
    else:
        result_dot = pattern_dot.search(text)
        if result_dot:
            result = re.sub(r"\s+[\[\/]?at[\]\/]?\s+", "@", result_dot.group())
            result = re.sub(r"\s+[\[\/]?dot[\]\/]?\s+", ".", result)
            output_lists.append((i, result))
        else:
            output_lists.append((i, "None"))

with open(output_file, 'w', newline='') as wfile:
    writer = csv.writer(wfile)
    writer.writerow(["Id", "Category"])
    for l in output_lists:
        writer.writerow(l)



    

