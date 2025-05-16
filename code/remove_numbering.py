import re

with open('your_data.txt', 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

with open('your_data.txt', 'w', encoding='utf-8') as outfile:
    for line in lines:
        # Remove leading numbers like "1.", "2.", etc. (with optional space after the dot)
        cleaned_line = re.sub(r'^\s*\d+\.\s*', '', line)
        outfile.write(cleaned_line)
