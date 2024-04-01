from dateutil.parser import parse
import re

def convert_to_iso_format(date_str):
    match = re.search(r'(\bJan\b|\bFeb\b|\bMar\b|\bApr\b|\bMay\b|\bJun\b|\bJul\b|\bAug\b|\bSep\b|\bOct\b|\bNov\b|\bDec\b)\s+\d{1,2},\s+\d{4}', date_str)
    if match:
        date_part = match.group()
        date = parse(date_part, fuzzy=True)
        return date.strftime('%Y-%m-%d')
    else:
        return "Invalid date format"