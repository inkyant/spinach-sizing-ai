
import requests
from datetime import datetime, timedelta

URL = "https://dli.suntrackertech.com:8443/DLI/api/get_DLI/"
longitude = "38"
latitude = "21"
r = requests.get(url=(URL + longitude + "," + latitude))
data = r.json()

def growthfunc(size, month):
    current_month = month
    next_month = (datetime.now() + timedelta(days=30)).strftime('%B'),
    current_month_dli, next_month_dli = None, None
    for entry in data:
        if entry['month'] == current_month:
            current_month_dli = entry['dli_val']
        elif entry['month'] == next_month:
            next_month_dli = entry['dli_val']
    return(size*(1+(0.0046)*current_month_dli))