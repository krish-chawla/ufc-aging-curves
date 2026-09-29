import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

url = "http://ufcstats.com/event-details/638cfec7ec559d6e"
resp = requests.get(url, headers=HEADERS, timeout=20)
print("STATUS:", resp.status_code)
print("LENGTH:", len(resp.text))

soup = BeautifulSoup(resp.text, "lxml")
rows = soup.select("tr.b-fight-details__table-row")
print("ROWS FOUND (my selector):", len(rows))

# broader fallback: any table row at all
all_rows = soup.select("table tr")
print("ANY TABLE ROWS:", len(all_rows))
if all_rows:
    print("\n--- first row's raw HTML ---")
    print(all_rows[1].prettify()[:2000] if len(all_rows) > 1 else all_rows[0].prettify()[:2000])
print("\n--- RAW RESPONSE ---")
print(resp.text)
