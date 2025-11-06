import requests

def fetch_root():
    response = requests.get('http://example.com/')
    print(response.text)

if __name__ == '__main__':
    fetch_root()