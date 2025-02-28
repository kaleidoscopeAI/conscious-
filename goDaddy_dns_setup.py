import requests
import logging
import os

GODADDY_API_KEY = os.getenv("GODADDY_API_KEY")
GODADDY_API_SECRET = os.getenv("GODADDY_API_SECRET")
DOMAIN = "artificialthinker.com"

logging.basicConfig(level=logging.INFO)

def update_dns(server_ip):
    headers = {"Authorization": f"sso-key {GODADDY_API_KEY}:{GODADDY_API_SECRET}"}
    dns_data = [{"data": server_ip, "ttl": 600}]
    url = f"https://api.godaddy.com/v1/domains/{DOMAIN}/records/A"
    
    try:
        response = requests.put(url, json=dns_data, headers=headers)
        response.raise_for_status()
        logging.info(f"DNS updated successfully: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to update DNS: {e}")

if __name__ == "__main__":
    # Get Load Balancer DNS name from AWS (replace with actual command)
    load_balancer_dns = "your-load-balancer-dns-name.amazonaws.com" 
    update_dns(load_balancer_dns)

