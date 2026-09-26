import os
import re
import urllib.request

def fetch_noaa():
    url = 'https://oceanexplorer.noaa.gov/multimedia/georeferenced-side-scan-sonar-image/'
    dest_dir = os.path.join('data', 'downloaded')
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, 'noaa_sonar_georeferenced.jpg')
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
            # Look for image tags
            imgs = re.findall(r'(https://oceanexplorer\.noaa\.gov/[^"\'>]+\.(?:jpg|png|jpeg))', html)
            if not imgs:
                # relative paths
                imgs = re.findall(r'src="([^"]+\.(?:jpg|png|jpeg))"', html)
                imgs = [('https://oceanexplorer.noaa.gov' + i if i.startswith('/') else i) for i in imgs]
            
            print(f"Found {len(imgs)} candidate image URLs in NOAA page:")
            for img_url in imgs[:5]:
                print(f" - {img_url}")
                
            for img_url in imgs:
                if 'sidescan' in img_url.lower() or 'sonar' in img_url.lower() or 'georeferenced' in img_url.lower():
                    print(f"Downloading NOAA image from {img_url}")
                    r2 = urllib.request.Request(img_url, headers=headers)
                    with urllib.request.urlopen(r2, timeout=15) as resp2:
                        content = resp2.read()
                        dest = os.path.join(dest_dir, 'noaa_fig2_sidescan.png')
                        with open(dest, 'wb') as f:
                            f.write(content)
                    print(f"Successfully saved NOAA image: {dest} ({len(content)} bytes)")
                    return True
    except Exception as e:
        print(f"Error fetching NOAA image: {e}")
    return False

if __name__ == '__main__':
    fetch_noaa()
