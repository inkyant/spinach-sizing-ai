
from zipfile import ZipFile
import requests, json
from io import BytesIO


def segment_plant_image(file_path):

    r = requests.post('http://34.69.61.109:3003/', files={'image': open(file_path, 'rb')})

    try:    
        return json.loads(r.content.decode())
    except json.decoder.JSONDecodeError:
        print("error decoding json. This is likely a server error.")
        return []

    # write_byte = BytesIO(r.content)
    
    # with open("files.zip", "wb") as f:
    #     f.write(write_byte.getbuffer())

    # with ZipFile('files.zip', 'r') as zip: 
    #     zip.extractall('./masks')

    # os.remove('files.zip')

    # print('mask file extraction complete')


if __name__ == '__main__':
    
    for a in segment_plant_image('downscaled_IMG_0494.JPG'):
        print("%.3f" % a)

