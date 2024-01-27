
from api_tokens import REPLICATE_API_TOKEN, CDN_API_KEY, CDN_CLOUD_NAME, CDN_API_SECRET

import replicate
import cloudinary.uploader
import random

import os, cv2
import matplotlib.pyplot as plt

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN


cloudinary.config( 
  cloud_name = CDN_CLOUD_NAME, 
  api_key = CDN_API_KEY, 
  api_secret = CDN_API_SECRET
)
          
def segment_image(image_file: str):


    # # read image
    # img = cv2.imread(image_file)
    # print('Image Width is',img.shape[1])
    # print('Image Height is',img.shape[0])

    # img_downscaled = cv2.resize(img, None, fx = 0.15, fy = 0.15)

    # plt.imshow(img_downscaled)
    # plt.show()

    # cv2.imwrite('downscaled_' + image_file, img_downscaled)

    # # upload to CDN
    # public_id = str(random.randint(0,999)) + image_file
    # cloudinary.uploader.upload("./downscaled_" + image_file, public_id = public_id)

    # print('finished uploading')

    # AI Analysis
    output = replicate.run(
        "yyjim/segment-anything-everything:b28e02c3844df2c44dcb2cb96ba2496435681bf88878e3bd0ab6b401a971d79e",
        input={
            # "image": "https://res.cloudinary.com/" + CDN_CLOUD_NAME + "/image/upload/f_auto,q_auto/" + public_id,
            "image": "https://cdn.discordapp.com/attachments/461005989609603082/1200882583303626862/downscaled_IMG_0494.JPG?ex=65c7cc0f&is=65b5570f&hm=5dd8306e142435c6bca7fe9a62138f81eae41faf9397ba92e26041d76751e28a&",
            "mask_limit": 10,
        }
    )

    print(output)

    # cloudinary.uploader.destroy(public_id)
    

if __name__ == '__main__':
    segment_image('IMG_0494.JPG')


# input
# "https://cdn.discordapp.com/attachments/461005989609603082/1200882583303626862/downscaled_IMG_0494.JPG?ex=65c7cc0f&is=65b5570f&hm=5dd8306e142435c6bca7fe9a62138f81eae41faf9397ba92e26041d76751e28a&"
    
# result 
# ['https://replicate.delivery/pbxt/ZOhycP1ujE7GH5KYTYm3unYDLxiANuDcc6gNgmQBfEFCVYIJA/mask_0.png', 'https://replicate.delivery/pbxt/jctTsHsIyfRFRiJShF6KN4hBJ7Ousg79D2FhnH8X5sBCVYIJA/mask_1.png', 'https://replicate.delivery/pbxt/gaV8T1Wt6xLjFdauMTr9XPIJJIEZTXCPV2iLM2byG3WhKMkE/mask_2.png', 'https://replicate.delivery/pbxt/XHv70fLnMwXwO6ylNq7PEkDKSBWlubwPz9QOSr1tA0uCVYIJA/mask_3.png', 'https://replicate.delivery/pbxt/3K7TaI7C8NbzGd8gke3D55DcNZaI38BdeawwqaVt6UOFqwQSA/mask_4.png', 'https://replicate.delivery/pbxt/uM8cQs7ac5LmB9Xw9F9B54jXccaXmpFWrZIjo8aZCxUhKMkE/mask_5.png', 'https://replicate.delivery/pbxt/aAafvaLpaFTONatf7dtuPk3WADY3IdeiFDmph8KX8ioKUhhkA/mask_6.png', 'https://replicate.delivery/pbxt/B6d0qEdY9T5lN5nBfrye9dfYBT8OJeph4CskC85jQ0vXoCDJB/mask_7.png', 'https://replicate.delivery/pbxt/NbcdyXBfXSy3UyRZZ2KFLpGckx8Qtf2dFsi8ActYtynFqwQSA/mask_8.png', 'https://replicate.delivery/pbxt/M1GwLicuSBJoFBE8VhGIUUL7OafSI9gEF9cCLNU9mBADVYIJA/mask_9.png', 'https://replicate.delivery/pbxt/tUe3oIiySUTAYyDCAnntYwD26RvvmwHLS9Ln45Gbw4ODVYIJA/masked_0.png', 'https://replicate.delivery/pbxt/sQ14KcDBbMq0PRsUStU4U3wdokWriGIQMan0z5O6h8shKMkE/masked_1.png', 'https://replicate.delivery/pbxt/OnLPKahN9sZJMRRkVCGge0LEvmCQUYsCatkdIKjdEotDVYIJA/masked_2.png', 'https://replicate.delivery/pbxt/U7kW37pEQ5ocDFxeTMU4UgsXG5S445QglzycBP3ooiKEVYIJA/masked_3.png', 'https://replicate.delivery/pbxt/lQK5Jmzp1yrTM1kFjhqcccBgbqgcaWFyOffeO9FDo0GQUhhkA/masked_4.png', 'https://replicate.delivery/pbxt/powW27sKi5Z3PluYNmCgvOMyDthKLetT7cZHe2eZvbNTUhhkA/masked_5.png', 'https://replicate.delivery/pbxt/fLJVtPhqIRSQCCUULXytwmfvwKlq9kVvfSJ9Pvk3UidVUhhkA/masked_6.png', 'https://replicate.delivery/pbxt/Mqstlcfvs13vHi9YMNvgNomLq6AL7QUPeFgY3RBBx4eUUhhkA/masked_7.png', 'https://replicate.delivery/pbxt/yALyBJ00pD4INxKq49QAqegz2DcA1Iq6GOxs0698qx0FVYIJA/masked_8.png', 'https://replicate.delivery/pbxt/rw3hTqFYVM79H5hAtAb7J8fXHfeThL91hwRq4q1iAxfxoCDJB/masked_9.png']