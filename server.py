
## THIS FILE IS FOR THE SERVER DO NOT RUN LOCALLY

from zipfile import ZipFile
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/", methods=['POST'])
def index():
    
    file = request.files['image']

    if not file:
        return {}
    
    file.save('input_image.png')


    # TODO: image processing code here!
    return_values = [1, 2, 3, 4, 5, 6, 7]


    return jsonify(return_values)

    # # writing files to a zipfile 
    # with ZipFile('files.zip','w') as zip: 

    #     for file in file_paths: 
    #         zip.write(file) 
    
    # return open('files.zip', 'rb')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3003)