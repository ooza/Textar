from flask import Flask, request, Response , render_template
import importlib
import os # this will be your file name; minus the `.py`

app = Flask(__name__)

@app.route('/')
def dynamic_page():
    os.system('python ./run.py')
    return "Done"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
