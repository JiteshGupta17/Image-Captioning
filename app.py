from flask import Flask,render_template,redirect,request
import Caption_it
from gtts import gTTS
import os

# __name__ == __main__
app = Flask(__name__)


## for GET request
@app.route('/')
def hello():
	return render_template("index.html")


## for POST request
@app.route('/',methods = ['POST'])
def marks():

	if request.method == 'POST':

		f = request.files['userfile']
		path = "./static/{}".format(f.filename) 
		f.save(path)

		path2 = "./static/{}".format(f.filename) + ".mp3"

		caption = Caption_it.caption_this_image(path)
		output = gTTS(text = caption, lang = 'en',slow = False)
		output.save(path2)

		result_dic = {
		'image' : path,
		'caption' : caption,
		'sound' : path2
		}

	return render_template("index.html",your_result = result_dic)

if __name__ == '__main__':
	app.run(debug = True)
