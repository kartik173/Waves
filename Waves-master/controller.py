from flask import Flask, redirect, url_for, request,render_template
from digit_recog import prediction

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
	
	if request.method == 'POST':
		img = request.form['nm']
		res=prediction.fun(img)
		return render_template('result.html',res=res)

	else:
		img = request.args.get('nm')
		res=prediction.fun(img)
		return render_template('result.html',res=res)


if __name__ == '__main__':
	app.run(debug=True)
