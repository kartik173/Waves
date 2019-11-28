from flask import Flask, request,render_template
from prediction import Predictions
#import random

app = Flask(__name__)
mapping={0:1,1:10,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        import sounddevice as sd
        from scipy.io.wavfile import write
        
        fs = 22050  # Sample rate
        seconds = 5  # Duration of recording
        
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('test/temp/output.wav', fs, myrecording)
        res=Predictions().testAudio()
        
#        label=res.keys()
#        prob=res.values()
#        print(label,prob)
#        [random.choice([x for x in range(48,78)]) for j in range(3)]
#        
#        res.append(random.choice([x for x in range(9) if x not in res]))
#        res.append(random.choice([x for x in range(9) if x not in res]))
        return render_template('result.html',res=res)

if __name__ == '__main__':
    app.run(debug=True)
