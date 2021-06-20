from flask import Flask, request, render_template
import model

app = Flask(__name__, template_folder="templates")


@app.route('/home')
def home():
    return render_template('index.html')  # Render home.html


@app.route('/result', methods=['POST', 'GET'])
def classify_type():
    #rf = model.read_data()
    try:
        age = request.args.get('age')
        ejection_fraction = request.args.get('ejection_fraction')
        serum_creatinine = request.args.get('serum_creatinine')
        serum_sodium = request.args.get('serum_sodium')
        time = request.args.get('time')

        # Get the output from the classification model
        result = model.model_predict(
            age, ejection_fraction, serum_creatinine, serum_sodium, time)

        # Render the output in new HTML page
        return render_template('output.html', result=result)
    except:
        return 'Error'


if(__name__ == '__main__'):
    app.run(debug=True)
