
from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)

with open("xgb_model.pickle",'rb') as f:
    xgb_model = pickle.load(f)
f.close()

with open("encode_shipment.pickle",'rb') as f:
    encode_shipment = pickle.load(f)
f.close()

with open("le_country.pickle",'rb') as f:
    le_country = pickle.load(f)
f.close()

with open("minmax.pickle",'rb') as f:
    minmax = pickle.load(f)
f.close()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    print(request.form)
    country = str(request.form['dest_country'])
    manu_country = str(request.form['manu_country'])
    shipment = str(request.form['shipmode'])
    unit_of_measure = int(request.form['measure'])
    line_item_quantity = int(request.form['li_qty'])
    line_item_value = float(request.form['li_value'])
    pack_price = float(request.form['pprice'])
    unit_price = float(request.form['uprice'])
    first_line_designation = int(request.form['fld'])
    weight = float(request.form['weight'])
    line_item_insurance = float(request.form['li_insu'])

    counted_encoded, manu_country_encoded = le_country.transform(np.array([country])), le_country.transform(np.array([manu_country]))
    numericals = np.array([unit_of_measure, line_item_quantity, line_item_value, pack_price, unit_price, first_line_designation, weight, line_item_insurance])
    shipment_encoded = encode_shipment.transform(np.array([shipment]).reshape(-1,1)).squeeze()
    to_scale = np.delete(numericals, 5)
    numericals_scaled = np.insert(minmax.transform(to_scale.reshape(1,-1)).squeeze(), 5, first_line_designation)
    input_x = np.hstack((counted_encoded, numericals_scaled, manu_country_encoded, shipment_encoded)).reshape(1,-1)
    predicted_class = xgb_model.predict(input_x)[0]

    return render_template ('result.html',prediction_text="The shipment cost is {}".format(predicted_class))

if __name__=='__main__':
    app.run(port=8000, debug=True)

