import keras

from flask import Flask, request
import json
import numpy as np

app = Flask(__name__)

model = keras.models.load_model("model0001_2hl18n_5.h5")


@app.route("/predict", methods=["POST"])
def predict():
    event = json.loads(request.data)
    values = event["datas"]
    values = list(map(float, values))
    pre = np.array(values)
    pre = pre.reshape(1, -1)
    res = model.predict(pre)
    print("ini hasil prediksi lewat API", res)
    return str(res[0])


@app.route("/predict-taksasi", methods=["POST"])
def predict_taksasi():
    event = json.loads(request.data)
    print("ini event", event)
    values = event["datas"]

    if np.ndim(values) > 1:
        values = np.array([list(map(float, sublist)) for sublist in values])
        pre = values
    else:
        values = np.array(list(map(float, values)))
        pre = values.reshape(1, -1)

    # print("ini values", values)
    print("cek", pre)

    result_prediction = []

    for data in pre:
        pre = (data.reshape(1, -1))
        for i in range(len(pre)):
            for j in range(len(pre[i])):
                if j == 0:
                    pre[i][j] = (pre[i][j] - 1) / (22 - 1)
                elif j == 1:
                    pre[i][j] = (pre[i][j] - 2001) / (2004 - 2001)
                elif j == 2:
                    pre[i][j] = (pre[i][j] - 3.8) / (39.2 - 3.8)
                elif j == 3:
                    pre[i][j] = (pre[i][j] - 444) / (5761 - 444)
                elif j == 4:
                    pre[i][j] = (pre[i][j] - 56) / (274 - 56)
                elif j == 5:
                    pre[i][j] = (pre[i][j] - 4) / (56 - 4)
                elif j == 6:
                    pre[i][j] = (pre[i][j] - 3) / (20 - 3)
                elif j == 7:
                    pre[i][j] = (pre[i][j] - 20) / (22 - 20)
                elif j == 8:
                    pre[i][j] = (pre[i][j] - 56) / (860 - 56)
        #     bahan_uji[i][j] = (bahan_uji[i][j] - min_value[j]) / (max_value[j] - min_value[j])
        result = model.predict(pre)
        result_prediction.append(result.item())

    # print("ini bahan uji", pre[0])

    # result = model.predict(pre)
    # print("ini hasil prediksi lewat API", result)
    return result_prediction


if __name__ == "__main__":
    app.run(debug=True)
