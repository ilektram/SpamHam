import joblib
from os import path, listdir
from typing import List
from aiohttp import web
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


from config import OUT_PATH, TRAINED_MODEL
from utils import LOG, get_feature_df, get_tfidf

routes = web.RouteTableDef()
scaler = MinMaxScaler(feature_range=(-1, 1)).fit(array([1, 0, .5]).reshape(-1, 1))
col_names = pd.read_csv('test.csv', index_col=0, nrows=0).columns.tolist()


def get_prediction(x):
    pred_proba = model.predict_proba(x)[:, 1]
    score = pred_proba.reshape(-1, 1)
    return score


def load_model(model_dir: str, model_filename: str):
    LOG.info("Loading...")
    if model_filename in listdir(model_dir):
        with open(path.join(model_dir, model_filename), 'rb') as f:
            model = joblib.load(f)
        LOG.info(f"Successfully loaded model {model_filename} from {model_dir}!")
        return model
    else:
        LOG.error(f"Trained model {model_filename} is not in expected {model_dir} directory! Please retrain the model.")


def load_tfidf_vect(dir: str):
    LOG.info("Loading vectorizer...")
    filename = "tfidf_vect.joblib"
    if filename in listdir(dir):
        with open(path.join(dir, filename), 'rb') as f:
            vect = joblib.load(f)
        LOG.info(f"Successfully loaded vectorizer {filename} from {dir}!")
        return vect
    else:
        LOG.error(f"Fitted vectorizer {filename} is not in expected {dir} directory! Please regenerate.")


def get_features_from_text(s: str, tfidf_vect: TfidfVectorizer, col_names: List[str]=col_names):
    text_df = pd.DataFrame({"Text": [s]}, columns=['Text']+col_names)
    new_df = get_feature_df(text_df)
    tfidf_df = get_tfidf(new_df, tfidf_vect, fit=False)
    x = tfidf_df.drop(['Text', 'feature_list', 'lemmas', 'n_pos'], axis=1)
    x.fillna(0, inplace=True)
    return x.iloc[0]


def parse_input(json_text):
    try:
        text_body = json_text["Text"]
        if type(text_body) != str:
            LOG.warning(f"Wrong user input. User posted us the following input {text_body} of type: {type(text_body)}")
            return ValueError
        return str(text_body).strip()
    except KeyError:
        LOG.warning(f"Wrong user input. User posted us the following: {json_text}")
        return KeyError


@routes.get('/healthcheck')
async def healthcheck(request):
    return web.Response(text="Thumbs up!")


@routes.post('/spam')
async def predict_sentiment(request):
    payload = await request.json()
    x = parse_input(payload)
    if not x:
        return web.Response(text="No or empty input received. Please post your text body as json of the form {'Text': text body string}.", status=400)
    elif x is KeyError:
        return web.Response(text="Wrong input. Please post your text body as json of the form {'Text': text body string}.", status=400)
    elif x is ValueError:
        return web.Response(text="Wrong input type. Please post your text body as json of the form {'Text': text body string}.", status=400)
    else:
        try:
            ft = get_features_from_text(x, tfidf_v)
            predicted_spam = get_prediction([ft])
            return web.json_response({"spam probability": round(predicted_spam[0][0], 3)})
        except Exception as e:
            LOG.error(f"Application errored: {e.__repr__()}")
            return web.Response(text="Something has gone very wrong indeed...", status=500)


if __name__ == "__main__":
    tfidf_v = load_tfidf_vect(OUT_PATH)
    model = load_model(OUT_PATH, TRAINED_MODEL)
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
