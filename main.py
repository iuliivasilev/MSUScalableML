from time import sleep
from src.data_loader import DataLoader
from src.model import ProjectModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_object, save_object
import logging
import argparse
import os


# Настройка логирования
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

SAVE_DIR = "loc"
os.makedirs(SAVE_DIR, exist_ok=True)


def step(data_stream, model):
    X, y = data_stream.get_data(batch_size=100)  # Загружаем порцию данных
    if X is None or y is None:
        return  # Если данных больше нет, выходим
    if model.is_fit():
        logging.info("Testing on new data")
        y_pred = model.predict(X)
        # print(y, y_pred)
        logging.info(f"R2: {r2_score(y, y_pred)}")
        
        # print("Coefs")
        # print(data_stream.get_weights())
        # print(p_model.model.coef_)
    model.fit(X, y)  # Обучаем модель на текущей порции данных
    logging.info(f"Iter: {model.get_iter()}")
    logging.info("Added new data")
    logging.info(f"Quality on full data: {model.score()}")


def all(n_iter=10):
    data_stream = DataLoader()
    p_model = ProjectModel(LinearRegression())

    # Обучение модели на порционных данных
    for _ in range(n_iter):
        step(data_stream, p_model)
        sleep(5)  # Задержка для имитации реального времени


def stepwise():
    dl_path = os.path.join(SAVE_DIR, "DataLoader.pkl")
    mdl_path = os.path.join(SAVE_DIR, "Model.pkl")
    data_stream = load_object(dl_path)
    p_model = load_object(mdl_path)
    if data_stream is None:
        data_stream = DataLoader()
    if p_model is None:
        p_model = ProjectModel(LinearRegression())
    
    step(data_stream, p_model)
    save_object(data_stream, dl_path)
    save_object(p_model, mdl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=['single', 'all'],
                        help="Action type", default='all')
    args = parser.parse_args()

    if args.mode == 'all':
        all()
    else:
        stepwise()