from time import sleep
from src.data_loader import DataLoader
from src.model import ProjectModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
import argparse
import pickle
import os

# Настройка логирования
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

SAVE_DIR = "loc"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_object(obj, filename):
    """Сохраняет объект в файл через pickle."""
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"Object saved to {filepath}")


def load_object(filename):
    """Загружает объект из файла через pickle."""
    filepath = os.path.join(SAVE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            logging.info(f"Object loaded from {filepath}")
            return pickle.load(f)
    return None


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
    data_stream = load_object("DataLoader.pkl")
    p_model = load_object("Model.pkl")
    if data_stream is None:
        data_stream = DataLoader()
    if p_model is None:
        p_model = ProjectModel(LinearRegression())
    
    step(data_stream, p_model)
    save_object(data_stream, "DataLoader.pkl")
    save_object(p_model, "Model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=['single', 'all'],
                        help="Action type", default='all')
    args = parser.parse_args()

    if args.mode == 'all':
        all()
    else:
        stepwise()