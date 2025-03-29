from src.data_loader import DataLoader
from src.model import ProjectModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging

# Настройка логирования
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    data_stream = DataLoader()
    # Инициализация тренера
    p_model = ProjectModel(LinearRegression())

    # Обучение модели на порционных данных
    for i in range(10):
        X, y = data_stream.get_data(batch_size=100)  # Загружаем порцию данных
        if X is None or y is None:
            break  # Если данных больше нет, выходим из цикла
        if i > 0:
            logging.info("Testing on new data")
            y_pred = p_model.predict(X)
            # print(y, y_pred)
            logging.info(f"R2: {r2_score(y, y_pred)}")
            
            # print("Coefs")
            # print(data_stream.get_weights())
            # print(p_model.model.coef_)
        p_model.fit(X, y)  # Обучаем модель на текущей порции данных
        logging.info("Added new data")
        logging.info(f"Quality on full data: {p_model.score()}")

if __name__ == "__main__":
    main()