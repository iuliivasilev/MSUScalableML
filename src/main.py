from data.data_loader import DataLoader
from model.trainer import Trainer


def main():
    data_stream = DataLoader()
    # Инициализация тренера
    trainer = Trainer()

    # Обучение модели на порционных данных
    while True:
        X, y = data_stream.get_data(batch_size=1000)  # Загружаем порцию данных
        if X is None or y is None:
            break  # Если данных больше нет, выходим из цикла
        trainer.fit(X, y)  # Обучаем модель на текущей порции данных


if __name__ == "__main__":
    main()