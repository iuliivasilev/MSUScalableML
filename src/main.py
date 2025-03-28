from data.data_generator import generate_data
from data.data_loader import get_data
from model.trainer import Trainer

def main():
    # Инициализация тренера
    trainer = Trainer()
    
    # Генерация начальных данных
    generate_data(num_samples=10000)  # Генерируем 10,000 образцов данных

    # Обучение модели на порционных данных
    while True:
        X, y = get_data(batch_size=100)  # Загружаем порцию данных
        if X is None or y is None:
            break  # Если данных больше нет, выходим из цикла
        trainer.train(X, y)  # Обучаем модель на текущей порции данных

if __name__ == "__main__":
    main()