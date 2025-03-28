def get_data(batch_size=100):
    import numpy as np

    # Генерация случайных данных
    X = np.random.rand(batch_size, 1) * 10  # 100 случайных значений от 0 до 10
    y = 2 * X + np.random.randn(batch_size, 1)  # Линейная зависимость с добавлением шума

    return X, y