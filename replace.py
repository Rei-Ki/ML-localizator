

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('./image-data/labels-map.csv', header=None)
    print(df.head(3))

    def replace_path(path):
        return path.replace('R:\\Projects\\hangul\\localizator\\', 'V:\\Projects\\')
    
    # Применяем функцию ко всем элементам DataFrame
    df = df.applymap(replace_path)

    # Сохраняем результат в новый CSV-файл
    df.to_csv('labels-map.csv', index=False, header=False)

