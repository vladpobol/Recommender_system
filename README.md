# Рекомендательная система

<img src="https://github.com/vladpobol/Recommender_system/blob/master/plots_and_diagrams/diag_1.drawio.png" alt="Header">

* #### __Решаемая задача__ - Рекомендация 5 постов для юзера по запросу, .
* #### __Используемый стек__ - `Python` `FastAPI` `Psycopg2` `SQLAlchemy`, `Pandas` `Numpy` `Sklearn` `CatBoost` `Transformers` `PyTorch`

## [`Подключение к базе и выгрузка данных`](https://github.com/vladpobol/Recommender_system/blob/master/connect_database.py "посмотреть код")

* ❗ Перед использованием кода необходимо указать URI вашей базы данных в переменной DATABASE_URI.
* С целью практики в этом файле реализовано два способа подключения к базе данных: через psycopg и sqlalchemy.
* `get_data_with_psycopg` используется для выгрузки уже предобработанных данных.
* `get_data_with_sqlalchemy` используется для выгрузки исходных данных с последующей предобработкой. 

## [`Предобработка данных`](https://github.com/vladpobol/Recommender_system/blob/master/preprocessing_data.py "посмотреть код")

1. Подготовка данных про пользователей:
    - Все признаки, в которых менее 15 уникальных значений будем кодировать через `OneHotEnсoding`.
    - Те признаки, в которых больше через `LabelEncoder`.
      
2. Подготовка текстов:
    - Удаление лишних символов (пробелов, знаков препинания).
    - Удаление стоп-слов, с помощью библиотеки `nltk` импортируем список неважных слов для английского языка и убираем их.
    - Лемматизация, используем `WordNetLemmatizer` из библиотеки `nltk` для приведения всех слов к общей форме.
      
3. Достаем признаки из TF-IDF:
    - Формируем большую матрицу с TF-IDF коэффициентами для каждого слова по всем текстам, в итоговой таблице более чем 60 тысяч признаков.
    - Вычитаем из них среднее и уменьшаем их размерность с помощью метода главных компонент `PCA`, на выходе имеем 50 признаков.
    - Кластеризуем их с помощью `KMeans`, получим разбиение на 15 кластеров.
    - Считаем расстояние до каждого кластера.
      
4. Извлекаем признаки из эмбеддингов:
    - Загружаем эмбеддинги (процесс их извлечения рассмотрен в следующей главе).
    - Вычитаем среднее и уменьшаем размерность до 50 колонок.
    - Выделяем 15 кластеров через `KMeans` и считаем расстояния до кластеров.
   
## [`Получение эмбедингов постов с помощью BERT`](https://github.com/vladpobol/Recommender_system/blob/master/get_embeddings_with_BERT.py "посмотреть код")

* Подготавливаем тексты.
* Будем использовать предобученую модель и токенайзер.
* Токенизация слов будем происходить на стадии создания датасета.
* Проходимся по текстам и получаем эмбединги.
* Загружаем их в папку "BERT_embeddings".

## [`Обучение моделей`](https://github.com/vladpobol/notebooks/blob/main/Recommender_system/train_models.ipynb "посмотреть ноутбук")

1. **Первая модель**
   
   <img src="https://github.com/vladpobol/Recommender_system/blob/master/plots_and_diagrams/control_model.jpeg" width="500">
   
      - для обучения использованы только признаки из TF-IDF.
      - кластеризация с помощью KMeans.
  
3. **Вторая модель** - использованы TF-IDF и фичи из ембеддингов.
      
   <img src="https://github.com/vladpobol/Recommender_system/blob/master/plots_and_diagrams/test model.jpeg" width="500">
   
      - Использованы признаки из ембеддингов и TF-IDF.
      - Кластеризация с помощью DBSCAN и KMeans.
        
* Данные про пользователей одинаковы.

## [`Анализ результатов A/B тестирования`](https://github.com/vladpobol/notebooks/blob/main/Recommender_system/analysis_of_ab_test_results.ipynb "посмотреть ноутбук")

* Необходимо было понять значимо ли различаются модели по целевой метрике.
* Hitrate лучшей модели на тестовых данных составил 0.6.
* Модели сравнивались по двум метрикам
     - Hitrate

       <img src="https://github.com/vladpobol/Recommender_system/blob/master/plots_and_diagrams/hitrate_models.jpeg">

     - Среднее количество лайков на пользователя
 
       <img src="https://github.com/vladpobol/Recommender_system/blob/master/plots_and_diagrams/mean_count_likes.jpeg">
  
* Были затетектированы статистически значемые разлиичия.
  







