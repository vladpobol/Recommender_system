# Рекомендательная система

* #### __Решаемая задача__ - Обычно у социальной сети есть пользователи и какой-либо контент, который мы хотим им рекоммендовать максимально качественно, так чтобы интересующие нас метрики росли, именно эту цель и преследует мой проект.
* #### __Используемый стек__ - `Python` `FastAPI` `Psycopg2` `SQLAlchemy`, `Pandas` `Numpy` `Sklearn` `CatBoost` `Transformers` `PyTorch`


\
## Подключение к базе данных
![.](https://www.google.com/url?sa=i&url=https%3A%2F%2Fproglib.io%2Fp%2F11-tipov-sovremennyh-baz-dannyh-kratkie-opisaniya-shemy-i-primery-bd-2020-01-07&psig=AOvVaw26Khyz0pT3JXtaeiPae7T0&ust=1687385320859000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCOj3yqju0v8CFQAAAAAdAAAAABAE)
\
За подключение к базе отвечает файл [connect_database](https://github.com/vladpobol/Recommender_system/blob/master/connect_database.py "посмотреть код")


* С целью практики в этом файле реализовано два способа подключения к базе данных

\
## Предобработка данных
![.](https://www.google.com/url?sa=i&url=https%3A%2F%2Floginom.ru%2Fblog%2Fdata-cleansing&psig=AOvVaw1wqyK526-_kcvqmpek8OAo&ust=1687385670680000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCLjJtM_v0v8CFQAAAAAdAAAAABAE)
\
За предобработкуу данных и выгрузку их в базу отвечает [preprocessing_data](https://github.com/vladpobol/Recommender_system/blob/master/preprocessing_data.py "посмотреть код")







