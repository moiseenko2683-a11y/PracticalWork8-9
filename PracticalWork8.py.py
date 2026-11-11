
import pandas as pd
from pathlib import Path

# Шлях до CSV-файлу (на рівень вище папки програми)
csv_path = Path(__file__).resolve().parent.parent / "exports_data.csv"

# Завантаження даних
df = pd.read_csv(csv_path, sep=None, engine="python")
# Очищення назв стовпців
df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print("\nНазви стовпців:")
print(df.columns.tolist())

# Перевірка пропущених значень
print("\nКількість пропущених значень:")
print(df.isnull().sum())

# Перевірка дублікатів
print("\nКількість дублікатів:")
print(df.duplicated().sum())

# Видалення дублікатів
df = df.drop_duplicates()

# Фільтрація рядків із наявними значеннями експорту
df = df.dropna(subset=["value"])

print("\nДані після фільтрації:")
print(df.shape)

# Сортування даних за країною та роком
df_sorted = df.sort_values(by=["country", "year"])

# Сортування даних за країною та роком
df_sorted = df.sort_values(by=["country", "year"])

print("\nВідсортовані дані:")
print(df_sorted.head(10))

# Групування даних за роками
year_stats = df.groupby("year")["value"].agg(
    ["count", "mean", "min", "max"]
)

print("\nАналіз експорту за роками:")
print(year_stats)

# Збереження оброблених даних у новий CSV
df.to_csv("exports_data_processed.csv", index=False)

print("\nОброблені дані збережено у exports_data_processed.csv")

print("Файл успішно завантажено!")

print("\nПочаткові дані:")
print(df)

print("\nПерші 5 рядків:")
print(df.head())

print("\nІнформація про DataFrame:")
df.info()

print("\nРозмір DataFrame:")
print(df.shape)

print("\nОписова статистика:")
print(df.describe())

# ==========================================
# ЗАВДАННЯ 2. АНАЛІЗ ВЕЛОСИПЕДНИХ ДАНИХ
# ==========================================

# Завантаження даних за 2017 рік
bike_path = Path(__file__).resolve().parent.parent / "comptagevelo2017.csv"

bike_df = pd.read_csv(bike_path, sep=None, engine="python")

print("\n========== ВЕЛОСИПЕДНІ ДАНІ ==========")

print("\nПерші 5 рядків:")
print(bike_df.head())

print("\nРозмір таблиці:")
print(bike_df.shape)

print("\nНазви стовпців:")
print(bike_df.columns.tolist())

print("\nІнформація про дані:")
bike_df.info()

print("\nОписова статистика:")
print(bike_df.describe())

# ==========================================
# АНАЛІЗ ВЕЛОСИПЕДНОГО РУХУ
# ==========================================

import matplotlib.pyplot as plt

# Перетворення дати
bike_df["Date"] = pd.to_datetime(
    bike_df["Date"],
    format="%d/%m/%Y"
)

# Визначення стовпців із велосипедними лічильниками
station_columns = bike_df.columns[2:]

# Перетворення значень лічильників на числові
bike_df[station_columns] = bike_df[station_columns].apply(
    pd.to_numeric, errors="coerce"
)

# Загальна кількість велосипедистів на всіх маршрутах
total_cyclists = bike_df[station_columns].sum().sum()

print("\nЗагальна кількість велосипедистів:")
print(int(total_cyclists))

# Додавання номера місяця
bike_df["Month"] = bike_df["Date"].dt.month

# Вибір трьох велосипедних маршрутів
selected_paths = ["Berri1", "Boyer", "Brébeuf"]

# Помісячна кількість велосипедистів
monthly_paths = bike_df.groupby("Month")[selected_paths].sum()

print("\nКількість велосипедистів за місяцями:")
print(monthly_paths)

# Найпопулярніший місяць для кожного маршруту
print("\nНайпопулярніший місяць для маршрутів:")

for path in selected_paths:
    popular_month = monthly_paths[path].idxmax()
    popular_value = monthly_paths[path].max()

    print(
        f"{path}: місяць {popular_month}, "
        f"кількість велосипедистів: {int(popular_value)}"
    )

# Помісячна динаміка маршруту Berri1
berri_monthly = monthly_paths["Berri1"]

print("\nПомісячна динаміка маршруту Berri1:")
print(berri_monthly)

# Побудова графіка
plt.figure(figsize=(10, 5))
plt.plot(
    berri_monthly.index,
    berri_monthly.values,
    marker="o"
)

plt.title("Помісячна кількість велосипедистів — Berri1 (2017)")
plt.xlabel("Місяць")
plt.ylabel("Кількість велосипедистів")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig("berri1_monthly.png")
plt.show()
import nltk
from nltk.corpus import gutenberg

# Завантаження текстового корпусу Gutenberg
nltk.download('gutenberg')

# Завантаження тексту відповідно до варіанта 6
words = gutenberg.words('bryant-stories.txt')

# Загальна кількість слів
total_words = len(words)

# Кількість унікальних слів
unique_words = len(set(words))

print("\n========== АНАЛІЗ ТЕКСТУ NLTK ==========")
print("Назва тексту: bryant-stories.txt")
print("Загальна кількість слів:", total_words)
print("Кількість унікальних слів:", unique_words)
# Топ-10 найчастіше вживаних слів
from collections import Counter

word_freq = Counter(words)
top_10 = word_freq.most_common(10)

print("\nТоп-10 найчастіше вживаних слів:")
for word, count in top_10:
    print(word, "-", count)

# Побудова стовпчикової діаграми
import matplotlib.pyplot as plt

words_top = [item[0] for item in top_10]
counts_top = [item[1] for item in top_10]

plt.figure(figsize=(10, 5))
plt.bar(words_top, counts_top)
plt.title("Топ-10 найчастіше вживаних слів")
plt.xlabel("Слова")
plt.ylabel("Кількість повторень")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_10_words.png")
plt.show()
import string
import re
from nltk.corpus import stopwords

# Завантаження стоп-слів
nltk.download('stopwords')

# Список англійських стоп-слів
stop_words = set(stopwords.words('english'))

# Видалення пунктуації, стоп-слів та інших зайвих символів
filtered_words = [
    word.lower()
    for word in words
    if word.isalpha() and word.lower() not in stop_words
]

# Топ-10 слів після очищення
filtered_top_10 = Counter(filtered_words).most_common(10)

print("\nТоп-10 слів після видалення стоп-слів і пунктуації:")
for word, count in filtered_top_10:
    print(word, "-", count)

# Побудова діаграми
filtered_labels = [item[0] for item in filtered_top_10]
filtered_counts = [item[1] for item in filtered_top_10]

plt.figure(figsize=(10, 5))
plt.bar(filtered_labels, filtered_counts)
plt.title("Топ-10 слів після очищення тексту")
plt.xlabel("Слова")
plt.ylabel("Кількість повторень")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top_10_clean_words.png")
plt.show()
from nltk.stem import PorterStemmer

# Створення об'єкта стемера
stemmer = PorterStemmer()

# Стемінг очищених слів
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Визначення топ-10 після стемінгу
stemmed_top_10 = Counter(stemmed_words).most_common(10)

print("\nТоп-10 слів після стемінгу:")
for word, count in stemmed_top_10:
    print(word, "-", count)

print("\nПриклади стемінгу:")
for word in filtered_words[:15]:
    print(word, "->", stemmer.stem(word))