# Ընտրում ենք Python-ի թեթև աշխատանքային տարբերակը
FROM python:3.10-slim

# Սահմանում ենք աշխատանքային պանակը կոնտեյների ներսում
WORKDIR /app

# Նախապես տեղադրում ենք համակարգային անհրաժեշտ գրադարաններ, եթե պահանջվեն
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Տեղափոխում ենք պահանջվող ֆայլերը կոնտեյներ
# (Համոզվիր, որ requirements.txt ունես, կամ ուղղակի տեղափոխում ենք main.py-ն ու մոդելը)
COPY requirements.txt .

# Տեղադրում ենք Python գրադարանները
RUN pip install --no-cache-dir -r requirements.txt

# Տեղափոխում ենք մնացած ֆայլերը (main.py, ensemble_package.pkl)
COPY . .

# Բացում ենք պորտը
EXPOSE 8000

# Հրաման սերվերը գործարկելու համար
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]