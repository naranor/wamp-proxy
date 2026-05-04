# Task Type: Needle (Universal Fact Retrieval)
DATA = [
    # --- Technical (Existing) ---
    "What is the production database port?", "Какой порт у основной базы данных?",
    "Find the API key in the configuration.", "Найди API ключ в конфигурации.",
    "What was the backup encryption password?", "Какой был пароль для шифрования бэкапа?",
    "Identify the server IP address.", "Укажи IP-адрес сервера.",
    "Retrieve the JWT expiration time.", "Найди время истечения JWT токена.",
    "What is the specific version of Kubernetes used?", "Какая конкретно версия Kubernetes используется?",
    "Find the database connection string.", "Найди строку подключения к базе данных.",
    
    # --- Art & Culture ---
    "Who painted the Starry Night?", "Кто написал картину 'Звездная ночь'?",
    "Where is the Louvre museum located?", "Где находится музей Лувр?",
    "In what year was the Mona Lisa created?", "В каком году была создана Мона Лиза?",
    "Identify the sculptor of David.", "Кто скульптор Давида?",
    "What art movement did Picasso belong to?", "К какому художественному направлению принадлежал Пикассо?",
    "Name the lead singer of Queen.", "Назови вокалиста группы Queen.",
    "Where was Mozart born?", "Где родился Моцарт?",
    
    # --- Medicine & Science ---
    "What is the recommended dose of Ibuprofen?", "Какова рекомендуемая доза Ибупрофена?",
    "What are the symptoms of dehydration?", "Каковы симптомы обезвоживания?",
    "Who discovered penicillin?", "Кто открыл пенициллин?",
    "What is the boiling point of nitrogen?", "Какова температура кипения азота?",
    "Identify the chemical formula of glucose.", "Назови химическую формулу глюкозы.",
    "How many planets are in the solar system?", "Сколько планет в солнечной системе?",
    "What is the distance from Earth to the Moon?", "Какое расстояние от Земли до Луны?",
    
    # --- History & Geography ---
    "When did the French Revolution start?", "Когда началась Французская революция?",
    "Who was the first president of the USA?", "Кто был первым президентом США?",
    "What is the capital of Japan?", "Какая столица у Японии?",
    "Name the longest river in Africa.", "Назови самую длинную реку в Африке.",
    "In what year did World War II end?", "В каком году закончилась Вторая мировая война?",
    "Identify the largest ocean on Earth.", "Укажи самый большой океан на Земле.",
    "Who built the Great Wall of China?", "Кто построил Великую Китайскую стену?",
    
    # --- Philosophy & General ---
    "Who wrote 'Thus Spoke Zarathustra'?", "Кто написал 'Так говорил Заратустра'?",
    "What is the population of New York City?", " какое население Нью-Йорка?",
    "Name the three branches of government.", "Назови три ветви власти.",
    "What is the official currency of Switzerland?", "Какая официальная валюта в Швейцарии?",
    "Who is the author of '1984'?", "Кто автор романа '1984'?",
    "Identify the CEO of Apple.", "Кто генеральный директор Apple?",
    "What is the height of Mount Everest?", "Какая высота у Эвереста?",
    
    # --- Short Fragments (The Hard ones) ---
    "Port number?", "Номер порта?",
    "Artist name.", "Имя художника.",
    "Start date.", "Дата начала.",
    "IP address.", "IP адрес.",
    "Dosage?", "Дозировка?",
    "Location?", "Местоположение?",
    "Budget amount.", "Сумма бюджета.",
    "Capital city.", "Столица.",
    "Author.", "Автор.",
    "Key?", "Ключ?",
    "Secret.", "Секрет.",
    "Version.", "Версия.",
    "Population.", "Население.",
    "Coordinates.", "Координаты.",
    "Price?", "Цена?",
    "Quantity.", "Количество."
] * 2 # Doubling to reach 100+
