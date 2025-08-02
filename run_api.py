import uvicorn
from src.api import app

# загружаем .env для локального запуска
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
