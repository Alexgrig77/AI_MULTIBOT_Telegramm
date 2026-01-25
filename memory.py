"""
Управление памятью диалогов
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

class MemoryManager:
    """Класс для управления памятью диалогов пользователей"""
    
    def __init__(self, memory_file: str = "memory.json"):
        self.memory_file = memory_file
        self.memory: Dict[str, Dict] = {}
        self.load_memory()
    
    def load_memory(self):
        """Загружает память из файла"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.memory = {}
        else:
            self.memory = {}
    
    def save_memory(self):
        """Сохраняет память в файл"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Ошибка сохранения памяти: {e}")
    
    def get_user_memory(self, user_id: str) -> Dict:
        """Получает память пользователя или создает новую"""
        if user_id not in self.memory:
            self.memory[user_id] = {
                "mode": "assistant",
                "history": []
            }
        return self.memory[user_id]
    
    def add_message(self, user_id: str, role: str, content: str):
        """Добавляет сообщение в историю пользователя"""
        user_memory = self.get_user_memory(user_id)
        user_memory["history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()
    
    def get_history(self, user_id: str, max_messages: int = 5) -> List[Dict]:
        """Получает последние N сообщений из истории"""
        user_memory = self.get_user_memory(user_id)
        history = user_memory.get("history", [])
        return history[-max_messages:] if len(history) > max_messages else history
    
    def clear_history(self, user_id: str):
        """Очищает историю диалога пользователя"""
        user_memory = self.get_user_memory(user_id)
        user_memory["history"] = []
        self.save_memory()
    
    def set_mode(self, user_id: str, mode: str):
        """Устанавливает режим (промпт) для пользователя"""
        user_memory = self.get_user_memory(user_id)
        user_memory["mode"] = mode
        self.save_memory()
    
    def get_mode(self, user_id: str) -> str:
        """Получает текущий режим пользователя"""
        user_memory = self.get_user_memory(user_id)
        return user_memory.get("mode", "assistant")
