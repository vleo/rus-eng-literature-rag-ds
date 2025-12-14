from typing import List, Dict, Any
import logging
import os
import glob

logger = logging.getLogger(__name__)

class TextLoader:
    """Загрузчик и парсер файлов в формате plain text"""
    
    @staticmethod
    def parse_text_file(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
        """
        Парсинг текстового файла и извлечение информации
        """
        try:
            # Пробуем разные кодировки если указанная не работает
            encodings = [encoding, 'utf-8', 'windows-1251', 'cp1251', 'koi8-r', 'iso-8859-1']
            
            content = None
            used_encoding = None
            
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        content = f.read()
                    used_encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                # Если все кодировки не сработали, используем utf-8 с игнорированием ошибок
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                used_encoding = 'utf-8 (with error ignore)'
            
            # Извлекаем базовые метаданные из имени файла
            metadata = TextLoader._extract_metadata(file_path, used_encoding)
            
            # Разбиваем содержимое на секции (например, по главам или абзацам)
            content_sections = TextLoader._extract_content(content, file_path)
            
            return {
                'metadata': metadata,
                'content': content_sections,
                'file_path': file_path
            }
            
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            return {}

    @staticmethod
    def _extract_metadata(file_path: str, encoding: str) -> Dict[str, str]:
        """Извлечение метаданных из текстового файла"""
        metadata = {
            'file_name': os.path.basename(file_path),
            'title': os.path.splitext(os.path.basename(file_path))[0],
            'author': '',
            'genre': 'text',
            'language': '',
            'year': '',
            'encoding': encoding
        }
        
        return metadata
    
    @staticmethod
    def _extract_content(content: str, file_path: str) -> List[Dict[str, Any]]:
        """Извлечение текстового содержимого из файла"""
        sections = []
        
        try:
            # Попробуем определить структуру документа
            lines = content.split('\n')
            
            # Определяем возможные заголовки глав (например, строки в верхнем регистре или начинающиеся с "Глава", "Chapter")
            section_patterns = [
                r'^\s*[Гг]лава\s+\d+',  # Глава N (Russian)
                r'^\s*[Cc]hapter\s+\d+',  # Chapter N (English)
                r'^\s*[Pp]art\s+\d+',  # Part N (English)
                r'^\s*[Чч]асть\s+\d+',  # Часть N (Russian)
                r'^\s*[Ss]ection\s+\d+',  # Section N
                r'^\s*[Tt]itle\s+\d+'  # Title N
            ]
            
            import re
            
            current_section = []
            section_title = "Основной текст"
            section_index = 0
            
            for line_num, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Проверяем, не является ли строка заголовком раздела
                is_section_header = False
                
                # Проверяем паттерны заголовков
                for pattern in section_patterns:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        # Сохраняем предыдущую секцию
                        if current_section:
                            text_content = '\n'.join(current_section).strip()
                            if text_content:
                                sections.append({
                                    'section_index': section_index,
                                    'text': text_content,
                                    'title': section_title
                                })
                                section_index += 1
                        
                        # Начинаем новую секцию
                        section_title = line_stripped
                        current_section = []
                        is_section_header = True
                        break
                
                # Если это не заголовок раздела, добавляем к текущей секции
                if not is_section_header and line_stripped:
                    current_section.append(line_stripped)
            
            # Добавляем последнюю секцию
            if current_section:
                text_content = '\n'.join(current_section).strip()
                if text_content:
                    sections.append({
                        'section_index': section_index,
                        'text': text_content,
                        'title': section_title
                    })
            
            # Если не найдено структуры, создаем один раздел со всем текстом
            if not sections:
                if content.strip():
                    sections.append({
                        'section_index': 0,
                        'text': content.strip(),
                        'title': 'Полный текст'
                    })
                    
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
        
        return sections

    @staticmethod
    def load_text_directory(directory_path: str, file_pattern: str = "*.txt") -> List[Dict[str, Any]]:
        """Загрузка всех текстовых файлов из директории"""
        # Поддерживаемые расширения
        text_extensions = ["*.txt", "*.text", "*.md", "*.markdown"]
        
        files = []
        for ext in text_extensions:
            files.extend(glob.glob(os.path.join(directory_path, ext)))
            files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        documents = []
        
        for file_path in files:
            logger.info(f"Processing text file: {file_path}")
            
            try:
                parsed_data = TextLoader.parse_text_file(file_path)
                
                if parsed_data and parsed_data['content']:
                    # Создаем документы для каждой секции
                    for section in parsed_data['content']:
                        if len(section['text']) > 50:  # Минимальная длина текста
                            documents.append({
                                'text': section['text'],
                                'metadata': {
                                    'file': parsed_data['metadata']['file_name'],
                                    'title': parsed_data['metadata']['title'],
                                    'author': parsed_data['metadata']['author'],
                                    'genre': parsed_data['metadata']['genre'],
                                    'year': parsed_data['metadata']['year'],
                                    'section_title': section['title'],
                                    'section_index': section['section_index'],
                                    'language': parsed_data['metadata']['language'],
                                    'source': 'text',
                                    'encoding': parsed_data['metadata'].get('encoding', 'unknown')
                                }
                            })
                    
                    logger.info(f"Successfully processed {file_path}: {len(parsed_data['content'])} sections")
                else:
                    logger.warning(f"No content extracted from {file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        return documents